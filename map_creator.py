from cProfile import label
from dataclasses import dataclass
from enum import Enum, auto
import re
import shutil
from typing import List
import pandas as pd
import json
import graphviz

TASK_REPORT_PATH = r"Inventories Local_Task_Report_INT23_05_Oct_2022\00_00_Supertask_051016540.csv"
STAT_ACT = "Inventories Local"
DRAWIO_CSV_HEADERS = ['id', 'name', 'info', 'group', 'type', 'edges']


class ObjTypes(Enum):
    """Classification of all known object types"""
    BULK = auto()
    SURVEY = auto()
    DELETE = auto()
    COPY_DATASETS = auto()
    AGGREGATE = auto()
    PRORATE = auto()
    FORMULA = auto()


@dataclass
class RefinedObject():  # TODO tidy up the elif mess?
    """"""
    RAW: dict

    @property
    def id(self):
        """Returns object id"""
        return self.RAW['Order']

    @property
    def name(self):
        """Returns object name"""
        return self.RAW['Name']

    @property
    def info(self):
        """Returns the assigned info as a string per object type"""
        obj_type = self.RAW['Object Type']
        if obj_type == 'bulk_import':
            return '<br>'+('<br>').join(self.RAW['Object Details'].values())
        elif obj_type == 'survey_import':
            return '<br>'+self.RAW['Object Details']['description']
        elif obj_type in ['DELETE DATA', 'COPY BETWEEN DATASETS']:
            sel_crit = []
            for key, val in self.RAW['Selection Criteria'].items():
                sel_crit.append(key + ' = ' + val)
            return '<br>'+('<br>').join(sel_crit)
        elif obj_type == 'FORMULA':
            details = 'Items Calculated: ['
            for key, val in self.RAW['Object Details']['items to calculate'].items():
                details += key + ": " + val + ", "
            details = details[:-2]
            details += ']<br>Formula: '
            details += self.RAW['Object Details']['formula']
            return '<br>'+details
        elif obj_type in ['AGGREGATE', 'PRORATE']:
            details = []
            for key, val in self.RAW['Object Details'].items():
                details.append(key + ': ' + val)
            return '<br>'+('<br>').join(details)
        else:
            raise Exception("Unknown object type: " + obj_type)

    @property
    def group(self):
        """Returns the group"""
        if self.RAW['Object Type'] in ['COPY BETWEEN DATASETS',
                                       'bulk_import', 'survey_import']:
            return 'sa'
        else:
            g = [ho['id']
                 for ho in hobjs if ho['name'] == self.RAW['Dataset']]
            if len(g) > 1:
                raise Exception('Too many matching datasets')
            return g[0]

    @property
    def obj_type(self):
        """Returns the assigned type as a string per object type"""
        obj_type = self.RAW['Object Type']
        if obj_type in ['bulk_import', 'survey_import']:
            return 'import'
        elif obj_type == 'DELETE DATA':
            return 'delete'
        elif obj_type == 'COPY BETWEEN DATASETS':
            return 'copy'
        elif obj_type == 'FORMULA':
            return 'formula'
        elif obj_type == 'AGGREGATE':
            return 'aggregate'
        elif obj_type == 'PRORATE':
            return 'prorate'
        else:
            raise Exception("Unknown object type: " + obj_type)

    @property
    def edges(self):
        """Returns the list of edges"""
        obj_type = self.RAW['Object Type']
        if obj_type in ['bulk_import', 'survey_import']:
            edge = [(self.id, f'cluster_{ho["id"]}')
                    for ho in hobjs if ho['name'] == self.RAW['Dataset']]
            if len(edge) > 1:
                raise Exception('Too many edges')
            return edge
        elif obj_type in ['COPY BETWEEN DATASETS']:
            sd = self.RAW["Object Details"]["source dataset"]["name"]
            edges = [(f"cluster_{[ho['id'] for ho in hobjs if ho['name'] == sd][0]}_out", self.id),
                     (self.id, f"cluster_{[ho['id'] for ho in hobjs if ho['name'] == self.RAW['Dataset']][0]}")]
            print(edges)
            return edges
        elif obj_type in ['DELETE DATA', 'FORMULA', 'AGGREGATE', 'PRORATE']:
            edge = []
            dr = dataset_register[dataset_register['Dataset']
                                  == self.RAW['Dataset']].reset_index()
            idx = dr[dr['Name'].str.strip() == self.RAW['Name']
                     ].index.tolist()[0]
            if idx != dr.index.tolist()[-1]:
                edge.append((self.id, str(dr.iloc[idx+1, 0])))
            return edge
        else:
            raise Exception("Unknown object type: " + obj_type)


def convert_objects(tr: pd.DataFrame) -> pd.DataFrame():
    """Converts all objects in a task report into a map dataframe"""
    def jsonify(s) -> dict:
        """Convert the awkwardly formatted CORD strings to a json string and
        returns the json as a dict"""
        s = s.replace("'", "")
        s = re.sub('{([^{}]+) = ([^{}]+)}', r'"\1": "\2"', s)
        s = re.sub('{([^{}]+)-> (\(.*?\)) ?([^{}]+)}', r'"\1": "\3 \2"', s)
        s = re.sub('{([^{}]*?)( ?\")', r'"\1": {"', s)
        s = re.sub('(\"[} ]?)\"', r'\1, "', s)
        s = '{' + s + '}'
        # con checks: ^{(.*?( +)){(.*?)}
        return json.loads(s)

    objs = []
    # TODO looping over every cell - not optimal
    for i in tr.index.tolist():
        obj = dict()
        for c in tr.columns.tolist():
            # strip the leading and trailing whitespace from the cell
            val = str(tr.loc[i, c]).strip()
            # All awkwardly formatted cells have a { in them
            if '{' in val:  # convert that value to a dict
                obj[c] = jsonify(val)
            else:  # leave the value as a string
                obj[c] = val
        objs.append(obj)

    return objs


def register_holding_objs(objs: List) -> List[dict]:
    """Registers the holding objects and publishes to global
    variable hobjs"""
    global hobjs  # global not ideal but can't thing of a better way right now
    hobjs = []
    # add the stat act to the holding objects
    # hobjs.append({'name': STAT_ACT, 'id': 'sa', 'group': 'root',
    #              'type': 'sa', 'edges': []})
    # compile a complete set of datasets and edges
    datasets = set()
    edges = dict()
    ids = dict()
    for obj in objs:
        datasets.add(obj['Dataset'])
        if obj['Object Type'] != 'COPY BETWEEN DATASETS':
            continue
        name = obj['Object Details']['source dataset']['name']
        datasets.add(name)
        for i, d in enumerate(datasets):
            ids[d] = 'd'+str(i)
        try:
            edges[name].append((ids[name], obj['Order']))
        except KeyError:
            edges[name] = [(ids[name], obj['Order'])]
    # fill in the edges with blanks for dataframes without edges
    for name in datasets.difference(edges.keys()):
        edges[name] = []

    # add datasets to the holding objects
    for dataset in datasets:
        hobjs.append({'name': dataset, 'id': ids[dataset], 'group': 'sa',
                     'type': 'dataset', 'edges': edges[dataset]})


def create_obj_df(robjs: List[RefinedObject]) -> pd.DataFrame:
    """Creates and returns the object dataframe"""
    df = pd.DataFrame(columns=DRAWIO_CSV_HEADERS)
    # for ho in hobjs:
    #     r = len(df.index.tolist())
    #     df.loc[r, 'id'] = ho['id']
    #     df.loc[r, 'name'] = ho['name']
    #     df.loc[r, 'group'] = ho['group']
    #     df.loc[r, 'type'] = ho['type']
    #     df.loc[r, 'edges'] = ho['edges']
    for obj in robjs:
        r = len(df.index.tolist())
        df.loc[r, 'id'] = obj.id
        df.loc[r, 'info'] = obj.info
        df.loc[r, 'name'] = obj.name
        df.loc[r, 'group'] = obj.group
        df.loc[r, 'type'] = obj.obj_type
        df.loc[r, 'edges'] = obj.edges
    return df


def create_maps(robjs: List[RefinedObject]):
    """Create the graphviz map"""
    # create the drawio csv dataset
    objs_df = create_obj_df(robjs)
    print(objs_df.head())
    # print(objs_df.head(50))
    g = graphviz.Digraph('G', filename=f'{STAT_ACT}.dot')
    g.attr(compound='true')
    # first item will be the stat act
    # sa_obj = hobjs.pop(0)
    with (g.subgraph(name=f'cluster_{STAT_ACT}')) as sa:
        sa.attr(label=STAT_ACT)
        for ho in hobjs:
            with (sa.subgraph(name=f'cluster_{ho["id"]}')) as c:
                c.attr(label=ho['name'], rank='TB')
                c.node(f'cluster_{ho["id"]}', style='invis')
                try:
                    first = objs_df[objs_df['group'] == ho['id']]['id'].tolist()[
                        0]
                    last = objs_df[objs_df['group'] ==
                                   ho['id']]['id'].tolist()[-1]
                    c.edge(f'cluster_{ho["id"]}', first, style='invis')
                    c.edge(last, f'cluster_{ho["id"]}_out', style='invis')
                except IndexError:
                    c.edge(f'cluster_{ho["id"]}',
                           f'cluster_{ho["id"]}_out', style='invis')
                for _, r in objs_df[objs_df['group'] == ho['id']].iterrows():
                    c.node(r['id'], r['name'])
                    c.edges(r['edges'])
                c.node(f'cluster_{ho["id"]}_out', style='invis')
        for _, r in objs_df[objs_df['group'] == 'sa'].iterrows():
            sa.node(r['id'], r['name'])
            edges = r['edges']
            for (tail, head) in edges:
                ltail, lhead = '', ''
                if 'cluster_' in tail:
                    ltail = tail
                if 'cluster_' in head:
                    lhead = head
                sa.edge(tail, head, ltail=ltail, lhead=lhead)

        # TODO TODO TODO need to redo the edges. Might also be worth renaming to fit clusters/edges/nodes notation

    g.save()


if __name__ == "__main__":
    tr = pd.read_csv(TASK_REPORT_PATH, encoding='unicode_escape', skiprows=4,
                     keep_default_na=False)
    # get rid of the second header line
    tr.drop([0], inplace=True)
    # get rid of sub tasks and con checks from the task report
    tr = tr[tr['Object Type'] != 'SUB TASK']
    tr = tr[tr['Kind'] != 'CONSISTENCY CHECK DEFINITION']

    global dataset_register
    dataset_register = tr[tr['Object Type'] != 'bulk_import']
    dataset_register = dataset_register[dataset_register['Object Type']
                                        != 'survey_import']
    dataset_register = dataset_register[dataset_register['Object Type']
                                        != 'COPY BETWEEN DATASETS']
    dataset_register = dataset_register[['Order', 'Name', 'Dataset']]

    # get a list of the objects as dicts
    objs = convert_objects(tr)
    # register the holding objects (Stat Act & Datasets)
    register_holding_objs(objs)
    # refine the objects ready for drawio
    robjs = [RefinedObject(obj) for obj in objs]
    create_maps(robjs)
