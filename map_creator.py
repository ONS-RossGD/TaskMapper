from dataclasses import dataclass
from enum import Enum, auto
import re
from typing import List
import pandas as pd
import json

TASK_REPORT_PATH = r"Inventories Local_Task_Report_INT23_05_Oct_2022\00_00_Supertask_051016540.csv"
STAT_ACT = "Inventories Local"
DRAWIO_CSV_HEADERS = ['id', 'name', 'info', 'parent', 'shape', 'links']


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
            return ('<br>').join(self.RAW['Object Details'].values())
        elif obj_type == 'survey_import':
            return self.RAW['Object Details']['description']
        elif obj_type in ['DELETE DATA', 'COPY BETWEEN DATASETS']:
            sel_crit = []
            for key, val in self.RAW['Selection Criteria'].items():
                sel_crit.append(key + ' = ' + val)
            return ('<br>').join(sel_crit)
        elif obj_type == 'FORMULA':
            details = 'Items Calculated: '
            for key, val in self.RAW['Object Details']['items to calculate'].items():
                details += key + ": " + val + "<br>"
            details += 'Formula: '
            details += self.RAW['Object Details']['formula']
            return details
        elif obj_type in ['AGGREGATE', 'PRORATE']:
            details = []
            for key, val in self.RAW['Object Details'].items():
                details.append(key + ': ' + val)
            return ('<br>').join(details)
        else:
            raise Exception("Unknown object type: " + obj_type)

    @property
    def parent(self):
        """Returns the parent"""
        if self.RAW['Object Type'] in ['COPY BETWEEN DATASETS',
                                       'bulk_import', 'survey_import']:
            return STAT_ACT
        else:
            return self.RAW['Dataset']

    @property
    def shape(self):
        """Returns the assigned shape as a string per object type"""
        obj_type = self.RAW['Object Type']
        if obj_type in ['bulk_import', 'survey_import']:
            return 'note'
        elif obj_type == 'DELETE DATA':
            return 'sum'
        elif obj_type == 'COPY BETWEEN DATASETS':
            return 'rounded rectangle'
        elif obj_type == 'FORMULA':
            return 'parallelogram'
        elif obj_type == 'AGGREGATE':
            return 'trapezoid'
        elif obj_type == 'PRORATE':
            return 'document'
        else:
            raise Exception("Unknown object type: " + obj_type)

    @property
    def links(self):
        """Returns the assigned shape as a string per object type"""
        obj_type = self.RAW['Object Type']
        if obj_type in ['bulk_import', 'survey_import', 'COPY BETWEEN DATASETS']:
            link = [ho['id']
                    for ho in hobjs if ho['name'] == self.RAW['Dataset']]
            if len(link) > 1:
                raise Exception('Too many links')
            return '"'+str(link[0])+'"'
        elif obj_type in ['DELETE DATA', 'FORMULA', 'AGGREGATE', 'PRORATE']:
            link = '""'
            dr = dataset_register[dataset_register['Dataset']
                                  == self.RAW['Dataset']].reset_index()
            idx = dr[dr['Name'].str.strip() == self.RAW['Name']
                     ].index.tolist()[0]
            if idx != dr.index.tolist()[-1]:
                link = '"' + str(dr.iloc[idx+1, 0]) + '"'
            return link
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
    hobjs.append({'name': STAT_ACT, 'id': 'sa', 'parent': '-',
                 'shape': 'container', 'links': '""'})
    # compile a complete set of datasets and links
    datasets = set()
    links = dict()
    for obj in objs:
        datasets.add(obj['Dataset'])
        if obj['Object Type'] != 'COPY BETWEEN DATASETS':
            continue
        name = obj['Object Details']['source dataset']['name']
        datasets.add(name)
        try:
            links[name].append(obj['Order'])
        except KeyError:
            links[name] = [obj['Order']]
    # fill in the links with blanks for dataframes without links
    for name in datasets.difference(links.keys()):
        links[name] = ''

    # add datasets to the holding objects
    for i, dataset in enumerate(datasets):
        hobjs.append({'name': dataset, 'id': 'd'+str(i), 'parent': STAT_ACT,
                     'shape': 'horizontal container', 'links': '"' + (',').join(links[dataset]) + '"'})


def create_drawio_df(objs: List[RefinedObject]) -> pd.DataFrame:
    """Creates and returns the dataframe for the holding objects"""
    df = pd.DataFrame(columns=DRAWIO_CSV_HEADERS)
    for ho in hobjs:
        r = len(df.index.tolist())
        df.loc[r, 'id'] = ho['id']
        df.loc[r, 'name'] = ho['name']
        df.loc[r, 'parent'] = ho['parent']
        df.loc[r, 'shape'] = ho['shape']
        df.loc[r, 'links'] = ho['links']
    for obj in objs:
        r = len(df.index.tolist())
        df.loc[r, 'id'] = obj.id
        df.loc[r, 'info'] = obj.info
        df.loc[r, 'name'] = obj.name
        df.loc[r, 'parent'] = obj.parent
        df.loc[r, 'shape'] = obj.shape
        df.loc[r, 'links'] = obj.links
    return df


if __name__ == "__main__":
    tr = pd.read_csv(TASK_REPORT_PATH, encoding='unicode_escape', skiprows=4,
                     keep_default_na=False)
    # get rid of the second header line
    tr.drop([0], inplace=True)
    # get rid of sub tasks and con checks from the task report
    tr = tr[tr['Object Type'] != 'SUB TASK']
    tr = tr[tr['Kind'] != 'CONSISTENCY CHECK DEFINITION']

    global dataset_register
    dataset_register = tr[['Order', 'Name', 'Dataset']]

    # get a list of the objects as dicts
    objs = convert_objects(tr)
    # register the holding objects (Stat Act & Datasets)
    register_holding_objs(objs)
    # refine the objects ready for drawio
    robjs = [RefinedObject(obj) for obj in objs]
    # create the drawio csv dataset
    dio_csv = create_drawio_df(robjs)
