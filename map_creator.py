from dataclasses import dataclass
from enum import Enum, auto
import re
from typing import List, Optional
import uuid
import pandas as pd
import json
import graphviz

TASK_REPORT_PATH = r"Inventories Local_Task_Report_INT23_05_Oct_2022\00_00_Supertask_051016540.csv"
MAIN_STAT_ACT = "Inventories Local"
DRAWIO_CSV_HEADERS = ['id', 'name', 'info', 'cluster', 'type', 'edges']


class ObjectType(Enum):
    """Classification of all known object types"""
    BULK = auto()
    SURVEY = auto()
    DELETE = auto()
    COPY_DATASETS = auto()
    AGGREGATE = auto()
    PRORATE = auto()
    FORMULA = auto()


def node_style(t: ObjectType) -> str:
    """Returns the node styling based on Object Type"""
    if t in [ObjectType.BULK, ObjectType.SURVEY]:
        return {'color': 'plum', 'style': 'filled'}
    return {'shape': 'box'}


class Register():
    """"""
    clusters: dict  # Dictionary of clusters
    objects: List  # List of refined objects

    def __init__(self) -> None:
        self.clusters = dict()
        self.objects = []
        self.clusters[MAIN_STAT_ACT] = Cluster(self)

    def register_internal(self, robj):
        """"""
        # if cluster doesn't exist create it
        if not self.clusters.get(robj.dataset):
            self.clusters[robj.dataset] = Cluster(self)
        # import definitions go in the stat act and not the import dataset
        if robj.obj_type in [ObjectType.BULK, ObjectType.SURVEY]:
            self.clusters[MAIN_STAT_ACT].add_object(robj)
        else:  # otherwise place the object in it's dataset
            cluster = self.clusters[robj.dataset]
            cluster.add_object(robj)
            self.clusters[MAIN_STAT_ACT].add_cluster(cluster)
        self.objects.append(robj)


class Cluster():
    """"""
    ID: str
    register: Register
    objects: List
    children: List

    def __init__(self, register) -> None:
        self.register = register
        self.ID = 'cluster_'+uuid.uuid4().hex
        self.objects = []
        self.children = []
        self.extra_edges = []

    @property
    def nodes(self):
        """"""
        nodes = []
        for robj in self.objects:
            node = dict()
            node['id'] = robj.id
            node['label'] = robj.name
            node['style'] = node_style(robj.obj_type)
            nodes.append(node)
        if nodes == []:  # add a placeholder node if there are no nodes
            nodes = [{'id': self.ID+'_ph',
                      'label': '',
                      'style': {'style': 'invis'}}]
        return nodes

    @property
    def edges(self):
        """"""
        edges = []
        extra_edges = []
        for i, robj in enumerate(self.objects):
            # special case for imports as I want the arrow to go to the cluster
            if robj.obj_type in [ObjectType.BULK, ObjectType.SURVEY]:
                try:  # use the id of the first node
                    first_node = self.register.clusters[robj.dataset].nodes[0]['id']
                except IndexError:  # if there are no nodes then reference a placeholder
                    first_node = self.register.clusters[robj.dataset].ID+'_ph'
                edges.append((
                    robj.id,
                    first_node,
                    '',
                    self.register.clusters[robj.dataset].ID
                ))
            # same as above for copy calcs
            if robj.obj_type in [ObjectType.COPY_DATASETS]:
                extra_edges.append((
                    robj.id,
                    self.register.clusters[robj.raw['Dataset']].nodes[0]['id'],
                    '',
                    self.register.clusters[robj.raw['Dataset']].ID
                ))
            if i+1 < len(self.objects) and robj.obj_type not in [ObjectType.BULK, ObjectType.SURVEY]:
                edges.append((robj.id, self.objects[i+1].id, '', ''))
        return edges, extra_edges

    def add_object(self, robj):
        """"""
        self.objects.append(robj)

    def add_cluster(self, cluster):
        """"""
        self.children.append(cluster)


@dataclass
class RefinedObject():  # TODO tidy up the elif mess?
    """"""
    raw: dict  # raw object in dictionary format
    reg: Register  # the object register

    def __post_init__(self):
        """Runs immediately after init"""
        reg.register_internal(self)

    @property
    def id(self):
        """Returns object id"""
        return self.raw['Order']

    @property
    def name(self):
        """Returns object name"""
        return self.raw['Name']

    @property
    def info(self):
        """Returns the assigned info as a string per object type"""
        obj_type = self.raw['Object Type']
        if obj_type == 'bulk_import':
            return '<br>'+('<br>').join(self.raw['Object Details'].values())
        elif obj_type == 'survey_import':
            return '<br>'+self.raw['Object Details']['description']
        elif obj_type in ['DELETE DATA', 'COPY BETWEEN DATASETS']:
            sel_crit = []
            for key, val in self.raw['Selection Criteria'].items():
                sel_crit.append(key + ' = ' + val)
            return '<br>'+('<br>').join(sel_crit)
        elif obj_type == 'FORMULA':
            details = 'Items Calculated: ['
            for key, val in self.raw['Object Details']['items to calculate'].items():
                details += key + ": " + val + ", "
            details = details[:-2]
            details += ']<br>Formula: '
            details += self.raw['Object Details']['formula']
            return '<br>'+details
        elif obj_type in ['AGGREGATE', 'PRORATE']:
            details = []
            for key, val in self.raw['Object Details'].items():
                details.append(key + ': ' + val)
            return '<br>'+('<br>').join(details)
        else:
            raise Exception("Unknown object type: " + obj_type)

    @property
    def dataset(self):
        """Returns the objects holding dataset"""
        dataset = self.raw['Dataset']
        if self.raw['Object Type'] == 'COPY BETWEEN DATASETS':
            dataset = self.raw['Object Details']['source dataset']['name']
        return dataset

    @property
    def obj_type(self):
        """Returns the assigned type as a string per object type"""
        obj_type = self.raw['Object Type']
        if obj_type == 'bulk_import':
            return ObjectType.BULK
        elif obj_type == 'survey_import':
            return ObjectType.SURVEY
        elif obj_type == 'DELETE DATA':
            return ObjectType.DELETE
        elif obj_type == 'COPY BETWEEN DATASETS':
            return ObjectType.COPY_DATASETS
        elif obj_type == 'FORMULA':
            return ObjectType.FORMULA
        elif obj_type == 'AGGREGATE':
            return ObjectType.AGGREGATE
        elif obj_type == 'PRORATE':
            return ObjectType.PRORATE
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


def create_maps(reg: Register):
    """Create the graphviz map"""

    def add_nodes(graph: graphviz.Digraph, cluster: Cluster):
        """Adds all nodes from the cluster to the graph"""
        for node in cluster.nodes:
            graph.node(node['id'], node['label'], **
                       node['style'])

    def add_edges(graph: graphviz.Digraph, cluster: Cluster, extras: Optional[List] = []):
        """Adds all nodes from the cluster to the graph"""
        own, parent = cluster.edges
        for (tail, head, ltail, lhead) in own:
            graph.edge(tail, head, ltail=ltail, lhead=lhead)
        for (tail, head, ltail, lhead) in extras:
            graph.edge(tail, head, ltail=ltail, lhead=lhead)
        return parent

    g = graphviz.Digraph('G', filename=f'{MAIN_STAT_ACT}.dot')
    g.attr(compound='true')
    main_cluster = reg.clusters.pop(MAIN_STAT_ACT)

    with g.subgraph(name=main_cluster.ID) as main:
        main.attr(label=MAIN_STAT_ACT)
        extra_edges = []
        for label, cluster in reg.clusters.items():
            with (main.subgraph(name=cluster.ID)) as c:
                c.attr(label=label)
                extra_edges += add_edges(c, cluster)
                add_nodes(c, cluster)
        # add main level edges after children since children add extra edges
        add_nodes(main, main_cluster)
        add_edges(main, main_cluster, extra_edges)

        # import_nodes = [robj for robj in reg.objects if robj.obj_type in [
        #     ObjectType.BULK, ObjectType.SURVEY]]
        # with main.subgraph() as s:
        #     s.attr(rank='same')
        #     [s.node(node.id) for node in import_nodes]

    g.save()


if __name__ == "__main__":
    tr = pd.read_csv(TASK_REPORT_PATH, encoding='unicode_escape', skiprows=4,
                     keep_default_na=False)
    # get rid of the second header line
    tr.drop([0], inplace=True)
    # get rid of sub tasks and con checks from the task report
    tr = tr[tr['Object Type'] != 'SUB TASK']
    tr = tr[tr['Kind'] != 'CONSISTENCY CHECK DEFINITION']

    # get a list of the objects as dicts
    objs = convert_objects(tr)
    # create register of refined objects
    reg = Register()
    [RefinedObject(obj, reg) for obj in objs]

    create_maps(reg)

    # TODO sort out styling of elements and figure out how to add extra info
