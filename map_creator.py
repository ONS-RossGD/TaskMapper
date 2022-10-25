from dataclasses import dataclass, field
from enum import Enum, auto
import re
import sys
from typing import List, Tuple
import uuid
import pandas as pd
import json
import graphviz
from PyQt5.QtWidgets import QApplication, QFileDialog

# helpful graphviz video here: https://www.youtube.com/watch?v=MR6-TYelgkI


class ObjectType(Enum):
    """Classification of all known object types"""
    BULK = auto()
    SURVEY = auto()
    DELETE = auto()
    COPY = auto()
    AGGREGATE = auto()
    PRORATE = auto()
    FORMULA = auto()


@dataclass
class Edge():
    """"""
    tail: str
    head: str
    ltail: str
    lhead: str
    tooltip: str


@dataclass
class Node():
    """"""
    ID: str
    label: str
    style: str


def node_style(t: ObjectType) -> str:
    """Returns the node styling based on Object Type"""
    if t in [ObjectType.BULK, ObjectType.SURVEY]:
        return {'color': 'plum', 'shape': 'folder', 'style': 'filled'}
    if t in [ObjectType.DELETE]:
        return {'color': '#ff7663', 'style': 'filled', 'shape': 'box'}
    if t in [ObjectType.AGGREGATE]:
        return {'color': '#dedede', 'style': 'filled', 'shape': 'house'}
    if t in [ObjectType.PRORATE]:
        return {'color': '#dedede', 'style': 'filled', 'shape': 'invhouse'}
    if t in [ObjectType.FORMULA]:
        return {'color': '#dedede', 'style': 'filled', 'shape': 'parallelogram'}
    if t in [ObjectType.COPY]:
        return {'color': '#bafcff', 'style': 'filled', 'shape': 'box'}
    return {'shape': 'box'}


@dataclass
class Register():
    """"""
    hierarchy: dict  # hierarchy of stat acts, datasets and objects
    objects: dict  # dictionary of refined objects with id as key
    clusters: dict = field(default_factory=dict)  # init's clusters dict

    def __post_init__(self):
        """Returns a dictionary of all clusters"""
        for sa_label, sa_dict in self.hierarchy.items():
            self.clusters[sa_label] = Cluster(sa_label,
                                              self, [obj for obj_id, obj in self.objects.items() if obj_id in sa_dict['objects']])
            for dataset_label, obj_ids in sa_dict['datasets'].items():
                self.clusters[sa_label+'/'+dataset_label] = Cluster(sa_label+'/'+dataset_label,
                                                                    self, [obj for obj_id, obj in self.objects.items() if obj_id in obj_ids])


@dataclass
class RefinedObject():
    """"""
    raw: dict  # raw object in dictionary format

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
        if self.obj_type in [ObjectType.BULK]:
            return '<br/>'+('<br/>').join(self.raw['Object Details'].values())
        if self.obj_type in [ObjectType.SURVEY]:
            return '<br/>'+self.raw['Object Details']['description']
        if self.obj_type in [ObjectType.DELETE, ObjectType.COPY]:
            sel_crit = []
            for key, val in self.raw['Selection Criteria'].items():
                sel_crit.append(key + ' = ' + val)
            return '<br/>'+('<br/>').join(sel_crit)
        if self.obj_type in [ObjectType.FORMULA]:
            details = 'Items Calculated: ['
            for key, val in self.raw['Object Details']['items to calculate'].items():
                details += key + ": " + val + ", "
            details = details[:-2]
            details += ']<br/>Formula: '
            details += self.raw['Object Details']['formula']
            return '<br/>'+details
        if self.obj_type in [ObjectType.AGGREGATE, ObjectType.PRORATE]:
            details = []
            for key, val in self.raw['Object Details'].items():
                details.append(key + ': ' + val)
            return '<br/>'+('<br/>').join(details)

    @property
    def label(self):
        """Returns an html label"""
        replace_html = {'&': '&amp;',
                        '<': '&lt;',
                        '>': '&quot;',
                        "'": '&#39;'
                        }
        name = self.name
        for key, val in replace_html.items():
            name = name.replace(key, val)
        return f"<{name}<font color='#636363'>{self.info}</font>>"

    @property
    def dataset(self):
        """Returns the objects holding dataset"""
        dataset = self.raw['Dataset']
        # if self.obj_type in [ObjectType.COPY]:
        #     dataset = self.raw['Object Details']['source dataset']['name']
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
        elif obj_type in ['COPY BETWEEN DATASETS', 'COPY BETWEEN MODES']:
            return ObjectType.COPY
        elif obj_type == 'FORMULA':
            return ObjectType.FORMULA
        elif obj_type == 'AGGREGATE':
            return ObjectType.AGGREGATE
        elif obj_type == 'PRORATE':
            return ObjectType.PRORATE
        else:
            raise Exception("Unknown object type: " + obj_type)


@dataclass
class Cluster():
    """"""
    path: str
    reg: Register
    objects: List[RefinedObject]
    ID: str = field(default_factory=lambda: 'cluster_' +
                    uuid.uuid4().hex)  # auto-generate a unique id

    @property
    def nodes(self):
        """"""
        nodes = []
        for robj in self.objects:
            node = Node(robj.id, robj.label, node_style(robj.obj_type))
            nodes.append(node)
        if nodes == []:  # add a placeholder node if there are no nodes
            nodes = [Node(self.ID+'_ph', '', {'style': 'invis'})]
        return {self.path: nodes}

    @property
    def edges(self):
        """"""

        def _add_edge(d: dict, key: str, val: Edge):
            if not d.get(key):
                d[key] = []
            d[key] += [val]

        edges = dict()
        for i, robj in enumerate(self.objects):
            # special case for imports as I want the arrow to go to the cluster
            if robj.obj_type in [ObjectType.BULK, ObjectType.SURVEY]:
                sa, dataset = split_dataset_stat_act(robj.raw['Dataset'])
                cluster_ref = sa+'/'+dataset
                try:  # use the id of the first node
                    first_node = self.reg.clusters[cluster_ref].nodes[cluster_ref][0].ID
                except IndexError:  # if there are no nodes then reference a placeholder
                    first_node = self.reg.clusters[cluster_ref].ID+'_ph'
                _add_edge(edges, self.path.split('/')[0], Edge(
                    robj.id,
                    first_node,
                    '',
                    self.reg.clusters[cluster_ref].ID,
                    robj.name + ' -> ' + cluster_ref
                ))
            # same as above for copy calcs
            if robj.obj_type in [ObjectType.COPY]:
                sa, dataset = split_dataset_stat_act(
                    robj.raw['Object Details']['source dataset']['name'])
                cluster_ref = sa+'/'+dataset
                try:
                    last_node = self.reg.clusters[cluster_ref].nodes[cluster_ref][-1].ID
                except IndexError:
                    last_node = self.reg.clusters[cluster_ref].ID+'_ph'
                # if the copy calc is copying to the same dataset
                if cluster_ref == self.path:
                    _add_edge(edges, cluster_ref, Edge(
                        robj.id, robj.id, '', '', robj.name + ' -> ' + robj.name))
                # if the source and target datasets are from the same stat act
                elif cluster_ref.split('/')[0] == self.path.split('/')[0]:
                    # edge at sa level
                    _add_edge(edges, cluster_ref.split('/')[0], Edge(
                        last_node,
                        robj.id,
                        self.reg.clusters[cluster_ref].ID,
                        '',
                        cluster_ref + ' -> ' + robj.name
                    ))
                else:
                    # edge at root level
                    _add_edge(edges, 'root', Edge(
                        last_node,
                        robj.id,
                        self.reg.clusters[cluster_ref].ID,
                        '',
                        cluster_ref + ' -> ' + robj.name
                    ))

            if i+1 < len(self.objects) and robj.obj_type not in [ObjectType.BULK, ObjectType.SURVEY]:
                _add_edge(edges, self.path,
                          Edge(robj.id, self.objects[i+1].id, '', '', robj.name + ' -> ' + self.objects[i+1].name))
        return edges


def convert_objects(tr: pd.DataFrame) -> pd.DataFrame():
    """Converts all objects in a task report into a map dataframe"""
    def jsonify(s) -> dict:
        """Convert the awkwardly formatted CORD strings to a json string and
        returns the json as a dict"""
        s = s.replace("'", "")
        s = re.sub('{([^{}]+) = ([^{}]+)}', r'"\1": "\2"', s)
        s = re.sub(
            '({.*?)([^ ]*?\"classification mapping\": \")(.*?)(\")(})', r'\1\3\5', s)
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


def split_dataset_stat_act(dataset: str) -> Tuple[str, str]:
    sa_found = re.search('(.*?)( ?)\(from (.*)\)', dataset)
    sa = MAIN_STAT_ACT
    if sa_found:  # if an external stat act is found use it and seperate the dataset
        dataset = sa_found.group(1)
        sa = sa_found.group(3)
    if '/' in sa:
        split = sa.split('/')
        sa = split[0] + ' (Mode: ' + split[1] + ')'
    return sa, dataset


def get_cluster_hierarchy(objs: List):
    """Creates the clusters for later mapping"""
    main_sa = MAIN_STAT_ACT
    clusters = dict({main_sa: {'datasets': {}, 'objects': []}})

    for obj in objs:
        obj_id = obj['Order']
        # copy calcs can reference a source dataset which will need a cluster
        # however we don't want the copy calc to sit in the source dataset so
        # the obj_id is not added
        if obj['Object Type'] in ["COPY BETWEEN DATASETS", "COPY BETWEEN MODES"]:
            sa, dataset = split_dataset_stat_act(
                obj['Object Details']['source dataset']['name'])
            # Extra substitution as copy between modes has "/" which conflicts with path
            dataset = re.sub(
                '(.*)(\(from )([^\/]*)\/(.*?)\)', r'\1\2\3 (\4))', dataset)
            if not clusters.get(sa):  # if sa doesn't exist yet create it and add dataset
                clusters[sa] = {'datasets': {}, 'objects': []}
                clusters[sa]['datasets'] = {dataset: []}
            # otherwise just add dataset
            elif not clusters[sa]['datasets'].get(dataset):
                clusters[sa]['datasets'][dataset] = []
        # import definitions should go directly in sa not in dataset
        if obj['Object Type'] in ["bulk_import", "survey_import"]:
            if not clusters.get(main_sa):
                clusters[main_sa] = {'datasets': {}, 'objects': []}
            clusters[main_sa]['objects'] += [obj_id]
            # need to remember to add the dataset for reference though
            if not clusters[main_sa]['datasets'].get(obj['Dataset']):
                clusters[main_sa]['datasets'][obj['Dataset']] = []
        # otherwise make sure the cluster exists and the object is added
        else:
            if not clusters.get(main_sa):
                clusters[main_sa] = {'datasets': {}, 'objects': []}
                clusters[main_sa]['datasets'] = {obj['Dataset']: [obj_id]}
            elif not clusters[main_sa]['datasets'].get(obj['Dataset']):
                clusters[main_sa]['datasets'][obj['Dataset']] = [obj_id]
            else:
                clusters[main_sa]['datasets'][obj['Dataset']] += [obj_id]
    return clusters


def create_map(reg: Register, save_path: str):
    """"""
    nodes = dict()
    edges = dict()

    if save_path[-4:] != '.dot':
        save_path += '.dot'

    def add_nodes(n: dict):
        for key, vals in n.items():
            if not nodes.get(key):
                nodes[key] = []
            nodes[key] += vals

    def add_edges(e: dict):
        for key, val in e.items():
            if not edges.get(key):
                edges[key] = []
            edges[key] += val

    g = graphviz.Digraph(MAIN_STAT_ACT,
                         filename=save_path)
    g.attr(compound='true')
    # g.attr(rankdir='LR')
    # g.attr(splines='false')

    # merge dicts of nodes and edges for all clusters
    for cluster in reg.clusters.values():
        add_nodes(cluster.nodes)
        add_edges(cluster.edges)

    for path, node_list in nodes.items():
        split_path = path.split('/')
        stat_act = ''
        dataset = ''
        if len(split_path) == 1:
            stat_act = split_path[0]
        elif len(split_path) == 2:
            stat_act = split_path[0]
            dataset = split_path[1]
        else:
            raise Exception('Unrecognised path', path)
        sa_cluster = reg.clusters[stat_act]
        with g.subgraph(name=sa_cluster.ID) as sa:
            sa.attr(label=stat_act)
            if dataset:
                dataset_cluster = reg.clusters[path]
                with sa.subgraph(name=dataset_cluster.ID) as d:
                    d.attr(label=dataset)
                    for node in node_list:
                        d.node(node.ID, node.label, node.style)
            else:
                for node in node_list:
                    sa.node(node.ID, node.label, node.style)

    # cover the root edges first
    if 'root' in edges.keys():
        for edge in edges.pop('root'):
            g.edge(edge.tail, edge.head, ltail=edge.ltail,
                   lhead=edge.lhead, edgetooltip=edge.tooltip)

    for path, edge_list in edges.items():
        split_path = path.split('/')
        stat_act = ''
        dataset = ''
        if len(split_path) == 1:
            stat_act = split_path[0]
        elif len(split_path) == 2:
            stat_act = split_path[0]
            dataset = split_path[1]
        else:
            raise Exception('Unrecognised path')
        sa_cluster = reg.clusters[stat_act]
        with g.subgraph(name=sa_cluster.ID) as sa:
            sa.attr(label=stat_act)
            if dataset:
                dataset_cluster = reg.clusters[path]
                with sa.subgraph(name=dataset_cluster.ID) as d:
                    d.attr(label=dataset)
                    for edge in edge_list:
                        d.edge(edge.tail, edge.head,
                               ltail=edge.ltail, lhead=edge.lhead, edgetooltip=edge.tooltip)
            else:
                for edge in edge_list:
                    sa.edge(edge.tail, edge.head,
                            ltail=edge.ltail, lhead=edge.lhead, edgetooltip=edge.tooltip)

    g.save()


if __name__ == "__main__":
    global MAIN_STAT_ACT

    app = QApplication(sys.argv)
    task_rpt_zip, _ = QFileDialog.getOpenFileName(
        None, 'Select task report zip file...', '', '*.zip')
    if not task_rpt_zip:
        raise Exception('Task report zip not selected')

    MAIN_STAT_ACT = pd.read_csv(
        task_rpt_zip,
        encoding='unicode_escape',
        header=None,
        skiprows=2,
        nrows=1
    ).iloc[0, 0].split('Statistical Activity: ')[-1].split('Mode')[0].strip()
    tr = pd.read_csv(task_rpt_zip, encoding='unicode_escape', skiprows=4,
                     keep_default_na=False)
    # get rid of the second header line
    tr.drop([0], inplace=True)
    # get rid of sub tasks, dummy tasks and con checks from the task report
    tr = tr[tr['Object Type'] != 'SUB TASK']
    tr = tr[tr['Object Type'] != 'n/a']
    tr = tr[tr['Kind'] != 'CONSISTENCY CHECK DEFINITION']

    # get a list of the objects as dicts
    objs = convert_objects(tr)
    # create dict of refined objects with object id as key
    robjs = dict()
    for obj in objs:
        robj = RefinedObject(obj)
        robjs[robj.id] = robj

    reg = Register(get_cluster_hierarchy(objs), robjs)

    save_path, _ = QFileDialog.getSaveFileName(
        None, 'Choose save location...', '')
    if not save_path:
        raise Exception('Must choose save location')

    create_map(reg, save_path)

    # TODO sort out the / in path for copy between modes calc - change in regex?
