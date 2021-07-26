""" Module implementing Call Graph representation of individual projects (programs).
Notably, the CG representation contains the so-called 'levels', where each function contained
in the CG is assigned a level value that represents an estimation of the longest acyclic path
from root (presumably main) to the function itself.
"""
from __future__ import annotations
import operator
from typing import Dict, List, Iterator

import loader as load


# Mapping CG node (function name) -> function details
NodeMap = Dict[str, 'CGNode']


# Directory with CG JSON files
CG_DIR = 'call_graphs'
# CG JSON filename prefix
CG_PREFIX = 'cg'


class CGNode:
    """ Call Graph Node structure. Each node represents a function of the program and contains
    the neccessary details about the function.

    :ivar name: the function name
    :ivar level: estimation of the maximum possible call stack depth of the function
    :ivar callers: a collection of the caller functions, i.e., functions that can call this node
    :ivar callees: a collection of the callee functions, 
                   i.e., functions that can be called from this node
    """
    __slots__ = 'name', 'level', 'callers', 'callees'
    
    def __init__(self, json_node: load.JsonType) -> None:
        """ Constructor

        :param json_node: Node information in the JSON format
        """
        self.name: str = json_node['name']
        self.level: int = int(json_node['level'])
        # Callers and Callees are initialized later
        self.callers: NodeMap = {}
        self.callees: NodeMap = {}


class CallGraph:
    """ Call Graph structure containing function nodes and level mapping, i.e.,
    'level -> CG Nodes' used to quickly access all function nodes with the same level.

    :ivar project: name of the corresponding project
    :ivar cg: function name -> Node structure mapping
    :ivar max_level: depth of the call graph, i.e., maximum 'level' value across functions
    :ivar levels: level -> CG Nodes mapping
    """
    __slots__ = 'project', 'cg', 'max_level', 'levels'

    def __init__(self, project_name: str) -> None:
        """ Constructor

        :param project_name: name of the corresponding project
        """
        self.project: str = project_name
        # Construct the node mapping from a JSON file
        self.cg: NodeMap = self._parse_json_cg(
            load.load_and_deflate(f'{CG_PREFIX}_{project_name}', CG_DIR)
        )
        self.max_level: int = max(self.cg.values(), key=operator.attrgetter('level')).level
        self.levels: List[NodeMap] = self._build_levels()
    
    def __len__(self) -> int:
        """ Length protocol implementation.

        :return: number of levels
        """
        return len(self.levels)
    
    def __getitem__(self, level: int) -> NodeMap:
        """ Retrieves the CG Nodes associated with the given level.

        :return: Mapping of name -> node with the given level 
        """
        return self.levels[level]
    
    def __iter__(self) -> Iterator[NodeMap]:
        """ Iteration protocol implementation.

        :return: sequence of level 0, 1, 2, 3, ... NodeMaps
        """
        return iter(self.levels)

    @staticmethod
    def _parse_json_cg(json_cg: load.JsonType) -> NodeMap:
        """ Parse the given JSON formatted Call Graph into the Node mapping structure.

        :param json_cg: JSON formatted CG

        :return: function name -> CG Node mapping
        """
        # Access the actual CG Map
        json_cg = json_cg['perun_cg']['call_graph']['cg_map']
        # Build the basic 'function name: Node' map
        cg = {node['name']: CGNode(node) for node in json_cg.values()}
        # Store the Node references in callers and callees
        for node_name, node in json_cg.items():
            cg[node_name].callers = {caller: cg[caller] for caller in node['callers']}
            cg[node_name].callees = {callee: cg[callee] for callee in node['callees']}
        return cg

    def _build_levels(self) -> List[NodeMap]:
        """ Build the 'levels' mapping. Requires already constructed CG nodes map.

        :return: constructed levels mapping
        """
        levels: List[NodeMap] = [{} for _ in range(self.max_level + 1)]
        for node_name, node in self.cg.items():
            levels[node.level][node_name] = node
        return levels
