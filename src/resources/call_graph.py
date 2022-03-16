""" Module implementing Call Graph representation of individual projects (programs).
Notably, the CG representation contains the so-called 'levels', where each function contained
in the CG is assigned a level value that represents an estimation of the longest acyclic path
from root (presumably main) to the function itself.
"""
from __future__ import annotations
import operator
from typing import Dict, List, Set, Tuple, Iterable, Iterator, MutableMapping, Optional, \
                   Sequence, Union

import utils.values as values
import resources.loader as load
import optimization.diff_tracing as diff


# Mapping CG node (function name) -> function details
NodeMap = Dict[str, 'CGNode']
# Control flow's Basic Block (BB) as stored in a JSON, i.e., sequence of either 
# ["instruction", "operands"] or "function name" for special blocks that refer to a function call
JsonBB = Iterable[Union[Sequence[str], str]]
# Internal representation of an ASM instruction as ("instruction", "operands") or
# (CALL_INSTR_STUB, "function name")
CfInstr = Tuple[str, str]
# Control flow graph stored as a sequence of sorted basic blocks
ControlFlow = List['BasicBlock']
# Function rename mapping: old_name -> new_name
RenameMapping = MutableMapping[str, str]


# Some basic blocks are effectively stubs that contain only name of the called function
# Create "fake" instruction for such blocks and store the function name as the operand
_CALL_INSTR_STUB = '!call!'


class BasicBlock:
    """ Structure of the control flow's basic block. Each basic block contains a collection of
    ASM instructions and their operands as well as references to the "connected" blocks. 
    
    :ivar instr: a sorted collection of ASM instructions found in the basic block
    :ivar edges: a sorted collection of outgoing block edges
    """
    __slots__ = 'instr', 'edges'

    def __init__(self, instructions: JsonBB) -> None:
        """ Constructor

        :param instructions: a collection of instructions ["instr", "op"] or "function name"
        """
        self.instr: List[CfInstr] = []
        self.edges: List[int] = []  # We don't need "proper" reference, the Block index suffices
        for inst in instructions:
            # Differentiate between the actual instructions and function call stubs
            if isinstance(inst, str):
                self.instr.append((_CALL_INSTR_STUB, inst))
            else:
                self.instr.append((inst[0], inst[1]))

    def add_edge(self, to_block: int) -> None:
        """ Add outgoing edge reference to the block.

        :param to_block: index (to the BB list) of the target block
        """
        self.edges.append(to_block)


class CGNode:
    """ Call Graph Node structure. Each node represents a function of the program and contains
    the neccessary details about the function. Moreover, each node contains full control flow graph
    of the corresponding function (in ASM).

    :ivar name: the function name
    :ivar level: estimation of the maximum possible call stack depth of the function
    :ivar cf: the control flow graph structure consisting of sorted basic blocks
    :ivar callers: a collection of the caller functions, i.e., functions that can call this node
    :ivar callees: a collection of the callee functions, 
                   i.e., functions that can be called from this node
    """
    __slots__ = 'name', 'level', 'cf', 'callers', 'callees'
    
    def __init__(self, cg_node: load.JsonType, control_flow: load.JsonType) -> None:
        """ Constructor

        :param json_node: Node information in the JSON format
        :param control_flow: Control flow representation in the JSON format
        """
        self.name: str = cg_node['name']
        self.level: int = int(cg_node['level'])
        # We need an efficient and compact storage to save space when CGs are large
        self.cf: ControlFlow = self._build_control_flow(control_flow)
        # Callers and Callees are initialized later
        self.callers: NodeMap = {}
        self.callees: NodeMap = {}

    def find_similar(self, nodes: Iterable[CGNode], renames: RenameMapping) -> Optional[int]:
        """ Search a collection of nodes for a "similar" node. We define the similarity between
        two nodes A, B as follows:

          A.nb_callees == B.nb_r_callees AND |A.callers| == |B.callers|
        
        where:
         - "nb_callees" refer to callees that are not reached through graph backedge (non-backedge)
         - "nb_r_callees" are nb_callees with applied renaming 

        We leverage the similarity when pairing nodes from two different Call Graph versions
        that might have been renamed.

        :param nodes: a collection of nodes where to search for similarity
        :param renames: currently known function rename mapping

        :return index of the similar node in the node collection if found 
        """
        callees = {callee.name for callee in self.callees.values() if callee.level > self.level}
        # Search the nodes until a first match is found
        for idx, other in enumerate(nodes):
            if callees == other._callees(renames) and len(self.callers) == len(other.callers):
                return idx
        return None

    def _callees(self, renames: Optional[RenameMapping]) -> Set[str]:
        """ Obtain a set of non-backedge callees with applied renaming. If no renaming map 
        is known, simply return a set of non-backedge callees.

        :param renames: currently known function rename mapping

        :return: non-backedge set of function with or without applied renaming
        """
        # No renaming map is supplied, simply return the callees
        if renames is None:
            return set(func.name for func in self.callees.values() if func.level > self.level)
        # Apply rename
        return {
            renames.get(func.name, func.name)
            for func in self.callees.values() if func.level > self.level
        }
    
    @staticmethod
    def _build_control_flow(cf: load.JsonType) -> ControlFlow:
        """ Construct the control flow representation. The control flow is stored as a sorted 
        list of basic blocks, where each basic block constitutes of ASM instructions + operands
        and outgoing block edges.

        :param cf: the JSON-formatted control flow

        :return: constructed control flow representation
        """
        # Some functions might miss the control flow data
        if not cf:
            return []
        # Create the block objects
        cf_blocks = [BasicBlock(block) for block in cf['blocks']]
        # Set the outgoing edges for each block
        for edge_from, edge_to in cf['edges']:
            cf_blocks[edge_from].add_edge(edge_to)
        return cf_blocks


class CallGraph:
    """ Call Graph structure containing function nodes and level mapping, i.e.,
    'level -> CG Nodes' used to quickly access all function nodes with the same level.

    :ivar project: name of the corresponding project
    :ivar version: identification of the project version as SHA of git commit
    :ivar funcs: function name -> Node structure mapping
    :ivar max_level: depth of the call graph, i.e., maximum 'level' value across functions
    :ivar levels: level -> CG Nodes mapping
    """
    __slots__ = 'project', 'version', 'funcs', 'max_level', 'levels'

    def __init__(self, project_name: str, version_index: Optional[int] = None) -> None:
        """ Constructor

        :param project_name: name of the corresponding project
        :param version_index: index of an old cg to load
        """
        self.project: str = project_name
        # Construct the call graph file name
        filename = f'{values.CG_PREFIX}_{project_name}'
        if version_index is not None:
            # We are loading a CG from some previous program version
            filename = f'{values.OLD_CG_PREFIX}_{project_name}_{version_index}'
        # Load the CG as JSON-formatted text
        cg_json = load.load_and_deflate(filename, values.CG_DIR)
        # Extract project version
        self.version: str = cg_json['perun_cg']['minor_version']
        # Construct the node mapping from a JSON file
        self.funcs: NodeMap = self._parse_json_cg(cg_json)
        self.max_level: int = max(self.funcs.values(), key=operator.attrgetter('level')).level
        self.levels: List[NodeMap] = self._build_levels()

    def difference(self, other: CallGraph) -> diff.CallGraphDiff:
        """ Compare two call graphs and provide collections of new, deleted, renamed and changed
        functions wrt to the other call graph.

        :param other: the other call graph to compute the difference from

        :return: an object representing the differences of the two call graphs
        """
        return diff.CallGraphDiff(self, other)

    def functions(self) -> Set[str]:
        """ Retrieve the names of all functions in the call graph.

        :return: set of function names
        """
        return set(self.funcs.keys())

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
    def _parse_json_cg(json_graph: load.JsonType) -> NodeMap:
        """ Parse the given JSON formatted Call Graph into the Node mapping structure.

        :param json_graph: JSON formatted CG

        :return: function name -> CG Node mapping
        """
        # Access the actual CG Map
        json_cg = json_graph['perun_cg']['call_graph']['cg_map']
        json_cfg = json_graph['perun_cg']['control_flow']
        # Build the basic 'function name: Node' map
        cg = {node['name']: CGNode(node, json_cfg.get(node['name'], {})) for node in json_cg.values()}
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
        for node_name, node in self.funcs.items():
            levels[node.level][node_name] = node
        return levels


def list_old_call_graphs(project_name: str) -> List[str]:
    """ List old call graph files.

    :param project_name: name of the corresponding project ('*' for all)

    :return: list of the old call graph file names 
    """
    return load.list_files(values.CG_DIR, f'{values.OLD_CG_PREFIX}_{project_name}_*')