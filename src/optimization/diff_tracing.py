from __future__ import annotations
import re
import itertools
import operator
from typing import Dict, Generator, Iterable, List, Sequence, Set, Union, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from resources.call_graph import CallGraph, CGNode, BasicBlock, NodeMap, CfInstr


# Function rename map old_name -> new_name
RenameMap = Dict[str, str]
# Colors used for register coloring for specific instruction operands
RegisterColors = List[str]
# Colored instruction type, ("instruction", ["color1", "color2", ...])
ColoredInstr = Tuple[str, RegisterColors]


# The set of ASM JUMP instruction that are omitted during the operands check
JUMP_INSTRUCTIONS = {
    'call', 'jmp', 'je', 'jne', 'jz', 'jnz', 'jg', 'jge', 'jnle', 'jnl', 'jl', 'jle', 'jnge',
    'jng', 'ja', 'jae', 'jnbe', 'jnb', 'jb', 'jbe', 'jnae', 'jna', 'jxcz', 'jc', 'jnc', 'jo',
    'jno', 'jp', 'jpe', 'jnp', 'jpo', 'js', 'jns'
}


# Delimiters used in the disassembly operands specification
OPERANDS_DELIM = re.compile(r'([\s,:+*\-\[\]]+)')


def _build_registers_set() -> Set[str]:
    """ In order to correctly color and map registers, we need the set of all registers used
    in the assembly - currently, we restrict ourselves only to the x86-64 architecture.

    Since the set of registers is rather large, we construct the set (instead of simple
    enumeration) using some base names and prefixes/suffixes/counters.

    :return: the set of x86-64 registers
    """
    # The set of registers used in the x86-64 architecture
    registers = set()

    reg_classes: Dict[str, Sequence[str]] = {
        'full': ['ax', 'cx', 'dx', 'bx'],
        'partial': ['sp', 'bp', 'si', 'di'],
        'segment': ['ss', 'cs', 'ds', 'es', 'fs', 'gs'],
        'ip': ['ip'],
        '64b': ['r'],
        'sse': ['xmm'],
        'avx': ['ymm'],
        'prefix': ['r', 'e', ''],
        'postfix': ['l', 'h'],
        '64b-post': ['', 'd', 'w', 'b'],
    }
    reg_counters: Dict[str, Tuple[int, int]] = {
        '64b-cnt': (8, 15),
        'sse-cnt': (0, 7)
    }

    # Create Rrr, Err, rr, rL, rH variants
    for reg in reg_classes['full']:
        registers |= {'{}{}'.format(pre, reg) for pre in reg_classes['prefix']}
        registers |= {'{}{}'.format(reg[:1], post) for post in reg_classes['postfix']}

    # Create Rrr, Err, rr, rrL variants
    for reg in reg_classes['partial']:
        registers |= {'{}{}'.format(pre, reg) for pre in reg_classes['prefix']}
        registers.add('{}{}'.format(reg, reg_classes['postfix'][0]))

    # Add segment registers as-is
    registers |= set(reg_classes['segment'])
    # Create RIP, EIP, IP registers
    registers |= {'{}{}'.format(pre, reg_classes['ip'][0]) for pre in reg_classes['prefix']}
    # Create 64b register variants R8-R15
    start64, end64 = reg_counters['64b-cnt']
    for idx in range(start64, end64 + 1):
        registers |= {
            '{}{}{}'.format(reg_classes['64b'][0], str(idx), post) for post in reg_classes['64b-post']
        }
    # Create sse and avx register variants XMM0-XMM7 / YMM0 - YMM7
    start_sse, end_sse = reg_counters['sse-cnt']
    for idx in range(start_sse, end_sse + 1):
        registers |= {
            '{}{}'.format(reg, str(idx)) for reg in [reg_classes['sse'][0], reg_classes['avx'][0]]
        }
    return registers


class CallGraphDiff:
    """ Class representing the differences between two call graphs. Specifically, we compute
    collections of new, deleted, renamed and changed functions. This class is a reimplementation
    of the Diff Tracing method in Perun.

    :ivar new_funcs: functions from A and not B
    :ivar del_funcs: functions from B and not A
    :ivar renamed: mapping of function name changes (renames) from B to A
    :ivar changed: functions that have changed in A
    """
    __slots__ = 'new_funcs', 'del_funcs', 'renames', 'changed'

    # A collection of x86-64 registers
    x86_registers: Set[str] = _build_registers_set()

    def __init__(self, cg_A: CallGraph, cg_B: CallGraph) -> None:
        """ Constructor:

        :param cg_A: the first call graph
        :param cg_B: the second call graph
        """
        # Initially, new and del funcs also contain renames, the _filter_renames function removes
        # them after the renames are found
        self.new_funcs: NodeMap = {
            name: cg_A.funcs[name] for name in cg_A.funcs.keys() - cg_B.funcs.keys()
        }
        self.del_funcs: NodeMap = {
            name: cg_B.funcs[name] for name in cg_B.funcs.keys() - cg_A.funcs.keys()
        }
        # Name changes B -> A
        self.renames: RenameMap = self._filter_renames()
        self.changed: NodeMap = self._compare_cg(cg_A, cg_B)
    
    def diff_functions(self) -> List[str]:
        """ Provides a list of new and changed functions that should be thus profiled.

        :return: a collection of new + changed function names
        """
        # We need to rename the changed functions to obtain functions from the new call graph. 
        return list(
            self.new_funcs.keys() | {self.renames.get(func, func) for func in self.changed.keys()}
        )

    def _filter_renames(self) -> RenameMap:
        """ Find function renames in the new and del collections, create a rename mapping and
        remove the renamed functions from the new and del collections.

        :return: function old_name -> new_name mapping
        """
        renamed: RenameMap = {}
        # If new or deleted set is empty, no renaming could have taken place
        if not self.new_funcs or not self.del_funcs:
            return renamed
        
        # Sort the new and deleted sets by level
        new_srt, del_srt = self._sort_by_level(self.new_funcs), self._sort_by_level(self.del_funcs)
        # For each new function, try to find some deleted one that is "similar"
        for new_func in new_srt:
            match_idx = new_func.find_similar(del_srt, renamed)
            # Similar function in del was found
            if match_idx is not None:
                match = del_srt[match_idx]
                renamed[match.name] = new_func.name
                # Remove the renamed function from the new/deleted maps
                del self.new_funcs[new_func.name]
                del self.del_funcs[match.name]
                # Remove the matched deleted function from del_srt
                del del_srt[match_idx]
        return renamed

    def _compare_cg(self, cg_A: CallGraph, cg_B: CallGraph) -> NodeMap:
        """ Call Graph comparison algorithm for detecting changed functions. We perform the
        comparison on the assembly level to omit flagging source code changes that have no 
        impact on the resulting program, such as renaming variables, updating comments, etc.

        :param cg_A: the first call graph
        :param cg_B: the second call graph

        :return: collection of changed functions.
        """
        changed = {}
        # Inspect all functions that are present in both call graphs 
        # (although possibly renamed in the CG A)
        funcs_to_inspect = {
            name: cg_B.funcs[name] for name in cg_B.funcs.keys() - self.del_funcs.keys()
        }
        for func_B in funcs_to_inspect.values():
            # Obtain the function counterpart in A
            func_A = cg_A.funcs[self.renames.get(func_B.name, func_B.name)]
            # Obtain the set of B callees with renaming applied to allow for comparison
            callees_B = {self.renames.get(callee, callee) for callee in func_B.callees.keys()}
            # Check if the callees and control flow graphs match
            if func_A.callees.keys() != callees_B or not self._same_cfg(func_A.cf, func_B.cf):
                changed[func_B.name] = func_B
        return changed

    def _same_cfg(self, cf_A: Sequence[BasicBlock], cf_B: Sequence[BasicBlock]) -> bool:
        """ Compare the control flow graphs of two supposedly matching functions.

        :param cf_A: the first control flow graph
        :param cf_B: the second control flow graph

        :return: True if the CFGs do match, False otherwise
        """
        # Check if the number of basic blocks matches
        if len(cf_A) != len(cf_B):
            return False
        # Check the individual basic blocks
        for block_A, block_B in zip(cf_A, cf_B):
            # Check that the edges match exactly and that the number of instructions is the same
            if block_A.edges != block_B.edges or len(block_A.instr) != len(block_B.instr):
                return False
            # Compare the actual content of the instruction list
            if not self._same_instructions(block_A.instr, block_B.instr):
                return False
        return True
            
    def _same_instructions(self, instr_A: Iterable[CfInstr], instr_B: Iterable[CfInstr]) -> bool:
        """ The Coloring mode performs a register coloring and subsequently compares the 
        instructions by searching for possible bijection. This ensures that simple reordering 
        of instructions or change of used registers is not regarded as a semantic change.

        :param instr_A: list of (instruction, operands) tuples representing the basic block in A
        :param instr_B: list of (instruction, operands) tuples representing the basic block in B

        :return bool: True if the blocks match, False otherwise
        """
        def _color_registers(operand_parts: Iterable[str]) -> Generator[str, None, None]:
            """ Identify registers within the parsed operand and color them.
            Colored registers are represented simply by the '<r>' expression.

            :param operand_parts: collection of tokens from the parsed operand

            :return: updated operand tokens where colored registers are substituted
            """
            for expr in operand_parts:
                # The expression is an register
                if expr in self.x86_registers:
                    # Fetch the register's color, or assign it a new one
                    instr_colors.append(color_map.setdefault(expr, str(next(color_counter))))
                    yield '<r>'
                # Not a register, simply return the token
                else:
                    yield expr

        # Perform the coloring on both the A and B cfg blocks
        instr_stack_A: List[ColoredInstr] = []
        instr_stack_B: List[ColoredInstr] = []
        for instr_set, stack in [(instr_A, instr_stack_A), (instr_B, instr_stack_B)]:
            # We represent different colors by a simple integer counter values casted to strings
            color_counter = itertools.count()
            color_map: Dict[str, str] = {}
            # Parse the instructions and operands, substitute and color registers
            for inst_A, oper in [inst for inst in instr_set if inst[0] not in JUMP_INSTRUCTIONS]:
                instr_colors: RegisterColors = []
                op_parts = re.split(OPERANDS_DELIM, oper)
                # Construct instruction string fit for comparison
                instr_full = '{} '.format(inst_A) + ''.join(_color_registers(op_parts))
                stack.append((instr_full, instr_colors))
            # Sort the instruction stack to invalidate instruction reordering
            stack.sort(key=lambda inst: inst[0])

        color_map = {}
        # Traverse the instructions stacks and compare the elements
        for (inst_A, colors_A), (inst_B, colors_B) in zip(instr_stack_A, instr_stack_B):
            # Non-matching instructions means a change is present
            if inst_A != inst_B:
                return False
            # Try to map the colors:
            # - when the color of A and B register is different, map it: "A_clr" -> "B_clr"
            # - all subsequent instances of the same register have to match the mapped color
            for col_A, col_B in zip(colors_A, colors_B):
                if color_map.setdefault(col_A, col_B) != color_map[col_A]:
                    return False
        return True

    @staticmethod
    def _sort_by_level(nodes: NodeMap, reverse: bool=True) -> List[CGNode]:
        """ Sort the nodes in a map according to their level. 
        By default, the sorting is in descending order, i.e., highest_level -> ... -> lowest_level

        :param nodes: the collection of nodes to sort
        :param reverse: sets the descending (True) or ascending (False) sorting order

        :return: sorted list of nodes
        """
        return sorted(nodes.values(), key=operator.attrgetter('level'), reverse=reverse)
