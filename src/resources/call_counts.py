from __future__ import annotations
from typing import Dict, List, Iterable, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    import resources.loader as load


# Function name -> [call counts per thread]
PerThreadCalls = Dict[str, List[int]]
# Function call count as saved in global calls / total calls
FuncCalls = Dict[str, int]
# Generic call count type
# Suitable for generic functions that operate on per-thread / global / total mappings
CountT = TypeVar('CountT', int, List[int])


class CallCounts:
    __slots__ = 'per_thread', 'global_only', 'total', 'max_global', 'total_calls'

    def __init__(self, stats: load.JsonType) -> None:
        self.per_thread: PerThreadCalls = self._compute_per_thread(stats)
        self.global_only = self._compute_global(stats)
        self.total = {name: sum(calls) for name, calls in self.per_thread.items()}
        self.max_global = max(self.global_only.values())
        self.total_calls = sum(self.total.values())

    def sub_global(self, func_subset: Iterable[str]) -> FuncCalls:
        return self._subset(self.global_only, func_subset)

    @staticmethod
    def _subset(call_map: Dict[str, CountT], func_subset: Iterable[str]) -> Dict[str, CountT]:
        """ Retrieve a subset of the call_map mapping.

        :param func_subset: collection of function names to retrieve the call counts for

        :return: a subset of function name -> call count mapping
        """
        # No subset specified, return the whole call_map
        if func_subset is None:
            return call_map
        # Create the desired subset of the call count map
        return {func: call_map[func] for func in func_subset if func in call_map}

    @staticmethod
    def _compute_per_thread(stats: load.JsonType) -> PerThreadCalls:
        thread_calls: PerThreadCalls = {}
        tids = sorted(stats['dynamic-stats']['per_thread'].keys())
        for tid in tids:
            for func, func_stats in stats['dynamic-stats']['per_thread'][tid].items():
                if func_stats['sample'] != 1:
                    raise RuntimeError(
                        f"Dynamic stats contain sampled data ({func}: {func_stats['sample']})!"
                    )
                thread_calls.setdefault(func, []).append(func_stats['sampled_count'])
        return thread_calls

    @staticmethod
    def _compute_global(stats: load.JsonType) -> FuncCalls:
        global_calls: FuncCalls = {}
        for func, func_stats in stats['dynamic-stats']['global_stats'].items():
            global_calls[func] = func_stats['sampled_count']
        return global_calls
