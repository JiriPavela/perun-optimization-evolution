from __future__ import annotations
import copy
from typing import List, Dict, Iterable, Sequence

import loader as load
import workload as wl
import call_graph as cg
from ordered_enum import OrderedEnum


# Mapping of optimization strength -> filtered functions
OptimizationMap = Dict[int, List[str]]


# Optimization strength max value (i.e., 0 - 100)
OPT_STR_MAX = 100


# TODO: doc
class ProjectOptimization:
    # Pre-compute CG levels?
    # Pre-compute Loopus
    # Pre-compute Diff Tracing
    def __init__(self, project_name: str, call_graph: cg.CallGraph) -> None:
        self.cg_projection: OptimizationMap = self._precompute_cgp(call_graph)
        self.static_baseline: OptimizationMap = self._precompute_sb(project_name, call_graph)
        self.diff_tracing: OptimizationMap = self._precompute_dt(project_name, call_graph)
    
    @staticmethod
    def _precompute_cgp(call_graph: cg.CallGraph) -> OptimizationMap:
        # Create new call graph levels hierarchy that contains function names directly
        levels = [list(level.keys()) for level in call_graph]
        return ProjectOptimization._build_incremental_mapping(levels, call_graph.max_level)
    
    @staticmethod
    def _precompute_sb(project_name: str, call_graph: cg.CallGraph) -> OptimizationMap:
        # Init the function -> complexity mapping
        func_bounds = {func: Complexity.GENERIC for func in call_graph.functions()}
        # Load the bounds profile and update the function -> complexity mapping
        bounds_profile = load.load_json(f'{project_name}_bounds.perf', wl.PROF_DIR)
        for res in bounds_profile['resource_type_map'].values():
            func_name = res['uid']['function']
            # Ignore local bounds records and functions that are never actually called
            # TODO: handle functions with same name from different modules (and build targets) 
            # that can also have different complexities - luckily, not the case with CCSDS 
            if res['type'] == 'total bound' and func_name in func_bounds:
                func_bounds[func_name] = Complexity.from_poly(res['class'])
        # Reverse the func_bounds mapping (i.e., to complexity -> functions) and convert it to list
        complexity_ordering = {c: order for order, c in enumerate(Complexity.opt_ordering())}
        bounds = [
            # The +1 for an additional empty list used to avoid IndexError
            [] for _ in range(len(complexity_ordering) + 1)
        ]
        for func_name, bound in func_bounds.items():
            bounds[complexity_ordering[bound]].append(func_name)
        # Create strength -> filtered functions mapping using the bounds_map
        return ProjectOptimization._build_incremental_mapping(bounds, len(complexity_ordering))

    @staticmethod
    def _precompute_dt(project_name: str, call_graph: cg.CallGraph) -> OptimizationMap:
        opt_mapping = {}
        for opt_strength in range(OPT_STR_MAX + 1):
            diff = call_graph.difference(cg.CallGraph(project_name, opt_strength)).diff_functions()
            # Memory optimization: when subsequent diffs are the same, store only the reference 
            # to the previously created list 
            prev_diff = opt_mapping.get(opt_strength - 1, [])
            if diff == prev_diff:
                diff = prev_diff
            opt_mapping[opt_strength] = diff
        return opt_mapping

    @staticmethod
    def _build_incremental_mapping(source: Sequence[List[str]], max_value: int) -> OptimizationMap:
        mapping: OptimizationMap = {}
        step = max_value / OPT_STR_MAX

        old_threshold = -1
        for opt_strength in range(OPT_STR_MAX + 1):
            threshold = max_value - round(step * opt_strength)
            # Reference the previously computed collection of functions to filter
            mapping[opt_strength] = mapping.get(opt_strength - 1, [])
            # If the threshold has changed, extend the collection with additional filtered functions
            if threshold != old_threshold:
                mapping[opt_strength] = copy.copy(mapping[opt_strength]) + source[threshold]
                old_threshold = threshold
        return mapping


class Complexity(OrderedEnum):
    """ Enumeration of the complexity degrees that we distinguish in the Bounds collector output.
    """
    # Polynomial-to-complexity map
    _ignore_ = ['map']

    # Complexities
    ZERO = 'zero'
    CONSTANT = 'constant'
    LINEAR = 'linear'
    QUADRATIC = 'quadratic'
    CUBIC = 'cubic'
    QUARTIC = 'quartic'
    GENERIC = 'generic'

    @staticmethod
    def supported() -> List[str]:
        """ List the currently supported Complexity degrees.

        :return: CLI names of the supported complexities
        """
        return [complexity.value for complexity in Complexity]

    # TODO: doc
    @staticmethod
    def opt_ordering() -> Iterable[Complexity]:
        # Necessary since the internal order parameter is for comparison of complexity
        # This order is reversed, indexed from 0 and contains the actual enum elements
        return reversed([complexity for complexity in Complexity])

    @staticmethod
    def max(values: Iterable[Complexity]) -> Complexity:
        """ Compare a collection of Complexity values and select the one with maximum degree.

        :param values: the set of Complexity values

        :return: the Complexity object with the highest degree of polynomial
        """
        return sorted(values, key=lambda complexity: complexity.order, reverse=True)[0]

    @classmethod
    def from_poly(cls, polynomial: str) -> Complexity:
        """ Create a Complexity object from string representing a polynomial.

        :param polynomial: a string representation of a supported polynomial

        :return: the corresponding Complexity object
        """
        return Complexity.map.get(polynomial, cls.GENERIC)


Complexity.map = {
    'O(1)': Complexity.CONSTANT,
    'O(n^1)': Complexity.LINEAR,
    'O(n^2)': Complexity.QUADRATIC,
    'O(n^3)': Complexity.CUBIC,
    'O(n^4)': Complexity.QUARTIC
}

