from __future__ import annotations
import copy
import math
from typing import Generator, Any, Iterator, List, Dict, Iterable, Set, Tuple, TYPE_CHECKING

import utils.values as values
import resources.loader as load
import resources.call_graph as cg
import resources.call_counts as cc
from utils.ordered_enum import OrderedEnum


if TYPE_CHECKING:
    from project.workload import Workload


# Mapping of optimization strength -> filtered functions
# OptimizationMap = Dict[int, List[str]]
OptimizationMap = Dict[int, "OptNode"]
# Mapping of optimization strength -> function sampling
SamplingMap = Dict[int, "SampledFuncs"]
# Function name -> [call counts per thread]
ThreadFuncCalls = Dict[str, List[int]]
# Function call count as saved in global stats
GlobalFuncCalls = Dict[str, int]
# Mapping of function -> (sample count, ignored count) for a specific threshold value
ThresholdSampling = Dict[str, Tuple[int, int]]
# Function name -> (remaining calls, ignored calls)
OptimizedFuncs = Dict[str, Tuple[int, int]]
OptData = Tuple["OptNode", "OptNode", "OptNode", "OptNode", "SampledFuncs"]
StrInterval = Tuple[int, int]



# Optimization strength interval (i.e., 0 - 100)
OPT_STR_MAX = 100
OPT_STR_MIN = 0
# Dynamic baseline constants
_CONSTANT_MEDIAN_RATIO = 0.05
_MEDIAN_RESOLUTION = 10
_CONSTANT_THRESHOLD = 100
# Dynamic sampling constant
_THRESHOLD_EPS_RATIO = 0.1

# TODO: main is never removed, update the optimization maps


def _gen_thresholds(
    interval_start: int, interval_end: int, max_thresholds: int = OPT_STR_MAX + 1
) -> Generator[int, None, None]:
    """ Generate maximum of 'max_thresholds' uniformly distributed threshold values from the 
    given interval [start, end]. If the size of the interval (|interval size|) is smaller than 
    the 'max_thresholds', generate only |interval size| threshold values.

    :param interval_start: start of the interval
    :param interval_end: end of the interval
    :param max_thresholds: maximum number of thresholds to generate

    :return: generator that provides threshold values
    """
    interval_size = abs(interval_end - interval_start) + 1  # +1 to account for the interval edges
    # Compute the increment / decrement for each threshold
    step = 1 if interval_size <= max_thresholds else (interval_size - 1) / (max_thresholds - 1)
    step = -step if interval_end - interval_start < 1 else step
    # Number of thresholds to generate
    no_thresholds = min(interval_size, max_thresholds)
    for threshold in range(no_thresholds):
        yield interval_start + round(step * threshold)


class OptNode:
    __slots__ = 'parameter_value', 'funcs'

    def __init__(self, parameter: Any, funcs: List[str]) -> None:
        self.parameter_value: Any = parameter
        self.funcs: List[str] = funcs



class SampledFuncs:
    """ Representation of the sampling effect on functions based on the threshold.

    :ivar threshold: the sampling threshold value
    :ivar sampled: pre-computed results of sampling applied to functions
    """
    __slots__ = 'threshold', 'sampled'
    def __init__(self, call_counts: cc.CallCounts, threshold: int) -> None:
        """ Constructor

        :param g_calls: global function call counts
        :param t_calls: per-thread function call counts
        :param threshold: the sampling threshold value 
        """
        self.threshold: int = threshold
        self.sampled: ThresholdSampling = self._compute_sampled(call_counts)
    
    def get_totals(self) -> Tuple[int, int]:
        return sum(s[0] for s in self.sampled.values()), sum(s[1] for s in self.sampled.values())

    def __getitem__(self, key: str) -> Tuple[int, int]:
        return self.sampled[key]

    def _compute_sampled(self, call_counts: cc.CallCounts) -> ThresholdSampling:
        """ Pre-compute the sampling effect.

        :param g_calls: global function call counts
        :param t_calls: per-thread function call counts

        :return: sampling result for all functions in global stats
        """
        # If the threshold is 0 then no function is being profiled, let alone sampled
        if self.threshold == 0:
            return {
                func: (0, sum(calls)) for func, calls in call_counts.per_thread.items()
            }
        # Otherwise, compute the expected number of samples and ignored calls
        sampled: ThresholdSampling = {}
        threshold_low, threshold_high = self._get_threshold_interval()
        # The computation is based on the global call stats
        for func, func_calls in call_counts.global_only.items():
            new_sampling = 1
            # Update the sampling value if the number of samples changes noticeably
            if func_calls < threshold_low or func_calls > threshold_high:
                new_sampling = math.floor(new_sampling / (self.threshold / func_calls))
                # Normalize the sampling value
                new_sampling = 1 if new_sampling < 1 else new_sampling
            
            # Update the sampled dictionary with (sampled calls, ignored calls)
            sampled_calls = sum(
                1 + ((calls - 1) // new_sampling) for calls in call_counts.per_thread[func]
            )
            sampled[func] = (
                sampled_calls,
                sum(call_counts.per_thread[func]) - sampled_calls
            )
        return sampled

    def _get_threshold_interval(self) -> Tuple[float, float]:
        threshold_eps = self.threshold * _THRESHOLD_EPS_RATIO
        return (self.threshold - threshold_eps, self.threshold + threshold_eps)


class WorkloadOpt:
    # Pre-compute Dynamic baseline
    # Pre-compute Sampling
    def __init__(self, stats: load.JsonType, call_counts: cc.CallCounts) -> None:
        # TODO: call_counts only temporary
        self.call_counts: cc.CallCounts = call_counts
        self.dynamic_baseline: OptimizationMap = self._precompute_db(stats, call_counts)
        # TODO: handle case where samplings across different workloads have different maximum str!
        self.sampling: SamplingMap = self._precompute_sampling(call_counts)
    
    def print_steps(self) -> None:
        print('DB:')
        for step, node in self.dynamic_baseline.items():
            print(f' {step} ({node.parameter_value}); funcs: {len(node.funcs)}')
        print('DS:')
        for step, node in self.sampling.items():
            print(f' {step} ({node.threshold}); (samples, ignored): {node.get_totals()}')

    def apply_strength(self, db_str: int, ds_str: int, **_: Any) -> Tuple[OptNode, SampledFuncs]:
        return self.dynamic_baseline[db_str], self.sampling[ds_str]

    def strength_intervals(self, constr: StrConstraints) -> None:
        constr['db_str'] = (OPT_STR_MIN, max(self.dynamic_baseline.keys()))
        constr['ds_str'] = (OPT_STR_MIN, max(self.sampling.keys()))

    @staticmethod
    def _precompute_db(stats: load.JsonType, call_counts: cc.CallCounts) -> OptimizationMap:
        constant_funcs = [
            name for name, func_stats in stats['dynamic-stats']['global_stats'].items() 
            if WorkloadOpt._is_constant(func_stats)
        ]
        # Only three thresholds here
        return {
            0: OptNode((call_counts.max_global + 1, call_counts.max_global + 1), []),
            1: OptNode((_CONSTANT_THRESHOLD, call_counts.max_global + 1), constant_funcs),
            2: OptNode((0, 0), list(stats['dynamic-stats']['global_stats'].keys()))
        }
    
    @staticmethod
    def _precompute_sampling(call_counts: cc.CallCounts) -> SamplingMap:
        # Initialize no-opt strength in the map and simultaneously find the maximum number of calls
        # across all threads / processes and functions.
        # Sampling strength -> Sampling thresholds:
        #    0:  ceil(Max_calls / 2) + 1
        #  ...:  ...  
        #  100:  0
        return {
            opt_str: SampledFuncs(call_counts, threshold) 
            for opt_str, threshold in enumerate(_gen_thresholds(
                math.ceil(call_counts.max_global / 2) + 1, 0, 500)
            )
        }

    @staticmethod
    def _is_constant(func_stats: load.JsonType) -> bool:
        return (func_stats['IQR'] < func_stats['median'] * _CONSTANT_MEDIAN_RATIO or
                func_stats['median'] < _MEDIAN_RESOLUTION)


# TODO: doc
class ProjectOpt:
    # Pre-compute CG levels?
    # Pre-compute Loopus
    # Pre-compute Diff Tracing
    def __init__(self, project_name: str, call_graph: cg.CallGraph) -> None:
        self.cg_projection: OptimizationMap = self._precompute_cgp(call_graph)
        self.static_baseline: OptimizationMap = self._precompute_sb(project_name, call_graph)
        self.diff_tracing: OptimizationMap = self._precompute_dt(project_name, call_graph)
    
    def print_steps(self) -> None:
        print('CGP:')
        for step, node in self.cg_projection.items():
            print(f' {step} ({node.parameter_value}); funcs: {len(node.funcs)}')
        print('SB:')
        for step, node in self.static_baseline.items():
            print(f' {step} ({node.parameter_value}); funcs: {len(node.funcs)}')
        print('DT')
        for step, node in self.diff_tracing.items():
            print(f' {step} ({node.parameter_value}); funcs: {len(node.funcs)}')
    
    def apply_strength(
        self, cgp_str: int, sb_str: int, dt_str: int, **_: Any
    ) -> Tuple[OptNode, OptNode, OptNode]:
        return (
            self.cg_projection[cgp_str], self.static_baseline[sb_str], self.diff_tracing[dt_str]
        )
    
    def strength_intervals(self, constr: StrConstraints) -> None:
        constr['cgp_str'] = (OPT_STR_MIN, max(self.cg_projection.keys()))
        constr['sb_str'] = (OPT_STR_MIN, max(self.static_baseline.keys()))
        constr['dt_str'] = (OPT_STR_MIN, max(self.diff_tracing.keys()))

    @staticmethod
    def _precompute_cgp(call_graph: cg.CallGraph) -> OptimizationMap:
        # [(0, [main]), (1, [f, g, h]), ..., (max. [])]
        levels: List[Tuple[int, List[str]]] = []
        # Build the level 
        for level, level_funcs in enumerate(call_graph):
            levels.append((level, list(level_funcs.keys())))
        # Trailing empty list added for 'no optimization' strength
        levels.append((len(call_graph), []))
        # Reverse the list since CGP remove functions from the bottom of the CG
        levels.reverse()
        return ProjectOpt._build_incremental_mapping(levels)

    @staticmethod
    def _precompute_sb(project_name: str, call_graph: cg.CallGraph) -> OptimizationMap:
        # Init the function -> complexity mapping
        func_bounds = {func: Complexity.GENERIC for func in call_graph.functions()}
        # Load the bounds profile and update the function -> complexity mapping
        bounds_profile = load.load_json(f'{project_name}_bounds.perf', values.PROF_DIR)
        for res in bounds_profile['resource_type_map'].values():
            func_name = res['uid']['function']
            # Ignore local bounds records and functions that are never actually called
            # TODO: handle functions with same name from different modules (and build targets) 
            # that can also have different complexities - luckily, not the case with CCSDS 
            if res['type'] == 'total bound' and func_name in func_bounds:
                func_bounds[func_name] = Complexity.from_poly(res['class'])
        # Reverse the func_bounds mapping (i.e., to complexity -> functions) and convert it to list
        complexity_ordering = {c: order for order, c in enumerate(Complexity.opt_ordering())}
        # [(complexity1, [func_list]), (complexity2, [func_list]), ...]
        bounds: List[Tuple[str, List[str]]] = [
            (c.value, []) for c in complexity_ordering.keys()
        ]
        for func_name, bound in func_bounds.items():
            bounds[complexity_ordering[bound]][1].append(func_name)
        # Create strength -> filtered functions mapping using the bounds_map
        return ProjectOpt._build_incremental_mapping(bounds)

    @staticmethod
    def _precompute_dt(project_name: str, call_graph: cg.CallGraph) -> OptimizationMap:
        opt_mapping: OptimizationMap = {}
        for opt_strength, _ in enumerate(cg.list_old_call_graphs(project_name)):
            try:
                old_cg = cg.CallGraph(project_name, opt_strength)
                diff = call_graph.difference(old_cg).diff_functions()
                old_cg_version = old_cg.version
            except FileNotFoundError:
                diff = list(call_graph.funcs.keys())
                old_cg_version = '-'
            try:
                prev_diff = opt_mapping[opt_strength - 1].funcs
            except KeyError:
                prev_diff = []
            # Memory optimization: when subsequent diffs are the same, store only the reference 
            # to the previously created list 
            if diff == prev_diff:
                diff = prev_diff
            opt_mapping[opt_strength] = OptNode(old_cg_version, diff)
        return opt_mapping
    
    @staticmethod
    def _build_incremental_mapping(source: List[Tuple[Any, List[str]]]) -> OptimizationMap:
        mapping: OptimizationMap = {}
        for opt_strength, threshold in enumerate(_gen_thresholds(0, len(source) - 1)):
            try:
                previous_funcs = copy.copy(mapping[opt_strength - 1].funcs)
            except KeyError:
                previous_funcs = []
            mapping[opt_strength] = OptNode(
                source[threshold][0], previous_funcs + source[threshold][1]
                )
        return mapping


class OptSteps:
    def __init__(self, proj_opt: ProjectOpt, wl_opt: WorkloadOpt) -> None:
        self.project_opt: ProjectOpt = proj_opt
        self.workload_opt: WorkloadOpt = wl_opt

    def apply_strength(self, **str_params) -> OptData:
        return self.project_opt.apply_strength(**str_params) + self.workload_opt.apply_strength(**str_params)

    def get_strength_constraints(self) -> StrConstraints:
        constr = StrConstraints()
        self.project_opt.strength_intervals(constr)
        self.workload_opt.strength_intervals(constr)
        # constr['fitness'] = (20, 80)
        return constr


class StrConstraints:
    def __init__(self) -> None:
        self.constr_map: Dict[str, StrInterval] = {}
    
    def __iter__(self) -> Iterator[Tuple[str, StrInterval]]:
        return iter(self.constr_map.items())

    def __getitem__(self, key: str) -> StrInterval:
        return self.constr_map[key]

    def __setitem__(self, key: str, value: StrInterval) -> None:
        self.constr_map[key] = value


class OptimizedRun:
    __slots__ = (
        'func_map', 'filtered', 'sampled', 'ignored_cnt', 'saved_cnt', 'sample_hits', 'sample_miss', 
        'cgp_param', 'sb_param', 'dt_param', 'db_param', 'ds_param'
    )

    def __init__(self) -> None:
        self.func_map: OptimizedFuncs = {}
        self.filtered: int = 0
        self.sampled: int = 0
        self.saved_cnt: int = 0
        self.ignored_cnt: int = 0
        self.sample_hits: int = 0
        self.sample_miss: int = 0
        # TODO: refactor, maybe combine with the strength in a class?
        self.cgp_param: int = 0
        self.sb_param: str = ''
        self.dt_param: str = ''
        self.db_param: Tuple[int, int] = (0, 0)
        self.ds_param: int = 0

    def apply(self, wl: Workload, **strength: int) -> OptimizedRun:
        cgp_opt, sb_opt, dt_opt, db_opt, ds_opt = wl.opt_data.apply_strength(**strength)
        # Update the actual profiling parameters
        self.saved_cnt, self.ignored_cnt, self.sample_hits, self.sample_miss = 0, 0, 0, 0
        self.func_map = {}
        self.cgp_param = cgp_opt.parameter_value
        self.sb_param  = sb_opt.parameter_value
        self.dt_param  = dt_opt.parameter_value
        self.db_param  = db_opt.parameter_value
        self.ds_param  = ds_opt.threshold
        # Update the function map to correspond with the 
        filtered = set(cgp_opt.funcs + sb_opt.funcs + db_opt.funcs) - set(dt_opt.funcs) - {'main'}
        sampled = ds_opt.sampled.keys() - filtered
        self.filtered = len(filtered)
        self.sampled = len(sampled)
        
        for func, calls in wl.call_counts.total.items():
            if func in filtered:
                func_calls = (0, calls)
                self.ignored_cnt += calls
            elif func in sampled:
                func_calls = ds_opt[func]
                self.sample_hits += func_calls[0]
                self.sample_miss += func_calls[1]
            else:
                func_calls = (calls, 0)
                self.saved_cnt += func_calls[0]
            self.func_map[func] = func_calls
        return self


class OptEffect:
    def __init__(self, workload: Workload, opt_goal: int, **_) -> None:
        self.wl: Workload = workload
        self.opt_goal: float = float(opt_goal)
        self.optimized_run: OptimizedRun = OptimizedRun()
        # Can be extended, e.g., with error metrics
        self.time_ratio: float = 0.0
        self.data_ratio: float = 0.0
    
    def apply(self, **strength: int) -> OptEffect:
        self.optimized_run.apply(self.wl, **strength)
        self._compute_reductions()
        return self
    
    def print_effect(self) -> None:
        print(f'Individual')
        print(f'  |filtered|={self.optimized_run.filtered}')
        print(f'  |sampled|={self.optimized_run.sampled}')
        print(f'  |s,i,h,m|={self.optimized_run.saved_cnt, self.optimized_run.ignored_cnt, self.optimized_run.sample_hits, self.optimized_run.sample_miss}')

    def _compute_reductions(self) -> None:
        overheads = self.wl.metrics.overhead
        new_time_overhead = (
            self.optimized_run.saved_cnt * overheads.instrumented_hit
            + self.optimized_run.sample_hits * overheads.sample_hit
            + self.optimized_run.sample_miss * overheads.sample_miss
        )
        new_data_overhead = (
            ((self.optimized_run.saved_cnt + self.optimized_run.sample_hits)
            * overheads.data_per_record) + overheads.data_offset
        )
        self.time_ratio = new_time_overhead / overheads.tot_time_overhead
        self.data_ratio = new_data_overhead / overheads.tot_data_overhead


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
    def opt_ordering() -> List[Complexity]:
        # Necessary since the internal order parameter is for comparison of complexity
        # This order is indexed from 0 and contains the actual enum elements
        return [complexity for complexity in Complexity]

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

