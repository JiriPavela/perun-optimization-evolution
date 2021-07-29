""" Module containing the implementation of Solutions and their collections. Solution represents
an actual result of an evolutionary technique applied to a certain (project, workload) combination.
Generally, it is safe to assume that a Solution object is a fenotype with some additional attributes
such as fitness value, parameters used in genotype -> fenotype conversion etc.

Due to a significant conceptual difference between fenotypes of Problem A and B (see 'cases'), the
Solution class is in fact a Protocol (i.e., implicit interface) and each Problem should have its
own solution class that complies with the Solution protocol.

Note that due to the multiprocessing support, the results must be somehow transferable among
different processes. Since the Solution objects contain references to the associated projects and 
workloads (and the fact that subsequent pickle and unpickle operations do not preserve the 
original references), the final Solution objects updated by worker processes are transfered back to
the main controller process as dictionaries containing only the updated values. 
"""
from __future__ import annotations
import math
import copy
from typing import List, Dict, Sequence, Set, Any, Mapping, Iterable, Tuple, Callable, TypeVar, \
    Union, Generator, Optional, Protocol, Type

import pandas as pd

import loader
import project as proj
from workload import Workload


# Signature of an apply function
ApplySignature = Callable[..., int]
# Signature of a fitness function
FitnessSignature = Callable[..., float]
# Representation of an int interval, see the fitness function below
CallsInterval = Tuple[int, int, int, int]
# Function result prepared for DataFrame conversion 
FuncResultRow = List[Any]
# Parameter type supplied to the apply function
ApplyParam = Union[int, float]
# Solution class type variable
# https://mypy.readthedocs.io/en/stable/kinds_of_types.html#the-type-of-class-objects
S = TypeVar('S', bound='Solution')


# Minimum and maximum accptable function calls
MIN_CALLS = 1
MAX_CALLS = 1000000
# Acceptable deviation from the optimal call count
ACCEPTABLE_DEVIATION = 0.05


def fitness_(func: FuncResult) -> float:
    """ Computes the fitness value of the new sampled call count based on a function:
    
     f |
     i |            _____
     t |           /. . .\\
     n |          / . . . \\
     e |         /  . . .  \\
     s |  ______/   . . .   \\______
     s +--======+---+-+-+---+======-------> 
                A   B C D   E           calls
    
    Where
     A) The minimum number of calls
     B) The start of the "acceptable deviation interval"
     C) The ideal ("expected") number of sampled calls
     D) The end of the "acceptable deviation interval"
     E) The number of unsampled function calls or 
        the maximum function calls that is acceptable after sampling (MAX CALLS constant)
    
    The "acceptable deviation interval is +-5% of the "expected" number of calls"
    
    The function is generally not continuous, e.g., in cases when the "expected" call count (C)
    is close to the maximum number of acceptable calls (E). The end of the acceptable deviation 
    interval (D) overlaps the E, thus no slope can be computed.  
     
     f |             _____
     i |            /. . .
     t |           / . . .
     n |          /  . . .
     e |         /   . . .
     s |  ______/    . . .______
     s +--======+----+-+-+======------->
                A    B C D&E         calls
    
    Mathematical specification of the function:
    
              { 0.0                     iff  calls < A
              { (1 / (B - A)) * X       iff  A <= calls /\\ calls <  B
    fitness = { 1.0                     iff  B <= calls /\\ calls <= D
              { 1 + (1 / (D - E)) * X   iff  D <  calls /\\ calls <= E
              { 0.0                     iff  calls > E

    The interval is precomputed in the FuncResult class

    :param func: an object representing project function to compute the fitness for

    :return: the computed fitness value
    """
    # Compute the resulting fitness
    # [A, B) interval
    if func.interval[0] <= func.sampled < func.interval[1]:
        slope = 1 / (func.interval[1] - func.interval[0])
        return slope * (func.sampled - func.interval[0])
    # [B, D] interval
    elif func.interval[1] <= func.sampled <= func.interval[2]:
        return 1.0
    # (D, E] interval
    elif func.interval[2] < func.sampled <= func.interval[3]:
        slope = 1 / (func.interval[2] - func.interval[3])
        return 1 + (slope * (func.sampled - func.interval[2]))
    # [-inf, A) and (E, inf] intervals
    else:
        return 0.0


class Solution(Protocol):
    """ From the evolutionary techniques perspective, a solution is essentialy a fenotype.

    Protocol class used as an implicit interface 
    """
    project: proj.Project
    workload: Workload
    # result: Dict[str, FuncResult]
    fitness: float
    parameters: Dict[str, ApplyParam]
    # _evaluate_set: Set[str]

    def __init__(self, project: proj.Project, workload: Workload) -> None:
        """ Construction
        ================
        Initialize a solution for a specific (project, workload) tuple.
        """

    def mp_extract(self) -> Dict[str, Any]:
        """ Multiprocess extraction
        =========================== 
        Extract essential data from the solution that will be serialized for the interprocess 
        communication. 
        
        Use to propagate the final Solution object back to the main process from the workers.
        """
    
    def mp_merge(self, mp_data: Mapping[str, Any]) -> None:
        """ Multiprocess merging
        ========================
        Merge deserialized essential data obtained through interprocess communication with the
        given Solution object, thus updating it.

        Use to update the original Solution object present in the main process.
        """
    
    def apply(self, apply_func: ApplySignature, **params: ApplyParam) -> Solution:
        """ Application of a Genotype mapping
        =====================================
        Apply a genotype -> fenotype mapping function (with given parameters) to obtain an
        actual individual in the problem domain.
        """
    
    def evaluate_fitness(self, fitness_func: FitnessSignature) -> float:
        """ Evaluate the Fitness
        ========================
        Use the supplied function to evaluate the fitness of a fenotype obtained from the
        'apply' function.
        """
    
    def apply_and_eval(
            self, 
            apply_func: ApplySignature, 
            fitness_func: FitnessSignature, 
            **apply_params: ApplyParam
    ) -> float:
        """ Apply and Evaluate
        ======================
        Combines the Apply and Evaluate step into one for convenience.
        """
    
    def to_dataframe(self) -> pd.DataFrame:
        """ Transform to a DataFrame
        ============================
        Transform a Solution into a Pandas DataFrame representation.
        """


class FuncResult:
    """ Representation of a single project function sa part of the resulting fenotype. The
    FuncResult is associated to a single workload, which is, however, not referenced in the object.

    We evaluate the fitness of each project function separately and then aggregate the fitness
    values to obtain the final fitness value of the whole fenotype.

    :ivar name: name of the function
    :ivar calls: call count of the function before sampling
    :ivar sampled: call count after sampling
    :ivar fitness: the fitness value associated with the current sampled call count
    :ivar interval: precomupted intervals of function calls used to compute the fitness value
    """
    __slots__ = 'name', 'calls', 'sampled', 'fitness', 'interval'

    def __init__(self, name: str, calls: int) -> None:
        """ Constructor

        :param name: name of the function
        :param calls: call count before sampling application
        """
        self.name: str = name
        self.calls: int = calls
        self.sampled: int = 1  # Even after sampling, functions have at least one call
        self.fitness: float = 0.0
        self.interval: CallsInterval = self._compute_interval()
    
    def _compute_interval(self) -> CallsInterval:
        """ Precompute the interval used in the fitness evaluation.

        :return: the computed interval of [A, B, D, E] values
        """
        # Get the ideal call count (bounded) of one half
        expected = min(self.calls // 2, MAX_CALLS)
        # Identify the subintervals used in the fitness function
        return (
            MIN_CALLS,
            max(int(expected - expected * ACCEPTABLE_DEVIATION), MIN_CALLS),
            math.ceil(expected + expected * ACCEPTABLE_DEVIATION),
            # Update the last interval if the maximum calls is lower than the deviation interval
            max(math.ceil(expected + expected * ACCEPTABLE_DEVIATION), min(MAX_CALLS, self.calls))
        )


class SamplingSolution:
    """ An implementation of a solution class for the Problem A (initial sampling estimation).

    :ivar project: a reference to the associated project
    :ivar workload: a reference to the associated workload
    :ivar result: function name -> function result mapping
    :ivar fitness: the total fitness of the fenotype
    :ivar parameters: used apply parameters
    :ivar _evaluate_set: precomputed set of functions to apply and evaluate
    """
    __slots__ = 'project', 'workload', 'result', 'fitness', 'parameters', '_evaluate_set'

    def __init__(self, project: proj.Project, workload: Workload) -> None:
        """ Constructror

        :param project: reference to the associated project
        :param workload: reference to the associated workload
        """
        self.project: proj.Project = project
        self.workload: Workload = workload
        self.result: Dict[str, FuncResult] = {}
        self.fitness: float = 0.0
        self.parameters: Dict[str, ApplyParam] = {}
        self._evaluate_set: Set[str] = set()

        # Initialize the result dictionary
        for funcs in self.project.call_graph:
            calls = self.workload.get_call_counts(funcs.keys())
            self.result.update({
                name: FuncResult(name, call_count) for name, call_count in calls.items()
            })
        # Initialize the evaluation set
        self._evaluate_set = set(name for name, func in self.result.items() if func.calls > 1)

    def mp_extract(self) -> Dict[str, Any]:
        """ Multiprocess extraction implementation.

        :return: essential solution data to transfer back to the main process
        """
        return {
            'project': self.project.name,
            'workload': self.workload.name,
            'result': self.result,
            'fitness': self.fitness,
            'parameters': self.parameters
        }
    
    def mp_merge(self, mp_data: Mapping[str, Any]) -> None:
        """ Multiprocess merging implementation.

        :param mp_data: dict-like structure of solution received from another process 
        """
        self.result = mp_data['result']
        self.fitness = mp_data['fitness']
        self.parameters = mp_data['parameters']

    def apply(self, apply_func: ApplySignature, **params: ApplyParam) -> SamplingSolution:
        """ Genotype application implementation.

        :param apply_func: supplied application function
        :param params: application function parameters

        :return: updated Solution object
        """
        self.fitness = 0.0
        self.parameters = params
        # Iterate project function by level value (i.e., all function from level 0, 1, 2, ...)
        for lvl, funcs in enumerate(self.project.call_graph):
            threshold = apply_func(lvl, **params)
            threshold = 1 if threshold < 1 else threshold
            # Update the new sampled call count based on the threshold
            for func in funcs:
                if func in self._evaluate_set:
                    self.result[func].sampled = ((self.result[func].calls - 1) // threshold) + 1
        return self
    
    def evaluate_fitness(self, fitness_func: FitnessSignature) -> float:
        """ Fitness evaluation implementation.

        :param fitness_func: fitness evaluation function

        :return: the computed fitness value
        """
        self.fitness = 0.0
        # Iterate all functions that can be evaluated
        for name in self._evaluate_set:
            # Compute individual fitness for each function and sum it
            self.result[name].fitness = fitness_func(self.result[name])
            self.fitness += self.result[name].fitness
        return self.fitness
    
    def apply_and_eval(
            self, 
            apply_func: ApplySignature, 
            fitness_func: FitnessSignature, 
            **apply_params: ApplyParam
    ) -> float:
        """ Apply and evaluate combination.

        :param apply_func: supplied application function
        :param params: application function parameters
        :param fitness_func: fitness evaluation function

        :return: the computed fitness value
        """
        return self.apply(apply_func, **apply_params).evaluate_fitness(fitness_func)

    def to_dataframe(self) -> pd.DataFrame:
        """ Dataframe transformation implementation.

        :return: constructed dataframe representing the solution
        """
        return pd.DataFrame(
            [
                [
                    self.project.name, self.workload.name, r.name, r.calls, r.sampled, r.fitness, 
                    self.project.call_graph.cg[r.name].level
                ]
                for r in self.result.values()
            ],
            columns=['project', 'workload', 'func', 'original', 'sampled', 'fitness', 'level']
        )


class SolutionSet:
    """ A collection of solutions for a specific suite. Generally, the SolutionSet is used to 
    define the projects and workloads that will be used in the evolutionary computations. 
    
    E.g., when supplying 3 projects and all of their workloads for a sampling estimation in one 
    SolutionSet, one result in total (i.e., estimation function and its parameters) will be 
    provided for all of the 3 projects and their workloads. When searching for a result on a
    per-project basis, each project should have its own SolutionSet.

    :ivar sol_type: the actual specific Solution class to use (e.g., SamplingSolution)
    :ivar solutions: project -> workload -> Solution mapping
    :ivar plot_params: additional plotting parameters, necessary for properly reconstructing the
                       plots when loading a previously saved SolutionSets
    """
    __slots__ = 'sol_type', 'solutions', 'plot_params'

    def __init__(self, solution_type: Type[S], suite: Optional[proj.Suite] = None) -> None:
        """ Constructor

        :param solution_type: specific Solution class to use (e.g., SamplingSolution)
        :param suite: when supplied, the SolutionSet will be constructed such that it covers
                      all of the projects and workloads in the suite. 
        """
        self.sol_type: Type[S] = solution_type
        self.solutions: Dict[str, Dict[str, Solution]] = {}
        # If suite is supplied, initialize the solutions mapping
        if suite is not None:
            self.solutions = {
                project.name: {
                    wl.name: self.sol_type(project, wl) for wl in project
                } for project in suite
            }
        self.plot_params: Dict[str, Any] = {}
    
    def add(self, project: proj.Project, wl: Workload) -> SolutionSet:
        """ Add additional project and workload to the solution set. Note that if a Solution 
        for the (project, workload) combination already exists, it will be overwritten.

        :param project: a project reference
        :param wl: a workload reference

        :return: updated SolutionSet object
        """
        self.solutions.setdefault(project.name, {})[wl.name] = self.sol_type(project, wl)
        return self

    def remove(self, projects: Iterable[str]) -> None:
        """ Remove all Solutions associated with the given projects.

        :param projects: a collection of projects that should have all of their Solutions removed
        """
        for project in projects:
            del self.solutions[project]

    def mp_update(self, mp_solution: Mapping[str, Any]) -> None:
        """ Update the SolutionSet with a solution obtained from other process.

        :param mp_solution: a dict-like solution object
        """
        self[mp_solution['project'], mp_solution['workload']].mp_merge(mp_solution)

    def set_plot_param(self, key: str, value: Any) -> None:
        """ Set an additional plot parameter associated with the SolutionSet.

        :param key: parameter name
        :param value: parameter value
        """
        self.plot_params[key] = value

    def save(self, file_name: Optional[str]) -> None:
        """ Save the SolutionSet under a specified file name.

        :param file_name: if set, the SolutionSet will be saved under the given file name.
        """
        loader.save_object(self, file_name, loader.EXPERIMENTS_LOC)

    @classmethod
    def load(cls, file_path: str) -> SolutionSet:
        """ Load a previously saved SolutionSet found under the given file path.

        :param file_path: path to the file to load 
        """
        return loader.load_object(file_path)

    def __len__(self) -> int:
        """ Length protocol implementation.

        :return: number of individual Solutions found in the SolutionSet
        """
        return sum(len(project_solutions) for project_solutions in self.solutions.values())

    def __getitem__(self, proj_workload: Tuple[str, str]) -> Solution:
        """ Retrieve the Solution object identified by project and workload names.

        :param proj_workload: (project name, workload name) combination
        
        :return: the Solution stored under the (project, workload) combination, if any
        """
        return self.solutions[proj_workload[0]][proj_workload[1]]
    
    def __iter__(self) -> Generator[Solution, None, None]:
        """ Iteration protocol implementation.

        :return: generator of individual solutions sorted acording to the underlying mapping 
        """
        return (sol for wl_map in self.solutions.values() for sol in wl_map.values())


def set_per_workload(
    suite: proj.Suite, solution_type: Type[S]
) -> Generator[SolutionSet, None, None]:
    """ Construct a separate SolutionSet for each (project, workload) combination.

    :param suite: suite that will be used to generate the per-project, per-workload SolutionSets
    :param solution_type: specific type of the Solution

    :return: generator of per-project, per-workload SolutionSets
    """
    for project in suite:
        for workload in project:
            yield SolutionSet(solution_type).add(project, workload)

def set_per_project(
    suite: proj.Suite, solution_type: Type[S]
) -> Generator[SolutionSet, None, None]:
    """ Construct a separate SolutionSet for each project. Each such Set will contain a Solution
    for each workload in the given project.

    :param suite: suite that will be used to generate the per-project SolutionSets
    :param solution_type: specific type of the Solution

    :return: generator of per-project SolutionSets
    """
    for project in suite:
        # Create a new SolutionSet for each project
        solutions = SolutionSet(solution_type)
        # Populate the Set with Solutions for all workloads
        for workload in project:
            solutions.add(project, workload)
        yield solutions

def merge_sets(sets: Sequence[SolutionSet]) -> SolutionSet:
    """ Merge multiple SolutionSets into one. Useful for situations, where e.g., separate 
    SolutionSets exist for each project (since the goal was to compute optimal results on a 
    per-project basis) and we want to plot and store them together.
    """
    # Handle edge-case where no sets are provided
    if not sets:
        raise RuntimeError("No SolutionSets provided for the merging operation.")

    # Create a copy of the first SolutionSet
    base: SolutionSet = copy.copy(sets[0])
    # Add all of the solutions found in the remaining solution sets
    for solution_set in sets[1:]:
        # Check that the solution types are the same
        if solution_set.sol_type != base.sol_type:
            raise RuntimeError(
                f'Different solution types ({solution_set.sol_type} != {base.sol_type})'\
                ' encountered in the provided set!'
            ) 
        for s in solution_set:
            base.solutions.setdefault(s.project.name, {})[s.workload.name] = s
    return base
