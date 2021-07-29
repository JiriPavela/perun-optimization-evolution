""" Module containing Project and project Suite structures. The Suite serves as a collection
of Projects (each comprised of generally multiple workloads) and is used when solving certain
cases (see 'cases' module) for multiple projects.
"""
from __future__ import annotations
import copy
from enum import Enum
from typing import List, Dict, Iterable, Generator, Optional, Any

import loader as load
import call_graph as cg
import workload as wl


# Mapping of project name -> Project structure
ProjectMap = Dict[str, 'Project']


# Optimization strength max value (i.e., 0 - 100)
OPT_STR_MAX = 100


# TODO: doc
class ProjectOptimization:
    # Pre-compute CG levels?
    # Pre-compute Loopus
    # Pre-compute Diff Tracing
    def __init__(self, project_name: str, call_graph: cg.CallGraph) -> None:
        self.cg_levels = self._precompute_levels(call_graph)
        self.complexities = self._precompute_complexities(project_name, call_graph)
        self.diffs = None
    
    @staticmethod
    def _precompute_levels(call_graph: cg.CallGraph) -> Dict[int, List[str]]:
        levels = {}
        step = call_graph.max_level / OPT_STR_MAX

        old_threshold = -1
        for opt_strength in range(OPT_STR_MAX + 1):
            level_threshold = call_graph.max_level - round(step * opt_strength)
            # Reference the previously computed collection of functions to filter
            levels[opt_strength] = levels.get(opt_strength - 1, [])
            # If the threshold has changed, extend the collection with additional filtered functions
            if level_threshold != old_threshold:
                levels[opt_strength] = copy.copy(levels[opt_strength]).extend(
                    ProjectOptimization._get_new_filtered(call_graph, level_threshold)
                )
                old_threshold = level_threshold
        return levels
    
    # TODO: refactor the final mapping in precompute functions
    @staticmethod
    def _precompute_complexities(project_name: str, call_graph: cg.CallGraph) -> Dict[int, List[str]]:
        # Init the function -> complexity mapping
        func_bounds = {
            func: Complexity.GENERIC for func in call_graph.cg.keys()
        }
        # Load the bounds profile and update the function -> complexity mapping
        bounds_profile = load.load_json(f'{project_name}_bounds.perf', wl.PROF_DIR)
        for res in bounds_profile['resource_type_map'].values():
            func_name = res['uid']['function']
            # Ignore local bounds records and functions that are never actually called
            # TODO: handle functions with same name from different modules that can also have
            # # different complexities - luckily, not the case with CCSDS 
            if res['type'] == 'total bound' and func_name in func_bounds:
                func_bounds[func_name] = Complexity.from_poly(res['class'])
        # Reverse the func_bounds mapping (i.e., to complexity -> functions) and convert it to list
        
        complexity_ordering = {c: order for order, c in enumerate(Complexity.opt_ordering())}
        # Additional empty list to avoid IndexError
        bounds = [
            [] for _ in range(len(complexity_ordering) + 1)
        ]
        for func_name, bound in func_bounds.items():
            bounds[complexity_ordering[bound]].append(func_name)
        # Create strength -> filtered functions mapping using the bounds_map
        step = len(complexity_ordering) / OPT_STR_MAX

        complexities = {}
        old_threshold = -1
        for opt_strength in range(OPT_STR_MAX + 1):
            complexity_thr = len(complexity_ordering) - round(step * opt_strength)
            complexities[opt_strength] = complexities.get(opt_strength - 1, [])
            if complexity_thr != old_threshold:
                complexities[opt_strength] = copy.copy(complexities[opt_strength]).extend(
                    bounds[complexity_thr]
                )
        return complexities

    @staticmethod
    def _get_new_filtered(call_graph: cg.CallGraph, level: int) -> List[str]:
        try:
            return list(call_graph.levels[level + 1].keys())
        except IndexError:
            return []
        



class Project:
    """ Project structure. Projects are identified by a name and keep track of associated
    call graph and workloads. It is assumed that project == program.

    :ivar name: name of the project
    :ivar call_graph: Call Graph structure of the project
    :ivar workloads: workloads collection associated with the project
    """
    __slots__ = 'name', 'call_graph', 'workloads'

    def __init__(self, project_name: str) -> None:
        """ Constructor

        :param project_name: name of the project
        """
        self.name: str = project_name
        self.call_graph: cg.CallGraph = cg.CallGraph(self.name)
        self.workloads: wl.WorkloadSet = wl.WorkloadSet(self.name)

    def __iter__(self) -> Generator[wl.Workload, None, None]:
        """ Iterator protocol implementation.

        :return: generator of project workloads
        """
        return (workload for workload in self.workloads)


class Suite:
    """ Project Suite structure. Used mostly for convenience when working with a collection
    of projects.
    """
    __slots__ = 'projects'

    def __init__(self, projects: Optional[Iterable[str]] = None) -> None:
        """ Constructor

        :param projects: a collection of projects to include in the suite. Projects can also
                         be supplied later through the 'add_project' method.
        """
        self.projects: ProjectMap = {}
        # Initialize the suite if possible
        if projects is not None:
            for project in projects:
                self.add_project(project)
    
    def __getitem__(self, project_name: str) -> Project:
        """ Retrieve the Project object identified by its name.

        :param project_name: project identifier

        :return: the associated Project object
        """
        return self.projects[project_name]
    
    def __len__(self) -> int:
        """ Length protocol implementation.

        :return: number of projects in the suite
        """
        return len(self.projects)
    
    def __iter__(self) -> Generator[Project, None, None]:
        """ Iterator protocol implementation.

        :return: generator of projects in the suite (ordering defined by the underlying mapping)
        """
        return (project for project in self.projects.values())

    def add_project(self, project_name: str) -> None:
        """ Add a new project to the suite. Replaces any project with the same name.

        :param project_name: name of the new project
        """
        self.projects[project_name] = Project(project_name)


# Collection of supported projects
def supported_projects() -> List[str]:
    return ['ccsds', 'cpython2', 'cpython3', 'vim', 'gedit']


class OrderedEnum(Enum):
    """ An ordered enumeration structure that ranks the elements so that they can be compared in
    regards of their order. Taken from:
        https://stackoverflow.com/questions/42369749/use-definition-order-of-enum-as-natural-order

    :ivar int order: the order of the new element
    """

    def __init__(self, *args: Any) -> None:
        """ Create the new enumeration element and compute its order.

        :param args: additional element arguments
        """
        try:
            # attempt to initialize other parents in the hierarchy
            super().__init__(*args)
        except TypeError:
            # ignore -- there are no other parents
            pass
        ordered = len(self.__class__.__members__) + 1
        self.order = ordered

    def __ge__(self, other: OrderedEnum) -> bool:
        """ Comparison operator >=.

        :param OrderedEnum other: the other enumeration element
        :return bool: the comparison result
        """
        if self.__class__ is other.__class__:
            return self.order >= other.order
        raise NotImplementedError

    def __gt__(self, other: OrderedEnum) -> bool:
        """ Comparison operator >.

        :param OrderedEnum other: the other enumeration element
        :return bool: the comparison result
        """
        if self.__class__ is other.__class__:
            return self.order > other.order
        raise NotImplementedError

    def __le__(self, other: OrderedEnum) -> bool:
        """ Comparison operator <=.

        :param OrderedEnum other: the other enumeration element
        :return bool: the comparison result
        """
        if self.__class__ is other.__class__:
            return self.order <= other.order
        raise NotImplementedError

    def __lt__(self, other: OrderedEnum) -> bool:
        """ Comparison operator <.

        :param OrderedEnum other: the other enumeration element
        :return bool: the comparison result
        """
        if self.__class__ is other.__class__:
            return self.order < other.order
        raise NotImplementedError


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

        :return list: CLI names of the supported complexities
        """
        return [complexity.value for complexity in Complexity]

    # TODO: doc
    @staticmethod
    def opt_ordering() -> List[Complexity]:
        # Necessary since the internal order parameter is for comparison of complexity
        # This order is reversed, indexed from 0 and contains the actual enum elements
        return reversed([complexity for complexity in Complexity])

    @staticmethod
    def max(values: Iterable[Complexity]) -> List[Complexity]:
        """ Compare a collection of Complexity values and select the one with maximum degree.

        :param collection values: the set of Complexity values

        :return Complexity: the Complexity object with the highest degree of polynomial
        """
        return sorted(values, key=lambda complexity: complexity.order, reverse=True)[0]

    @classmethod
    def from_poly(cls, polynomial: str) -> Complexity:
        """ Create a Complexity object from string representing a polynomial.

        :param str polynomial: a string representation of a supported polynomial

        :return Complexity: the corresponding Complexity object
        """
        return Complexity.map.get(polynomial, cls.GENERIC)


Complexity.map = {
    'O(1)': Complexity.CONSTANT,
    'O(n^1)': Complexity.LINEAR,
    'O(n^2)': Complexity.QUADRATIC,
    'O(n^3)': Complexity.CUBIC,
    'O(n^4)': Complexity.QUARTIC
}
