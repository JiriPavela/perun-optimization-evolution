""" Module containing Project and project Suite structures. The Suite serves as a collection
of Projects (each comprised of generally multiple workloads) and is used when solving certain
cases (see 'cases' module) for multiple projects.
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Iterable, Generator, Optional

import resources.call_graph as cg
import project.workload as wl
import optimization.optimizations as opt


# Mapping of project name -> Project structure
ProjectMap = Dict[str, 'Project']
# Map of workload -> optimization mode -> [main exclusive times] (across more profiling rounds)
RunMap = Dict[str, Dict[int, List[int]]]


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
        # Initialize the solution-relevant data
        # self.workloads.init_solution_data(self.call_graph)
        self.workloads.init_call_counts()

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
