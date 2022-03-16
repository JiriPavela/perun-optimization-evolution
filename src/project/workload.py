""" Module containing workload-related objects. Workloads refer to specific inputs of a project
(i.e., different programs for Python2&3, different images for CCSDS etc.). Generally, workloads
do not contain back-reference to the corresponding project as it's not needed. 
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Iterator, TYPE_CHECKING

import utils.values as values
import resources.loader as load
import resources.call_counts as cc
import optimization.metrics as ms
import optimization.optimizations as opt

if TYPE_CHECKING:
    import resources.call_graph as cg


# Workload name -> Workload object
WorkloadMap = Dict[str, "Workload"]

# TODO: find out if we can initialize workload right away
# TODO: we could, just need a default load of ProjectOpt and ProjectMetrics in case the jsons are
#       not available.
class Workload:
    """ Workload object containing relevant data or statistics about specific workload. 
    Specifically, the workload object should contain data relevant to either Problem A, 
    Problem B (see the 'cases' module) or both - note that data for at least one Problem 
    must be present.

    :ivar name: identifier of the workload
    :ivar call_counts: function -> call count mapping
    :ivar opt_data: structure containing Perun's workload-wise optimization computations and data
    :ivar metrics: metrics obtained from workload runs
    """
    __slots__ = 'name', 'call_counts', 'opt_data', 'metrics'
    
    def __init__(self, workload: str) -> None:
        """ Constructor

        :param project_name: name of the corresponding project
        :param workload: the workload identifier
        :param metrics: workload metrics as loaded from metrics file
        """
        self.name: str = workload
        # Used in the sampling cases
        self.call_counts: cc.CallCounts
        # Used in the optimization strength cases
        self.opt_data: opt.OptSteps
        self.metrics: ms.WorkloadMetrics
    
    def init_solution_data(
        self, call_counts: cc.CallCounts, opt_data: opt.OptSteps, 
        wl_metrics: ms.WorkloadMetrics
    ) -> None:
        self.call_counts = call_counts
        self.opt_data = opt_data
        self.metrics = wl_metrics


class WorkloadSet:
    """ A collection of workloads connected to a certain project. The workload set is built by 
    searching for relevant stats or profile files in the directory structure using the project name.

    :ivar project: name of the corresponding project
    :ivar workloads: mapping of workload name -> Workload object 
    """
    __slots__ = 'project', 'workloads'

    def __init__(self, project_name: str) -> None:
        """ Constructor

        :param project_name: name of the corresponding project
        """
        self.project: str = project_name
        self.workloads: WorkloadMap = self._build_workload_set(project_name)

    def init_call_counts(self) -> None:
        for wl_name, workload in self.workloads.items():
            stats = load.load_and_deflate(f'{values.DS_PREFIX}_{self.project}_{workload.name}', values.DS_DIR)
            workload.call_counts = cc.CallCounts(stats)

    def init_solution_data(self, call_graph: cg.CallGraph) -> None:
        proj_opt = opt.ProjectOpt(self.project, call_graph)
        proj_metrics = ms.ProjectMetrics(self.project)
        for wl_name, workload in self.workloads.items():
            stats = load.load_and_deflate(f'{values.DS_PREFIX}_{self.project}_{workload.name}', values.DS_DIR)
            call_counts = cc.CallCounts(stats)
            workload.init_solution_data(
                call_counts,
                opt.OptSteps(proj_opt, opt.WorkloadOpt(stats, call_counts)),
                proj_metrics[wl_name].compute_overheads(call_counts)
            )
            # print(f'\n{wl_name}\n=================')
            # workload.opt_data.project_opt.print_steps()
            # workload.opt_data.workload_opt.print_steps()
    
    def get_strength_limits(self) -> opt.StrConstraints:
        # The strength limits should be the same for all workloads
        # Get some element from the workload set
        return next(iter(self.workloads.values())).opt_data.get_strength_constraints()

    def __iter__(self) -> Iterator[Workload]:
        """ Implementation of the Iterator protocol.

        :return: iterator of the workloads (ordering based on the underlying mapping)
        """
        return iter(self.workloads.values())
    
    def __getitem__(self, workload: str) -> Workload:
        """ Retrieve a Workload object corresponding to the supplied workload name.

        :return: a Workload object
        """
        return self.workloads[workload]
    
    @staticmethod
    def _build_workload_set(project_name: str) -> WorkloadMap:
        """ Construct the workload name -> workload mapping.

        :param project_name: name of the project to construct the mapping for

        :return: the constructed workload mapping
        """
        # Obtain the files containing workload data for the given project 
        w_files = load.find_project_stats(project_name, values.DS_DIR, values.DS_PREFIX)
        # Obtain the project metrics
        workloads = {}
        # Parse and map each workload file
        for w_file in w_files:
            workload = w_file.split('_', maxsplit=2)[-1]
            workloads[workload] = Workload(workload)
        return workloads