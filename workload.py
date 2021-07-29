""" Module containing workload-related objects. Workloads refer to specific inputs of a project
(i.e., different programs for Python2&3, different images for CCSDS etc.). Generally, workloads
do not contain back-reference to the corresponding project as it's not needed. 
"""
from __future__ import annotations
from typing import Dict, Iterator, Iterable

import loader as load


# Type aliases
# Function name -> call count
FuncCount = Dict[str, int]
# Workload name -> Workload object
WorkloadMap = Dict[str, "Workload"]


# Dynamic Stats directory and file prefix
DS_DIR = 'stats'
DS_PREFIX = 'ds'
PROF_DIR = 'profiles'


class Profile:
    # Main time no-prof
    # Main time full-prof
    # Pre-compute Dynamic baseline
    # Pre-compute Sampling
    def __init__(self, project_name: str, workload: str) -> None:
        profile = load.load_object(f'{project_name}_{workload}.pbz2')


class Workload:
    """ Workload object containing relevant data or statistics about specific workload. 
    Specifically, the workload object should contain data relevant to either Problem A, 
    Problem B (see the 'cases' module) or both - note that data for at least one Problem 
    must be present.

    :ivar name: identifier of the workload
    :ivar call_count: function -> call count mapping
    :ivar calls: summarized and aggregated Perun profile. The aggregation is neccessary
                 due to extreme memory requirements of storing the bare profile.  
    """
    __slots__ = 'name', 'call_count', 'calls'
    
    def __init__(self, project_name: str, workload: str) -> None:
        """ Constructor

        :param project_name: name of the corresponding project
        :param workload: the workload identifier
        """
        # Extract the workload string from 'ds_<project>_<workload>'
        self.name: str = workload
        # Used in the sampling cases
        self.call_count: FuncCount = self._parse_stats(
            load.load_and_deflate(f'{DS_PREFIX}_{project_name}_{workload}', DS_DIR)
        )
        # TODO: Used in the optimization strength cases
        # This needs some clever techniques to not run out of memory
        # Precompute the effects of removing or sampling each function? Should be much more compact
        # Allow that either call_count or calls can be missing, but not both
        self.calls = None
    
    # TODO: this might not make sense when both 'call_count' and 'calls' is present
    def __len__(self) -> int:
        """ Length protocol implementation.

        :return: number of unique function names present in the call_count mapping
        """
        return len(self.call_count)

    # TODO: this might not make sense when both 'call_count' and 'calls' is present
    def __getitem__(self, func_name: str) -> int:
        """ Retrieve call count of the given function.

        :param func_name: name of the function to lookup

        :return: recorded call count of the function
        """
        return self.call_count[func_name]
    
    def get_call_counts(self, func_list: Iterable[str]) -> Dict[str, int]:
        """ Retrieve a subset of the call_count mapping.

        :param func_list: list of function names to retrieve the call counts for

        :return: a subset of function name -> call count mapping
        """
        return {func: self[func] for func in func_list if func in self.call_count}

    @staticmethod
    def _parse_stats(stats: load.JsonType) -> FuncCount:
        """ Parse the Dynamic stats file and construct the function name -> call count mapping

        :param stats: the Dynamic stats in JSON format

        :return: constructed function name -> call count mapping
        """
        return {
            f_name: f_stats['count']
            for f_name, f_stats in stats['dynamic-stats']['global_stats'].items()
        }


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
        w_files = load.find_project_stats(project_name, DS_DIR, DS_PREFIX)
        workloads = {}
        # Parse and map each workload file
        for w_file in w_files:
            workload = w_file.split('_', maxsplit=2)[-1]
            workloads[workload] = Workload(project_name, workload)
        return workloads
