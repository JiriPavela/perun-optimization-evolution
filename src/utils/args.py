""" Argument parsing and storage module. The arguments are parsed using the standard library 
'argparse' module.

Note that we distinguish the following:
 - Case:       a specific problem pattern that can be solved (using evolutionary computation 
               techniques) for any / all projects. E.g., finding the optimal parameter for linear 
               / logarithmic / quadratic / ... function to estimate the initial sampling employed 
               by the Dynamic Sampling technique. The actual problem patterns can be found in the 
               'cases' module.
 
 - Experiment: an actual solution of a specific case for specific project(s). One case is generally
               expected to have multiple saved experiments, however, one experiment can only relate
               to precisely one case. The available experiments can be found in the 'storage' dir.
"""
import os
import glob
import re
import argparse
import psutil
from typing import List, Set, Optional, Iterable

import cases as cases
import project.project as project
import utils.values as values
import show.plotter as plotter


class Arguments:
    """ The argument parsing and storage class.

    Currently, the supported CLI is structured as:
    prototype.py
      [-p <PLOT_TYPE>] [-x <EXCLUDE PROJECT> [-x ...]] {load, solve, list} ...
        load   <experiment> [<experiment>, ...] [-e <experiment> [<experiment>, ...]]
        solve  <case> [<case>, ...] [-s <file name>] [-w <no. worker processes>] [-g <goal>]
        list   [-c] [-e]
    

    :ivar plot: the requested type of plotting output
    :ivar plot_func: the actual function that performs the plotting
    :ivar exclude: a collection of projects that are to be excluded from solving (cases) or 
                   plotting (experiments)
    :ivar cmd: user-selected subcommand to execute
    :ivar experiments: a collection of experiments to load and plot
    :ivar cases: a collection of cases to solve
    :ivar save_file: a prefix of the saved file names
    :ivar workers: number of worker processes to use
    :ivar list_cases: indicates that the supported cases should be listed to the user
    :ivar list_experiments: indicates that the available experiments should be listed to the user

    """
    def __init__(self) -> None:
        """ Constructor
        """
        args = Arguments._parse()
        # General attributes
        self.plot: plotter.PlotType = plotter.PlotType(args.plot)
        self.plot_func: plotter.PlotCallable = plotter.PlotType.to_func(self.plot)
        self.exclude: Set[str] = set(args.exclude)
        # Selected subcommand
        self.cmd: str = args.subcommand
        # 'Load' attributes
        self.experiments: Set[str] = Arguments._flatten(getattr(args, 'experiment', []))
        # 'Solve' attributes
        self.cases: Set[str] = Arguments._flatten(getattr(args, 'case', []))
        self.save_file: Optional[str] = getattr(args, 'save', None)
        self.workers: int = getattr(args, 'workers', psutil.cpu_count(True) - 1)
        # 'List' attributes
        self.list_cases: bool = getattr(args, 'cases', False)
        self.list_experiments: bool = getattr(args, 'experiments', False)
        # List cases and experiments if no flag is given
        if not self.list_cases and not self.list_experiments:
            self.list_cases = True
            self.list_experiments = True

    def file_name(self, case: str) -> Optional[str]:
        """ Build the filename for a specific case. The names are created as <prefix>_<case name>

        :param case: name of the case to create the filename for

        :return: filename or None if saving is not enabled
        """
        if self.save_file is None:
            return None
        return f'{self.save_file}_{case}'
        
    @staticmethod
    def _flatten(nested: Iterable[Iterable[str]]) -> Set[str]:
        """ Flattens nested iterables of strings into a single set.

        :param nested: two levels of nested iterables of strings, e.g.: [['ab', 'cd'], ['ef']]

        :return: flattened set of strings, e.g.: {'ab', 'cd', 'ef'}
        """
        return set().union(*map(set, nested))  # type: ignore  # https://github.com/python/mypy/issues/6697

    @staticmethod
    def _parse() -> argparse.Namespace:
        """ Parse the user-supplied CLI arguments into a argparse Namespace object.

        :return: constructed Namespace object with parsed arguments
        """

        def shell_glob(arg_value: str) -> List[str]:
            """ Obtains all matching file paths from the experiment directory based on a filename 
            expression (glob).

            :param arg_value: the filename expression to evaluate

            :return: a list of matched file paths 
            """
            return glob.glob(os.path.join(values.EXPERIMENTS_DIR, arg_value))

        def case_regex(arg_value: str) -> List[str]:
            """ Obtains supported case names matching a supplied regex.

            :param arg_value: the regex to evaluate

            :return: a list of matching cases
            """
            pattern = re.compile(arg_value)
            return [case for case in cases.get_supported_cases() if pattern.match(case)]
        
        def workers_range(arg_value: str) -> int:
            """ Convert and normalize the number of worker processes. Generally, we impose no
            upper limit on the number of processes to run. However, numbers in interval of [0, 1]
            indicate single process computation, since it is generally faster than controller + 
            worker duo.

            :param arg_value: the number of worker processes to spawn

            :return: the normalized number of workers
            """
            # Convert to int
            i_arg_value = int(arg_value)
            # One worker does not make sense, it is faster to just use single main process
            if i_arg_value <= 1:
                i_arg_value = 0
            return i_arg_value
            
        # Obtain the number of physical and logical cores
        cores_phys, cores_logic = psutil.cpu_count(False), psutil.cpu_count(True)
        # Fallback to a single core variant if no information is available
        for core in [cores_phys, cores_logic]:
            if core is None:
                core = 1
        
        # Top-level parser
        parser = argparse.ArgumentParser(description='Evolutionary computation experiments.')
        # Top level options
        parser.add_argument(
            '-p', '--plot', choices=plotter.PlotType.supported(), default='grid', 
            help='plot the solutions in a 2x2 grid or as a single plot per page'
        )
        parser.add_argument(
            '-x', '--exclude', action='append', choices=project.supported_projects(), default=[], 
            help='do not solve or plot certain projects'
        )
        subparsers = parser.add_subparsers(required=True, dest='subcommand')

        # Subparser for the 'load' command
        load_parser = subparsers.add_parser(
            'load', 
            help='load a previously saved experument(s)'
        )
        load_parser.add_argument(
            'experiment', nargs='+', type=shell_glob, action='extend', 
            help='experiments to load (can use globbing)'
        )

        # Subparser for the 'solve' command
        solve_parser = subparsers.add_parser(
            'solve', 
            help='solve a selected case(s)'
        )
        solve_parser.add_argument(
            'case', nargs='+', type=case_regex, action='extend', 
            help='cases to solve'
        )
        # Solve command options
        solve_parser.add_argument(
            '-s', '--save', 
            help='save the obtained solution into a file <save>_<case>.bz2'
            )
        solve_parser.add_argument(
            '-w', '--workers', type=workers_range, default=cores_logic - 1, 
            help=f'number of worker processes to use in parallel (this CPU has {cores_phys}/' \
                 '{cores_logic} physical/logical cores) (default: {cores_logic - 1})'
        )
        solve_parser.add_argument(
            '-g', '--goal', 
            help='the optimization goal'
            )

        # Subparser for the 'list' command
        list_parser = subparsers.add_parser(
            'list', 
            help='list cases and/or experiments'
        )
        # List command options
        list_parser.add_argument(
            '-c', '--cases', action='store_true', 
            help='list the available cases'
        )
        list_parser.add_argument(
            '-e', '--experiments', action='store_true', 
            help='list the available experiments'
        )

        return parser.parse_args()

