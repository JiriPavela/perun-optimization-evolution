""" The entry program module. Contains basic logic for executing each CLI command.

Throughout this module, the terms "case" and "experiment" are used. For more details on
their respective meaning, visit the 'args' module. 
"""

import sys
import inspect as ins
from timeit import default_timer as timer

import solution as sol
import project as proj
import loader
import cases
import args


def solve(arg: args.Arguments) -> None:
    """ Solve cases - i.e., apply evolutionary computation techniques on a given problem with 
    a given input - specified in the Arguments object.

    :param arg: Argument object with parsed CLI parameters.
    """
    # Build a Suite object comprised of the selected projects
    suite = proj.Suite(set(proj.supported_projects()) - arg.exclude)
    # Solve each specified case
    for case in arg.cases:
        print(f'Solving case "{case}"')
        start = timer()
        # The actual case solving code
        result = cases.get_case_func(case)(suite, arg)
        end = timer()
        print(f'Solved case "{case}" in {end - start}s')
        # Plot the results
        arg.plot_func(result)
        # Save the obtained results if needed
        result.save(arg.file_name(case))


def load(arg: args.Arguments) -> None:
    """ Load and plot previously solved and saved results (referred to as 'experiments').

    :param arg: Argument object with parsed CLI parameters.
    """
    for experiment in arg.experiments:
        # Load the saved experiments (each experiment is saved as a separate SolutionSet)
        solutions = sol.SolutionSet.load(experiment)
        # Filter projects that are to be excluded
        solutions.remove(arg.exclude)
        # Plot the loaded experiment
        arg.plot_func(solutions)


def list(arg: args.Arguments) -> None:
    """ List the supported cases to solve and saved experiments to load.

    :param arg: Argument object with parsed CLI parameters.
    """
    # We list the cases
    if arg.list_cases:
        print('Cases:')
        for case in cases.get_supported_cases():
            print(f'\t{case}')
        print()
    # We list the experiments
    if arg.list_experiments:
        print('Experiments:')
        for exp in loader.list_saves(loader.EXPERIMENTS_LOC):
            print(f'\t{exp}')

def __map(arg: args.Arguments) -> None:
    """ Map the requested CLI command to an actual function to execute. Note that the name of the
    command must match a function present in this module!

    :param arg: Argument object with parsed CLI parameters.
    """
    # Build a command -> function map and index it with the command string
    cmd_func = {
        func_name: func for func_name, func in ins.getmembers(sys.modules[__name__], ins.isfunction)
        if not func_name.startswith('__')
    }[arg.cmd]
    # Run the function
    cmd_func(arg)


if __name__ == '__main__':
    # Program starting point
    # This guard is also neccessary because of the potential multiprocessing usage to prevent
    # a recursive process spawning
    arg = args.Arguments()
    __map(arg)

