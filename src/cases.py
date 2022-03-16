""" Module containing a collection of cases that can be solved. Clarification of the term 'case'
can be found in the 'args' module. Currently, the cases solve one of two problems:

  A) Finding the optimal function and its parameters for estimating the initial sampling thresholds
     for profiled functions (See Perun's Dynamic Sampling method).
  B) WIP: Finding the optimal combination of parameters across Perun's optimization techniques to 
     achieve a desired optimization gain of the Perun's profiling process.
"""
from __future__ import annotations
import sys
from inspect import getmembers, isfunction
from typing import List, Dict, Protocol, TYPE_CHECKING

import project.project as proj
import evolution.solution as sol
import evolution.ec as ec
import evolution.ec_utils as ecu
from utils.logger import glob_logger

# Type checking imports
if TYPE_CHECKING:
    from utils.args import Arguments


class CaseCallable(Protocol):
    """ Prototype of case functions (callables).
    """
    def __call__(self, suite: proj.Suite, arg: Arguments) -> sol.SolutionSet:
        """ Call special method.

        :param suite: suite of projects that are solved in the given case
        :param arg: parsed CLI parameters

        :return: computed solution for each (project, workload) pair
        """
        ...


REPEATS = 30
GOALS = [0.1, 0.25, 0.5, 0.75, 0.9]
ES_COMBINATIONS = [(4, 8), (8, 16), (5, 5), (10, 10), (20, 20)]
FITNESS_FUNCS = [
    ('t&d', ecu.fitness_str_time_data),
    ('t', ecu.fitness_str_time),
    ('d', ecu.fitness_str_data)
]
EVALUATIONS = 10000


# TODO: add cases for:
#  - Log variant with further precision improvement through GP
#  - Solving parameter combinations

# SAMPLING PROBLEM section
# ========================

def case_basic_log_per_project(suite: proj.Suite, arg: Arguments) -> sol.SolutionSet:
    """ Problem A
    -----------------
    Find optimal 'base' parameter in function 'base * log(level)' for each project individually. 
    """
    solutions_set = list(sol.set_per_project(suite, sol.SamplingSolution))
    for solutions in solutions_set:
        ec.basic_ea(
            solutions, ec.EvolutionParameters(), ecu.basic_log, ecu.fitness_sampling, arg.workers
        )
    result = sol.merge_sets(solutions_set)
    result.set_plot_param('suptitle', r'local [$f = base \cdot log(level)$]')
    return result


def case_basic_quad_per_project(suite: proj.Suite, arg: Arguments) -> sol.SolutionSet:
    """ Problem A
    -----------------
    Find optimal 'base' parameter in function 'base * level^2' for each project individually.
    """
    solutions_set = list(sol.set_per_project(suite, sol.SamplingSolution))
    for solutions in solutions_set:
        ec.basic_ea(
            solutions, ec.EvolutionParameters(), ecu.basic_quad, ecu.fitness_sampling, arg.workers
        )
    result = sol.merge_sets(solutions_set)
    result.set_plot_param('suptitle', r'local [$f = base \cdot level^2$]')
    return result


def case_basic_lin_per_project(suite: proj.Suite, arg: Arguments) -> sol.SolutionSet:
    """ Problem A
    -----------------
    Find optimal 'base' parameter in function 'base * level' for each project individually.
    """
    solutions_set = list(sol.set_per_project(suite, sol.SamplingSolution))
    for solutions in solutions_set:
        ec.basic_ea(
            solutions, ec.EvolutionParameters(), ecu.basic_lin, ecu.fitness_sampling, arg.workers
        )
    result = sol.merge_sets(solutions_set)
    result.set_plot_param('suptitle', r'local [$f = base \cdot level$]')
    return result


def case_basic_exp_per_project(suite: proj.Suite, arg: Arguments) -> sol.SolutionSet:
    """ Problem A
    -----------------
    Find optimal 'base' parameter in function 'base^level' for each project individually.
    """
    solutions_set = list(sol.set_per_project(suite, sol.SamplingSolution))
    for solutions in solutions_set:
        ec.basic_ea(
            solutions, 
            ec.EvolutionParameters(attr_low=1.0, attr_high=2.0), 
            ecu.basic_exp, ecu.fitness_sampling, arg.workers
        )
    result = sol.merge_sets(solutions_set)
    result.set_plot_param('suptitle', rf'local [$f = base^{{level}}$]')
    return result


def case_basic_exp_forall(suite: proj.Suite, arg: Arguments) -> sol.SolutionSet:
    """ Problem A
    -----------------
    Find optimal 'base' parameter in function 'base^level' for all projects globaly.
    """
    solutions = sol.SolutionSet(sol.SamplingSolution, suite)
    base, global_fitness = ec.basic_ea(
        solutions, 
        ec.EvolutionParameters(attr_low=1.0, attr_high=2.0), 
        ecu.basic_exp, ecu.fitness_sampling, arg.workers
    )
    solutions.set_plot_param(
        'suptitle', 
        rf'global [$f = base^{{level}}$] [$base = {base:.3f}$] [$fitness = {global_fitness:.3f}$]'
    )
    return solutions


# OPTIMIZATION STRENGTH section
# =============================


# TODO: refactor heavily, generalize
def case_strength_comma(suite: proj.Suite, arg: Arguments) -> sol.SolutionSet:
    # ES-combination -> goals -> fitness -> repeat
    for mu, lmbda in ES_COMBINATIONS:
        for opt_goal in GOALS:
            for fitness_name, fitness_func in FITNESS_FUNCS:
                for repeat in range(REPEATS):
                    solutions_set = list(sol.set_per_project(suite, sol.OptimizationSolution, opt_goal=opt_goal))
                    glob_logger.new_session(f'({mu},{lmbda})_{fitness_name}', repeat)
                    for solutions in solutions_set:
                        ec.opt_es_comma(
                            solutions,
                            # TODO: compute generations dynamically
                            ec.EvolutionParameters(generations=compute_generations(mu, lmbda), popsize=mu, pop_lambda=lmbda),
                            ecu.apply_opt_strength, fitness_func, 0
                        )
                    result = sol.merge_sets(solutions_set)
    return result


def case_strength_plus(suite: proj.Suite, arg: Arguments) -> sol.SolutionSet:
    # ES-combination -> goals -> fitness -> repeat
    for mu, lmbda in ES_COMBINATIONS:
        for opt_goal in GOALS:
            for fitness_name, fitness_func in FITNESS_FUNCS:
                for repeat in range(REPEATS):
                    solutions_set = list(sol.set_per_project(suite, sol.OptimizationSolution, opt_goal=opt_goal))
                    glob_logger.new_session(f'({mu}+{lmbda})_{fitness_name}', repeat)
                    for solutions in solutions_set:
                        ec.opt_es_plus(
                            solutions,
                            # TODO: compute generations dynamically
                            ec.EvolutionParameters(generations=compute_generations(mu, lmbda), popsize=mu, pop_lambda=lmbda),
                            ecu.apply_opt_strength, fitness_func, 0
                        )
                    result = sol.merge_sets(solutions_set)
    return result


def compute_generations(mu: int, lmbda: int) -> int:
    # mu + (generations * lmbda) ~ EVALUATIONS
    return int((EVALUATIONS - mu) / lmbda)


def get_supported_cases() -> List[str]:
    """ Retrieve the list of currently available cases to solve.
    
    :return: list of case names (each name without the initial 'case_' prefix)
    """
    return list(CASES.keys())


def get_case_func(case: str) -> CaseCallable:
    """ Get the case function corresponding to the given case name, 
    i.e., mapping <case name> -> <case function>

    :return: callable object (function) representing the case
    """
    return CASES[case]


# Case name -> Case function mapping
# Case functions are automatically extracted from the current module
CASES: Dict[str, CaseCallable] = {
    name[5:]: obj for name, obj in getmembers(sys.modules[__name__], isfunction) 
    if name.startswith('case_')
}



# Visualization
# =============
# 1) Boxplot comparison of comma, plus ES for different combinations of mu, lambda (y = fitness)
#    a) each combination will be measured repeteadly
#    b) different goal
# 2) Fitness progress per generation
#    a) with visualized variance
# 3) Show results for single objective (time, space), aggregated (0.5 + 0.5), dynamic fitness
# Data format:
# [name, generation, repeat, fitness, str_string, data_ratio, time_ratio] ?
#





# Optimization strength cases
# For 1 strength value
#   State-space:
#    - CCSDS:   SB  x  DB  x  CGP  x  DT  x  DS  x  TS
#    indiv:      6      3      7     100    100     100
#    pipel:      6      3      7     100    100      1     =  1.260.000
#
#    - CPython: SB  x  DB  x  CGP  x  DT  x  DS  x  TS
#    indiv:      1      3     100    100    100     100
#    pipel:      1      3     100    100    100      1     =  3.000.000
# For complete solution, an additional x100

# Population
# ES: 8 - 32, 4, 10, 20, 30
# GA: 100s -> 20 - 50
# ES: 5 + 10, 5, 10, 4 + 8, 8 + 16, 16 + 32

# !!! No. Generations * No. evaluated individuals in each Gen. = const for all EC approaches

# Approaches:
#  - (mu, lambda), mutation   **
#  - (mu + lambda), mutation
#  - [[ (mu, lambda), mutation + crossover
#  - (mu + lambda), mutation + crossover ]]
#  - [[ (mu, lambda), mutation + selfadaptation
#  - (mu + lambda), mutation + selfadaptation
#  - (mu, lambda), mutation + crossover + self-adaptation
#  - (mu + lambda), mutation + crossover + self-adaptation ]]
#  - GA, mutation + crossover  **
# 
# Mutation operators:
#  - BitFlip  XX
#  - UniformInt  
#  - Small step increments
#  - Gaussian distr. with int discretization?
# 
# Crossover operators:
#  - OnePoint  **
#  - TwoPoint
# 
# Selection:
#  - Ranking (single objective)
#  - Tournament (GA)
#  - NSGA 2 (multiobjective)
#
# Self-adaptation:
#  - BitFlip: Changing the number of flipped bits
#  - UniformInt: ???
#  - Gaussian: Changing the mean and sigma
# 
# Fitness:
#  - For each (non-error) criterion, the fitness should look like this
#
#    f |
#    i |            .
#    t |           /.\
#    n |          / . \
#    e |         /  .  \
#    s |  ______/   .   \______
#    s +--======+---+---+======----> 
#               A   B   C       optimization %
#
#             { 0.0                     iff  opt == A
#             { (1 / (B - A)) * X       iff  A < opt /\ opt < B
#   fitness = { 1.0                     iff  opt == B
#             { 1 + (1 / (B - C)) * X   iff  B < opt /\ opt < C
#             { 0.0                     iff  opt == C


# Fitness (single objective):
#  - Objective: Time
#
# Fitness (multiobjective):
#  - Objectives: Time, Space, Hotspot cov., Hotspot dist., (Models (sampling))
#     - Models should not figure here due time constraints and difficult approximation (simulation)
