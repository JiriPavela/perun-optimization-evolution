from __future__ import annotations
import math
import random
from typing import Dict, Any, Iterator, Iterable, List, Optional, Sequence, TYPE_CHECKING, Tuple, Generic, TypeVar

from numpy import result_type

if TYPE_CHECKING:
    from evolution.solution import FuncResult
    import optimization.optimizations as opt
    from deap import base

B = TypeVar('B')

STR_UNIFORM_THRESHOLD = 10
SMALLSTEP_LIMIT = 3


class BestTracker(Generic[B]):
    def __init__(self):
        self.best: Optional[B] = None
    
    def get_best(self) -> B:
        if self.best is None:
            raise KeyError("No best individual stored!")
        return self.best
    
    def update(self, population: Sequence[B]):
        if self.best is None:
            self.best = population[0]
        for ind in population:
            if ind.fitness > self.best.fitness:
                self.best = ind


# SAMPLING PROBLEM section
# ========================

def fitness_sampling(func: FuncResult) -> List[float]:
    """ Computes the fitness value of the new sampled call count based on a function:
    
     f |
     i |            _____
     t |          //. . .\\
     n |         // . . . \\
     e |        //  . . .  \\
     s |  _____//   . . .   \\______
     s +--======+---+-+-+---+======-------> 
                A   B C D   E           calls
    
    Where
     A) The minimum number of calls
     B) The start of the "acceptable deviation interval"
     C) The ideal ("expected") number of sampled calls
     D) The end of the "acceptable deviation interval"
     E) The number of unsampled function calls or 
        the maximum function calls that is acceptable 
        after sampling (MAX CALLS constant)
    
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
        return [func.slopes[0] * (func.sampled - func.interval[0])]
    # [B, D] interval
    elif func.interval[1] <= func.sampled <= func.interval[2]:
        return [1.0]
    # (D, E] interval
    elif func.interval[2] < func.sampled <= func.interval[3]:
        return [1 + (func.slopes[1] * (func.sampled - func.interval[2]))]
    # [-inf, A) and (E, inf] intervals
    return [0.0]


# Collection of "basic" functions used to estimate the initial sampling in Problem A (see 'cases')
def basic_exp(level: int, base: float) -> int:
    """ Basic exponential function.

    :param level: the level of the function to compute the sampling for (see 'call_graph')
    :param base: the modification parameter obtained through evolution

    :return: the sampling threshold to use for the given level and base values
    """
    return int(base ** level)


def basic_lin(level: int, base: float) -> int:
    """ Basic linear function.

    :param level: the level of the function to compute the sampling for (see 'call_graph')
    :param base: the modification parameter obtained through evolution

    :return: the sampling threshold to use for the given level and base values
    """
    return int(base * level)


def basic_quad(level: int, base: float) -> int:
    """ Basic quadratic function.

    :param level: the level of the function to compute the sampling for (see 'call_graph')
    :param base: the modification parameter obtained through evolution

    :return: the sampling threshold to use for the given level and base values
    """
    return int(base * (level * level))


def basic_log(level: int, base: float) -> int:
    """ Basic logarithmic function.

    :param level: the level of the function to compute the sampling for (see 'call_graph')
    :param base: the modification parameter obtained through evolution

    :return: the sampling threshold to use for the given level and base values
    """
    try:
        return int(base * math.log(level))
    except ValueError:
        return 1


# OPTIMIZATION STRENGTH section
# =============================


class OptIndividual:
    def __init__(self, constraints: opt.StrConstraints, fitness: base.Fitness) -> None:
        self.constr: opt.StrConstraints = constraints
        self.str_map: Dict[str, int] = {
            opt_name: random.randint(*opt_constr) for opt_name, opt_constr in self.constr
        }
        self._idx_mapper: List[str] = [opt_name for opt_name in self.str_map.keys()]
        self.fitness: base.Fitness = fitness

    def as_dict(self) -> Dict[str, int]:
        return self.str_map
    
    def at_idx(self, idx: int) -> str:
        return self._idx_mapper[idx]

    def get_str(self) -> str:
        result = ''
        for opt, strength in self.str_map.items():
            opt_name = opt.split('_')[0]
            result += f'{opt_name}:{strength};'
        return result

    def __len__(self) -> int:
        return len(self.str_map)

    def __getitem__(self, key: str) -> int:
        return self.str_map[key]

    def __setitem__(self, key: str, value: int) -> None:
        self.str_map[key] = value
    
    # def __eq__(self, other: OptIndividual) -> bool:
    #     # Necessary when using Hall of Fame
    #     return self.constr == other.constr and self.str_map == other.str_map

    def __iter__(self) -> Iterator[Tuple[str, int]]:
        return iter(self.str_map.items())


def small_step(value: int, constraints: opt.StrInterval) -> int:
    delta = random.randint(1, SMALLSTEP_LIMIT)
    direction = -1 if random.random() < 0.5 else 1
    value += delta * direction
    if value < constraints[0]:
        value = constraints[0]
    elif value > constraints[1]:
        value = constraints[1]
    return value


def mutate_strength(individual: OptIndividual) -> None:
    mutated_param = individual.at_idx(random.randint(0, len(individual) - 1))
    opt_constr = individual.constr[mutated_param]
    if opt_constr[1] - opt_constr[0] >= STR_UNIFORM_THRESHOLD:
        individual[mutated_param] = random.randint(*opt_constr)
    else:
        individual[mutated_param] = small_step(individual[mutated_param], opt_constr)
    del individual.fitness.values


def crossover_strength(indiv1: OptIndividual, indiv2: OptIndividual) -> None:
    # The individuals must have the same strength parameters!
    if indiv1.as_dict.keys() != indiv2.as_dict.keys():
        return
    cx_point = random.randint(1, len(indiv1) - 1)
    for idx, (opt_name, _) in enumerate(indiv1):
        if idx >= cx_point:
            indiv1[opt_name], indiv2[opt_name] = indiv2[opt_name], indiv1[opt_name]
    del indiv1.fitness.values
    del indiv2.fitness.values


def fitness_str_time(result: opt.OptEffect) -> List[float]:
    return [fitness_strength(result.opt_goal, result.time_ratio)]


def fitness_str_data(result: opt.OptEffect) -> List[float]:
    return [fitness_strength(result.opt_goal, result.data_ratio)]


def fitness_str_time_data(result: opt.OptEffect) -> List[float]:
    return [
        0.5 * fitness_strength(result.opt_goal, result.time_ratio)
        + 0.5 * fitness_strength(result.opt_goal, result.data_ratio)
    ]


def fitness_strength(opt_goal: float, opt_reached: float) -> float:
    """
    Fitness:
     - For each (non-error) criterion, the fitness should look like this
    
       f |
       i |            .
       t |          //.\\
       n |         // . \\
       e |        //  .  \\
       s |  _____//   .   \\______
       s +--======+---+---+======----> 
                  A   B   C       optimization % (interval = (0.0 - 1.0))
    
                { 0.0                               iff  opt == A
                { (1 / B) * X                       iff  A < opt /\\ opt < B
      fitness = { 1.0                               iff  opt == B
                { 1 + ((X - B) * (-1 / (C - B)))    iff  B < opt /\\ opt < C
                { 0.0                               iff  opt == C
    """
    # Since we're computing with floats and we want to avoid any epsilon corrections, we close
    # the intervals from both sides, thus the 'reached' value should always hit one of the intervals
    # Interval [A, B]
    if 0.0 <= opt_reached <= opt_goal:
        return (1 / opt_goal) * opt_reached
    # Interval [B, C]
    elif opt_goal <= opt_reached <= 1.0:
        slope = -1.0 / (1.0 - opt_goal)
        return 1 + ((opt_reached - opt_goal) * slope)
    # Shouldn't really happen
    return 0.0


def apply_opt_strength(opt_effect: opt.OptEffect, strength: Dict[str, int], **_: Any):
    return opt_effect.apply(**strength)


