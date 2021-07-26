""" Module containing implementation of evolutionary computation algorithms, such as:
      - basic Evolutionary Algorithm
      - Genetic Programming
      - Evolutionary Strategies

    for solving the cases (see 'cases' module).
"""
import random
import math
from typing import Tuple, Union, Dict, Any

from deap import base, creator, tools

import solution as sol
import evaluator


# Some default values used in the algorithms
# Population size, No. Generations, Mutation prob., Crossover prob.
NPOP, NGEN, PMUT, PCX = 20, 100, 0.5, 0.2


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


class EvolutionParameters:
    """ Class containing parameters of the evolution process. The class generally assumes only the
    most generic parameters employed in all evolutionary techniques:
      - Population size
      - Number of generations
      - Probability of mutation
      - Probability of crossover

    Additional parameters can be supplied as keyword arguments and will be made available in 
    the object through the . (dot) notation, i.e.:
      - EvolutionParameters(..., my_parameter=13.5) -> evo_param.my_parameter

    The general rules are that 1) each case should provide parameter values that are different
    from the default ones and 2) each algorithm should contain an initialization phase where all
    required parameters are set to their default value if no parameter value was supplied.

    :ivar pop: the requested population size
    :ivar gen: the number of generations
    :ivar pmut: the probability of mutation
    :ivar pcx: the probability of crossover
    """
    def __init__(
        self, popsize: int = NPOP, generations: int = NGEN, prob_mut: float = PMUT, 
        prob_cx: float = PCX, **kwargs: Any
    ) -> None:
        """ Constructor

        :param popsize: the requested population size
        :param generations: the number of generations
        :param prob_mut: the probability of mutation
        :param prob_cx: the probability of crossover
        """
        self.pop = popsize
        self.gen = generations
        self.pmut = prob_mut
        self.pcx = prob_cx
        # Set the additional kwargs parameters as attributes
        for attr, value in kwargs.items():
            setattr(self, attr, value)

    def update_defaults(self, defaults: Dict[str, Any]) -> None:
        """ Set default parameter values for those parameters that are not present in the object. 
        We do not impose any restriction on the type of the parameters.

        :param defaults: map of default parameters and their values
        """
        for attr, value in defaults.items():
            if not hasattr(self, attr):
                setattr(self, attr, value)

    def __setattr__(self, name: str, value: Any) -> None:
        """ Setattr override to allow for dynamic attributes type checking.
        https://mypy.readthedocs.io/en/latest/cheat_sheet_py3.html#when-you-re-puzzled-or-when-things-are-complicated

        :param name: name of the attribute
        :param value: value of the attribute
        """
        super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        """ Getattr override to allow for dynamic attributes type checking.
        https://mypy.readthedocs.io/en/latest/cheat_sheet_py3.html#when-you-re-puzzled-or-when-things-are-complicated

        :param name: name of the attribute
        
        :return: value of the attribute
        """
        return super().__getattribute__(name)


def basic_ea(
        solutions: sol.SolutionSet,
        evo_params: EvolutionParameters,
        apply_func: sol.ApplySignature, 
        fitness_func: sol.FitnessSignature,
        workers: int,
        **apply_params: Union[int, float]
) -> Tuple[float, float]:
    """ Evolutionary algorithm for solving the 'basic' variants of the initial sampling function
    where we tune only the 'base' parameter. 
    Heavily inspired by 'https://deap.readthedocs.io/en/master/overview.html'.

    :param solutions: a collection of solutions, one for each workload being solved
    :param evo_params: the supplied parameters for the evolution process
    :param apply_func: function to use for genotype -> fenotype mapping
    :param fitness_func: function to use for fitness evaluation
    :param workers: number of worker processes
    :param apply_params: additional parameters for the apply function

    :return: the best individual and its fitness value
    """

    # First make sure that we have all the parameters we need
    evo_params.update_defaults({
            'attr_low': 0.0, 'attr_high': 100.0,
            'cx_eta': 2.0,
            'mut_eta': 2.0, 'mut_mu': 1, 'mut_sigma': 5,
            'tourn_size': 3
        })

    # Create maximization fitness and an individual class
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)

    try:
        # Create an evaluator context that handles the multiprocess / single process evaluation
        with evaluator.Evaluator(workers, solutions, apply_func, fitness_func) as ev:
            # Set the individual and population initializers
            toolbox = base.Toolbox()
            toolbox.register(
                'attribute', 
                random.uniform, evo_params.attr_low, evo_params.attr_high
            )
            toolbox.register(
                'individual', 
                tools.initRepeat, creator.Individual, toolbox.attribute, n=1
            )
            toolbox.register(
                'population', 
                tools.initRepeat, list, toolbox.individual
            )
            # Set the evolution operators: crossover, mutation, selection
            # The evaluation will be performed by the evaluator
            toolbox.register(
                'mate', 
                tools.cxSimulatedBinary, eta=evo_params.cx_eta
            )
            toolbox.register(
                'mutate', 
                tools.mutGaussian, mu=evo_params.mut_mu, sigma=evo_params.mut_sigma, 
                indpb=evo_params.pmut
            )
            toolbox.register(
                'select', 
                tools.selTournament, tournsize=evo_params.tourn_size, k=evo_params.pop - 1
            )
            # Store the all-time best individual
            hof = tools.HallOfFame(1)

            # Evaluate fitness of the initial random population
            pop = toolbox.population(evo_params.pop)
            for ind in pop:
                ind.fitness.values = ev.evaluate(None, None, base=ind[0], **apply_params), 
            hof.update(pop)

            # Run all the generations
            for g in range(evo_params.gen):
                print(f'Generation: {g}')
                # Create new offsprings, always include the all-time best solution
                # The cloning is necessary since crossover and mutations work in-situ
                offsprings = list(map(toolbox.clone, toolbox.select(pop))) + [toolbox.clone(hof[0])]

                # Perform the crossover (mating) among the offsprings
                for o1, o2 in zip(offsprings[::2], offsprings[1::2]):
                    if random.random() < evo_params.pcx:
                        toolbox.mate(o1, o2)
                        del o1.fitness.values
                        del o2.fitness.values
                # Additionally mutate some of the new offsprings
                for mutant in offsprings:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
                
                # Recalculate the fitness for the modified offsprings (mating, mutation)
                for ind in [o for o in offsprings if not o.fitness.valid]:
                    ind.fitness.values = ev.evaluate(None, None, base=ind[0], **apply_params), 
                
                # Update the population and all-time best
                pop[:] = offsprings
                hof.update(pop)

            # Re-evaluate the solutions to contain the all-time best results
            ev.evaluate(None, None, base=hof[0][0], **apply_params)
            return hof[0][0], hof[0].fitness.values[0]
    finally:
        # Make sure we remove the created Individual and Fitness classes when multiple algorithms
        # are run back to back in one session
        del creator.Individual
        del creator.FitnessMax
