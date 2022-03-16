""" Module implementing the fitness evaluation. Most importantly, the Evaluator class encapsulates
logic for single/multi process evaluation, thus switching between sequential and parallel fitness
evaluation is simple.
"""
from __future__ import annotations
import multiprocessing as mp
from threading import BrokenBarrierError
from queue import Empty
from typing import List, Dict, Any, Literal, Optional, Type

import evolution.solution as sol


InputData = Dict[str, Any]
ResultData = Any


class ProcessTermination(Exception):
    """Raised when a worker process should terminate.
    """
    def __init__(self) -> None:
        """Constructor
        """
        super().__init__("")


def split_solutions(workers: int, solutions: sol.SolutionSet) -> List[List[sol.Solution]]:
    """ Split the solutions in the SolutionSet into chunks that can be distributed among
    worker processes.

    :param workers: number of processes to split the solutions for
    :param solutions: a collection of solutions

    :return: a list of solution chunks, each chunk is to be processes by separate process
    """
    sols = list(solutions)
    return [sols[i::workers] for i in range(min(workers, len(sols)))]


class Synchronizator:
    """ Class encapsulating synchronization primitives and exporting methods for simple 
    communication among main and worker processes.

    :ivar result_queue: queue used for sending results from workers back to the main process
    :ivar input_queue: queue used for sending inputs to the worker processes
    :ivar termination: a flag indicating whether the worker processes should terminate 
    :ivar input_ready: synchronization barrier that opens when the next input is ready in the
                       input_queue. We assume that the main process sends |WORKERS| inputs through
                       the queue, i.e., that each worker should obtain exactly one input to perform
                       the computations on.
    :ivar _timeout: internal timeout used when performing operations on the queues
    """
    def __init__(self, workers: int) -> None:
        """ Constructor

        :param workers: number of worker processes
        """
        self.result_queue: mp.Queue = mp.Queue()
        self.input_queue: mp.Queue = mp.Queue()
        self.termination: mp.synchronize.Event = mp.Event()
        # The barrier must open only when all the workers are ready (i.e., can accept new inputs)
        # and the main process sent the required number of inputs
        self.input_ready: mp.synchronize.Barrier = mp.Barrier(workers + 1)
        self._timeout: float = 0.5
    
    def stop_workers(self) -> None:
        """ Indicate to the worker processes that they should terminate. The termination is 
        indicated by both:
          A) breaking the barrier, so that processes waiting before the barrier are woken up 
             and can safely exit,
          B) setting the termination flag, so that input waiting loop can be interrupted. 
        """
        self.input_ready.abort()
        self.termination.set()

    def wait_for_input(self) -> None:
        """ Wait for the barrier to open which signalizes that a new input is ready in the queue.
        """
        try:
            self.input_ready.wait()
        except BrokenBarrierError:
            # The barrier has been closed = termination
            raise ProcessTermination()
    
    def next_input(self) -> InputData:
        """ Get new input from the input queue.

        :return: new input to process
        """
        # The loop is necessary in order to allow proper termination if needed 
        while not self.termination.is_set():
            try:
                return self.input_queue.get(timeout=self._timeout)
            except Empty:
                continue
        # The termination flag is set
        raise ProcessTermination()

    def send_result(self, result: ResultData) -> None:
        """ Send result back to the main process. The queue should not ever be full (otherwise
        something is really wrong) as its maximum size is |WORKERS| and as such does not need
        a looping construct around the 'put' operation.

        :param result: a computation result that should be propagated to the main process
        """
        self.result_queue.put(result)
    
    def next_result(self) -> ResultData:
        """ Read a new result from the result_queue.

        :return: the new result
        """
        return self.result_queue.get()

    def send_input(self, input: InputData) -> None:
        """ Send new input to the input_queue.

        :param input: the new input
        """
        self.input_queue.put(input)

    def close_queues(self) -> None:
        """ Indicate that the queues are ready to be closed on this side of the communication.
        """
        for queue in [self.input_queue, self.result_queue]:
            queue.close()
            queue.join_thread()
    
    def drain_queues(self) -> None:
        """ Safely drain any remaining contents of both queues to ensure proper cleanup and not
        a deadlock. See: 'https://docs.python.org/3/library/multiprocessing.html#pipes-and-queues'
        """
        for queue in [self.input_queue, self.result_queue]:
            try:
                while True:
                    queue.get(timeout=self._timeout)
            except Empty:
                pass


class WorkerProcess(mp.Process):
    """ The worker process implementation.

    :ivar sync: the synchronization wrapper
    :ivar solutions: solutions that will be evaluated by this worker process
    :ivar apply_func: the genotype -> fenotype mapping function
    :ivar fitness_func: the fitness evaluation function
    """
    def __init__(self, sync: Synchronizator, solutions: List[sol.Solution]) -> None:
        """ Constructor

        :param sync: the synchronization object
        :param solutions: a collection of solutions to evaluate
        """
        mp.Process.__init__(self)
        self.sync: Synchronizator = sync
        self.solutions: List[sol.Solution] = solutions
        self.apply_func: Optional[sol.ApplySignature] = None
        self.fitness_func: Optional[sol.FitnessSignature] = None
    
    def run(self) -> None:
        """ The worker process computation loop. The process works as follows:
          1) Wait at the barrier until the next input is ready.
          2) Obtain the next input from the queue.
          3) Process the input and compute the sum fitness.
          4) Send the fitness value back to the main process.
        
        The worker process may be terminated properly (by signalling so using the 
        termination event) or abruptly through catching the interrupt signal.
        """
        try:
            # Repeat until some external (proper or abrupt) termination happens
            while True:
                # Get the next input from the main process
                self.sync.wait_for_input()
                data = self.sync.next_input()
                # Update the apply and fitness functions if present
                self.apply_func = data.pop('__apply', self.apply_func)
                self.fitness_func = data.pop('__fitness', self.fitness_func)
                if self.apply_func is None or self.fitness_func is None:
                    print(f"{self.name}: Missing 'apply' or 'fitness' function")
                    raise ProcessTermination()
                # Compute the new fitness value
                fitness = _sum_fitnesses([
                    solution.apply_and_eval(self.apply_func, self.fitness_func, **data) 
                    for solution in self.solutions
                ])
                # Propagate the fitness back to the main process
                self.sync.send_result(fitness)
        # Termination is signalled by the main process
        except ProcessTermination:
            # Send the serialized solutions back to update the objects in the main process
            for solution in self.solutions:
                self.sync.send_result(solution.mp_extract())
            self.sync.close_queues()
            print(f'{self.name}: Terminating')
        # Abrupt termination, send empty message back to properly synchronize the termination
        except KeyboardInterrupt:
            self.sync.send_result(None)
            self.sync.close_queues()
            print(f'{self.name}: Abruptly Terminating (CTRL+C)')
        return


class Evaluator:
    """ The evaluation context manager. This is the only class from this module that has to be 
    explicitly created in order to perform the evaluation - everything else happens under the hood
    of the Evaluator context manager.

    The Evaluator supports two modes - single process and multi process evaluation. In the multi
    process evaluation, solutions from the solution set are distributed among the specified number
    of worker processes at the start of the context manager. The solutions are then subsequently 
    provided with inputs on which to evaluate the assigned solutions. The computed results are
    then sent back and processed by the main process.

    :ivar workers_count: number of requested worker processes (0 for single process)
    :ivar solutions: the complete set of solutions to evaluate
    :ivar sync: the synchronization class used to communicate with the worker processes
    :ivar workers: the actual worker processes
    :ivar apply_func: the genotype -> fenotype mapping function
    :ivar fitness_func: the fitness evaluation function
    :ivar _funcs_sent: flag indicating whether new apply and fitness functions should be sent to
                       the worker processes
    """
    def __init__(
        self, workers: int, solutions: sol.SolutionSet, 
        apply_func: sol.ApplySignature, fitness_func: sol.FitnessSignature
    ) -> None:
        """ Constructor

        :param workers: no. of worker processes
        :param solutions: a collection of solutions
        :param apply_func: the genotype -> fenotype mapping function
        :param fitness_func: the fitness evaluation function
        """
        self.workers_count: int = workers  # 0 for single process
        self.solutions: sol.SolutionSet = solutions
        self.sync: Optional[Synchronizator] = None
        self.workers: Optional[List[WorkerProcess]] = None
        self.apply_func: sol.ApplySignature = apply_func
        self.fitness_func: sol.FitnessSignature = fitness_func
        self._funcs_sent: bool = False
    
    def __enter__(self) -> Evaluator:
        """ The context manager entry sentinel.

        Does nothing for single process evaluation. 
        Starts and initializes the worker processes if multi process evaluation is selected. 
        """
        if self.workers_count > 0:
            # Distribute the solutions into per-process chunks
            sol_chunks = split_solutions(self.workers_count, self.solutions)
            # Handle the case where there is more processes than inputs
            if len(sol_chunks) < self.workers_count:
                print(
                    f"INFO: Less workloads ({len(self.solutions)}) than required workers"\
                    f"({self.workers_count}), reducing the number of workers."
                )
                self.workers_count = len(sol_chunks)
            # Create the Synchronizator object
            self.sync = Synchronizator(self.workers_count)
            # Create and start the worker processes
            self.workers = [WorkerProcess(self.sync, sol_chunks[i]) for i in range(len(sol_chunks))]
            for w in self.workers:
                w.start()
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], _: Any, __: Any) -> Literal[False]:
        """ The context manager exit sentinel.

        Does nothing for single process evaluation. Otherwise:
        1) Synchronizes the solution set in the main process according to the results 
           in the worker processes.
        2) Properly terminates the worker processes.
        3) Performs cleanup of the resources

        :param exc_type: type of the encountered exception, if any

        :return: indication of whether the potential exceptions should be propagated or not
        """
        print('Main exitting')
        if self.workers is None or self.sync is None:
            return False

        # Clean termination, collect the results
        if exc_type is None:
            self.sync.stop_workers()
            print('Main waiting for solution update')
            for _ in range(len(self.solutions)):
                self.solutions.mp_update(self.sync.next_result())
        # Abrupt termination, no results are expected (worker Solution objects are likely corrupted)
        elif isinstance(exc_type, KeyboardInterrupt):
            # Terminated workers return synchronization 'None' message
            for _ in range(self.workers_count):
                self.sync.next_result()
        # Drain queues to avoid potential deadlock
        print('Main draining queues')
        self.sync.drain_queues()
        # Wait for the workers to finish
        print('Main joining workers')
        for w in self.workers:
            w.join()
        # Close the communication queues
        print('Main closing queues')
        self.sync.close_queues()
        # Propagate exceptions
        return False

    def evaluate(
        self, apply_func: Optional[sol.ApplySignature], 
        fitness_func: Optional[sol.FitnessSignature], **apply_params: Any
    ) -> List[float]:
        """ Runs the evaluation in either a single or multi process fashion.

        The multi process approach requires some additional communication and coordination across
        main and worker processes.

        :param apply_func: the genotype -> fenotype mapping function
        :param fitness_func: the fitness evaluation function
        :param apply_params: parameters for the apply function

        :return: the computed fitness value
        """
        # Check if the apply or fitness functions have changed
        self._update_funcs(apply_func, fitness_func)
        # Single process section
        if self.sync is None:
            return _avg_fitness(_sum_fitnesses([
                solution.apply_and_eval(self.apply_func, self.fitness_func, **apply_params) 
                for solution in self.solutions
            ]), len(self.solutions))
        # Multiprocess section
        input_data = dict(apply_params)
        # Serialize also the functions if needed
        if not self._funcs_sent:
            input_data['__apply'] = self.apply_func
            input_data['__fitness'] = self.fitness_func
            self._funcs_sent = True
        # Send the inputs to the queue
        for _ in range(self.workers_count):
            self.sync.send_input(input_data)

        # Wait as the last process required to breach the barrier
        self.sync.wait_for_input()
        # Collect the results
        return _avg_fitness(_sum_fitnesses(
            [self.sync.next_result() for _ in range(self.workers_count)]
        ), len(self.solutions))

    def _update_funcs(
        self, apply_func: Optional[sol.ApplySignature], fitness_func: Optional[sol.FitnessSignature]
    ) -> None:
        """ Check if the used apply and fitness functions have changed (i.e., new ones are supplied
        to the evaluation function) and if so, update the internal references.

        :param apply_func: the genotype -> fenotype mapping function
        :param fitness_func: the fitness evaluation function
        """
        if apply_func is not None:
            self.apply_func = apply_func
            self._funcs_sent = False
        if fitness_func is not None:
            self.fitness_func = fitness_func
            self._funcs_sent = False


def _sum_fitnesses(results: List[List[float]]) -> List[float]:
    # All of the lists in results are expected to have the same length!
    if not results:
        return []
    res = results[0]
    for lst in results[1:]:
        for idx in range(len(res)):
            res[idx] += lst[idx]
    return res

def _avg_fitness(fitness: List[float], solutions: int) -> List[float]:
    for idx, fit in enumerate(fitness):
        fitness[idx] = fit / solutions
    return fitness
