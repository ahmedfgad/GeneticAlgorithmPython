"""Fitness dispatch and the lifetime of workers used by a GA run."""

import concurrent.futures
from contextlib import contextmanager
from itertools import repeat
import math
import os

import cloudpickle
import numpy


def _process_fitness_chunk(payload, tasks):
    """Transport dynamic callables as bytes through the standard executor.

    Each task receives its own GA snapshot, just as it does with the
    executor's normal pickle transport. Grouping tasks amortizes transfer
    of the snapshot without allowing worker-side GA changes to leak into
    the next fitness call.
    """
    results = []
    for solution, index in tasks:
        fitness_func, ga = cloudpickle.loads(payload)
        results.append(fitness_func(ga, solution, index))
    return results


class FitnessEvaluation:
    def __getstate__(self):
        # Executors contain locks and worker handles. Neither checkpoints
        # nor GA snapshots sent to workers should contain these resources.
        state = self.__dict__.copy()
        for name in ("_fitness_executor", "_fitness_executor_config",
                     "_fitness_run_active"):
            state.pop(name, None)
        return state

    def _shutdown_fitness_executor(self):
        executor = getattr(self, "_fitness_executor", None)
        self._fitness_executor = None
        self._fitness_executor_config = None
        if executor is not None:
            executor.shutdown(wait=True)

    @contextmanager
    def _fitness_pool(self):
        config = tuple(self.parallel_processing)
        executor_class = (concurrent.futures.ProcessPoolExecutor
                          if config[0] == "process"
                          else concurrent.futures.ThreadPoolExecutor)
        if not getattr(self, "_fitness_run_active", False):
            with executor_class(max_workers=config[1]) as executor:
                yield executor
            return

        if getattr(self, "_fitness_executor_config", None) != config:
            self._shutdown_fitness_executor()
            self._fitness_executor = executor_class(max_workers=config[1])
            self._fitness_executor_config = config
        yield self._fitness_executor

    def _map_fitness(self, tasks):
        """Yield ordered results without changing the fitness API."""
        if not tasks:
            return
        if self.parallel_processing is None:
            # A callback can disable parallelism during a running GA.
            self._shutdown_fitness_executor()
            for solution, index in tasks:
                yield self.fitness_func(self, solution, index)
            return

        with self._fitness_pool() as executor:
            if self.parallel_processing[0] == "thread":
                solutions, indices = zip(*tasks)
                yield from executor.map(self.fitness_func, repeat(self),
                                        solutions, indices)
            else:
                # Serialize the current state once per evaluation round.
                # Initializing workers once with a GA would leave their
                # generations, custom attributes and callbacks stale.
                payload = cloudpickle.dumps((self.fitness_func, self))
                workers = self.parallel_processing[1] or os.cpu_count() or 1
                chunk_size = max(1, math.ceil(len(tasks) / (workers * 4)))
                chunks = [tasks[start:start + chunk_size]
                          for start in range(0, len(tasks), chunk_size)]
                for results in executor.map(_process_fitness_chunk,
                                            repeat(payload), chunks):
                    yield from results

    def _evaluate_fitness(self, population, indices, adaptive=False):
        """Evaluate the specified rows, with identical validation in all modes.

        Adaptive offspring do not yet have population indices. Their
        fitness function therefore receives None in scalar and batch mode.
        """
        if not indices:
            return []
        batch_size = self.fitness_batch_size
        batched = batch_size not in (None, 1)
        if not batched:
            groups = [[index] for index in indices]
            tasks = []
            for index in indices:
                solution = population[index]
                if self.parallel_processing is not None:
                    solution = solution.copy()
                tasks.append((solution, None if adaptive else index))
        else:
            groups = [indices[start:start + batch_size]
                      for start in range(0, len(indices), batch_size)]
            tasks = [(population[group, :], None if adaptive else group)
                     for group in groups]

        fitness_values = []
        results = self._map_fitness(tasks)
        try:
            for group, result in zip(groups, results):
                self.num_fitness_evaluations += len(group)
                if batched:
                    if type(result) not in (list, tuple, numpy.ndarray):
                        raise TypeError("Expected to receive a list, tuple, or "
                                        "numpy.ndarray from the fitness function "
                                        f"but the value ({result}) of type {type(result)}.")
                    if len(result) != len(group):
                        raise ValueError("There is a mismatch between the number "
                                         "of solutions passed to the fitness function "
                                         f"({len(group)}) and the number of fitness "
                                         f"values returned ({len(result)}). They must match.")
                    values = result
                else:
                    values = [result]
                for value in values:
                    if (type(value) not in self.supported_int_float_types
                            and type(value) not in (list, tuple, numpy.ndarray)):
                        raise ValueError("The fitness function should return a "
                                         "number or an iterable (list, tuple, or "
                                         f"numpy.ndarray) but the value {value} "
                                         f"of type {type(value)} found.")
                    fitness_values.append(value)
        finally:
            # Close an out-of-run executor even if result validation fails.
            results.close()
        return fitness_values
