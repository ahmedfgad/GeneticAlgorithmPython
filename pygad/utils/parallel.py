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

    Parameters
    ----------
    payload : bytes
        Cloudpickle-serialized pair of the fitness callable and current GA.
    tasks : list of tuple
        Pairs of solution data and its fitness index argument. Solution
        data can be one chromosome or a batch; adaptive indices are None.

    Returns
    -------
    results : list
        Fitness results in task order. Deserialization and fitness-call
        exceptions propagate to the parent through the process executor.
    """
    results = []
    for solution, index in tasks:
        fitness_func, ga = cloudpickle.loads(payload)
        results.append(fitness_func(ga, solution, index))
    return results


class FitnessEvaluation:
    """Share fitness evaluation and worker cleanup across GA operations.

    GAEngine inherits this mixin so ordinary population evaluation and
    adaptive offspring evaluation use the same dispatch and validation.
    Users configure these operations through the GA constructor; the
    methods beginning with an underscore are internal helpers.
    """

    def __getstate__(self):
        """Return instance state without live worker resources.

        Returns
        -------
        state : dict
            A shallow copy of instance attributes excluding the executor,
            its configuration, the active-run flag, and saved fitness indexes. Cloudpickle uses
            it for checkpoints and GA snapshots sent to process workers.
        """
        # Executors contain locks and worker handles. Neither checkpoints
        # nor GA snapshots sent to workers should contain these resources.
        state = self.__dict__.copy()
        for name in ("_fitness_executor", "_fitness_executor_config",
                     "_fitness_run_active", "_saved_fitness_indexes"):
            state.pop(name, None)
        return state

    def _shutdown_fitness_executor(self):
        """Detach the run's executor and wait for submitted work to finish.

        Clears the stored executor and configuration even if no pool
        exists. Returns None; it is safe to call again after shutdown.
        """
        executor = getattr(self, "_fitness_executor", None)
        self._fitness_executor = None
        self._fitness_executor_config = None
        if executor is not None:
            executor.shutdown(wait=True)

    @contextmanager
    def _fitness_pool(self):
        """Provide the executor selected by normalized parallel_processing.

        Yields
        ------
        executor : concurrent.futures.Executor
            A temporary thread or process pool outside run(), or the pool
            reused during an active run. A configuration change closes the
            stored pool before creating its replacement. Temporary pools
            close on context exit; run() closes its own pool on exit.

        Notes
        -----
        This context manager expects parallel processing to be enabled.
        A worker count of None uses the selected executor's default.
        """
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
        """Dispatch fitness calls and yield their results in task order.

        Parameters
        ----------
        tasks : list of tuple
            Pairs of solution data and its fitness index argument, prepared
            by _evaluate_fitness(). An empty list creates no worker pool.

        Yields
        ------
        fitness : numeric or array-like
            One result per task, including a batch of fitness values when
            the task contains multiple solutions. Validation and counting
            are handled by _evaluate_fitness(), not by this dispatcher.

        Notes
        -----
        Threads share this GA instance. Process tasks receive separate GA
        snapshots refreshed each evaluation round. Grouping process tasks
        reduces state transfers without changing the fitness signature.
        Fitness and serialization exceptions propagate to the caller.
        """
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

        Parameters
        ----------
        population : numpy.ndarray
            Two-dimensional array containing the chromosomes to evaluate.
        indices : list of int
            Rows to evaluate, in result order. An empty list returns [].
        adaptive : bool, default False
            Pass None as the fitness index argument instead of population
            row indices when evaluating offspring for adaptive mutation.

        Returns
        -------
        fitness_values : list
            One scalar or objective vector per requested row. A batch can
            be smaller than fitness_batch_size after cached rows are skipped.

        Raises
        ------
        TypeError
            If a batch call returns neither list, tuple, nor numpy.ndarray.
        ValueError
            If a batch's result length differs from its solution count,
            or a fitness value has an unsupported type, shape, objective
            count, or non-finite objective value.

        Notes
        -----
        Counts solutions in each returned result before validation, rather
        than counting fitness-function calls. Exceptions propagate, and
        the result generator is closed to clean up temporary pools.
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
                    if isinstance(result, numpy.ndarray) and result.ndim == 0:
                        raise ValueError("A batched fitness_func must return one fitness value per solution, not a scalar array.")
                    if len(result) != len(group):
                        raise ValueError("There is a mismatch between the number "
                                         "of solutions passed to the fitness function "
                                         f"({len(group)}) and the number of fitness "
                                         f"values returned ({len(result)}). They must match.")
                    values = result
                else:
                    values = [result]
                for index, value in zip(group, values):
                    fitness_values.append(self._validate_fitness_value(
                        value, f"fitness_func for solution {index}"))
        finally:
            # Close an out-of-run executor even if result validation fails.
            results.close()
        return fitness_values

    def _validate_fitness_value(self, value, source, check_shape=True):
        """Validate one scalar or non-empty objective vector before selection.

        Scalars may include infinity to represent a perfect or rejected
        solution. Objective vectors must be finite because Pareto distance
        and normalization calculations require finite objective ranges.
        Every solution must return the same number of objectives.
        """
        if isinstance(value, numpy.ndarray) and value.ndim == 0:
            value = value.item()
        if type(value) in self.supported_int_float_types and type(value) is not object:
            shape = ()
            values = [value]
        elif isinstance(value, (list, tuple, numpy.ndarray)):
            try:
                array = numpy.asarray(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{source} must return a non-empty one-dimensional numeric objective vector.") from error
            if array.ndim != 1 or array.size == 0:
                raise ValueError(f"{source} must return a number or a non-empty one-dimensional objective vector.")
            shape = array.shape
            values = value
        else:
            raise ValueError(f"{source} must return a number or a non-empty one-dimensional objective vector, but received {type(value)}.")
        for objective in values:
            if type(objective) not in self.supported_int_float_types or type(objective) is object:
                raise ValueError(f"{source} contains a non-numeric fitness value: {objective!r}.")
            if isinstance(objective, (float, numpy.floating)):
                if numpy.isnan(objective) or (shape and not numpy.isfinite(objective)):
                    raise ValueError(f"{source} contains an invalid fitness value. NaN is not supported, and objective vectors must contain finite numbers.")
        if check_shape:
            expected_shape = getattr(self, '_fitness_value_shape', None)
            if expected_shape is not None and shape != expected_shape:
                raise ValueError(f"{source} has fitness shape {shape}, but all solutions must use the same fitness shape {expected_shape}.")
            self._fitness_value_shape = shape
        return value

    def _validate_population_fitness(self, fitness, source, check_shape=True):
        """Return validated population fitness without changing its shape."""
        values = [self._validate_fitness_value(value, source, check_shape=check_shape)
                  for value in fitness]
        shapes = [numpy.shape(value) for value in values]
        if shapes and any(shape != shapes[0] for shape in shapes):
            raise ValueError(f"{source} must use the same fitness shape for all solutions.")
        return numpy.asarray(values)
