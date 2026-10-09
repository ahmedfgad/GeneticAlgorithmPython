# Fitness Calculation and Performance

This page covers how PyGAD calculates the fitness efficiently: parallel processing, non-deterministic problems, reusing fitness values, and batch fitness calculation.

## Fitness Output Validation

For a single-objective problem, `fitness_func` returns one numeric value per solution. For a multi-objective problem, it returns a non-empty, one-dimensional list, tuple, or NumPy array of numeric objective values. Every solution must return the same number of objectives throughout a run, including cached solutions and offspring evaluated for adaptive mutation. Empty vectors, nested vectors, non-numeric values, and inconsistent objective counts raise a descriptive error before parent selection.

`NaN` is rejected. Single-objective fitness may be positive or negative infinity, for example to represent a perfect or rejected solution; proportional selection and numerical plots can still require finite scores. Multi-objective fitness must contain finite values because crowding distances and reference-point normalization use differences between objective values.

Batch evaluation returns one such fitness value per supplied solution, including a smaller final batch. Sequential, threaded, and process evaluation use the same validation. `on_fitness` outputs are validated too, whether the callback returns replacement values or edits the supplied array in place.

(saved-fitness-across-repeated-runs)=
## Saved Fitness across Repeated Runs

Calling `run()` again continues from `generations_completed` and extends the existing histories. Each run saves its starting population and final population. For two runs of 2 generations, `best_solutions_generations` contains `[0, 1, 2, 2, 3, 4]`. Both snapshots of generation 2 remain available. The corresponding `best_solutions_fitness` entries have the same positions, and `best_solutions` uses those positions when `save_best_solutions=True`.

When `save_solutions=True`, `solutions_generations` contains one generation number per saved population. `solutions` and `solutions_fitness` retain their existing flat layout, with one entry per solution. Population boundaries are recorded internally, including populations enlarged by NSGA-III. `best_solution_generation` reports the actual generation of the best saved fitness rather than its position in the history. For multi-objective histories, it uses the same NSGA-II ordering as `best_solution()`.

`on_fitness(ga_instance, population_fitness)` runs before parent selection for each generation. After it returns, PyGAD recomputes the best solution so the saved solution and fitness agree. The final population is saved without an additional `on_fitness` call. Previously saved arrays are independent of later callback edits. Callbacks receive the population fitness after cache reuse, so changes to already cached scores can accumulate if the callback repeatedly adds to them.

Checkpoints preserve the generation numbers and population boundaries. Older checkpoints containing a single-run history recover the generation numbers automatically. Older repeated-run checkpoints did not record run boundaries, so unavailable generation numbers are represented by `None`; `best_solution_generation` is `-1` if the winning snapshot has an unknown generation. New snapshots have their actual generation numbers. Plots use snapshot positions only for those unknown legacy entries. A complete example is available at [`examples/example_repeated_runs.py`](https://github.com/ahmedfgad/GeneticAlgorithmPython/tree/master/examples/example_repeated_runs.py).

(parallel-processing-guide)=
## Parallel Processing in PyGAD

Starting from [PyGAD 2.17.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-17-0), parallel processing is supported. This section explains how to use parallel processing in PyGAD.

According to the [PyGAD life cycle](https://pygad.readthedocs.io/en/latest/lifecycle.html#life-cycle-of-pygad), the computation can be parallelized in only 2 operations:

1. Population fitness calculation.
2. Mutation.

The reason is that the calculations in these 2 operations are independent (i.e. each solution/chromosome is handled independently from the others) and can be distributed across different processes or threads.

For the mutation operation, it does not do intensive calculations on the CPU. Its calculations are simple like flipping the values of some genes from 0 to 1 or adding a random value to some genes. So, it does not take much CPU processing time. Experiments proved that parallelizing the mutation operation across the solutions increases the time instead of reducing it. This is because running multiple processes or threads adds overhead to manage them. Thus, parallel processing cannot be applied on the mutation operation.

For the population fitness calculation, parallel processing can make a difference and reduce the processing time. But this depends on the type of calculations done in the fitness function. If the fitness function makes intensive calculations and takes much CPU time, then parallel processing will probably help cut down the overall time.

This section explains how parallel processing works in PyGAD and how to use it.

### How to Use Parallel Processing in PyGAD

Starting from [PyGAD 2.17.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-17-0), a new parameter called `parallel_processing` was added to the constructor of the `pygad.GA` class.

```python
import pygad
...
ga_instance = pygad.GA(...,
                       parallel_processing=...)
...
```

This parameter allows the user to do the following:

1. Enable parallel processing.
2. Select whether processes or threads are used.
3. Specify the number of processes or threads to be used.

These are 3 possible values for the `parallel_processing` parameter:

1) `None`: (Default) It means no parallel processing is used.
2) A positive integer referring to the number of threads to be used (threads, not processes).
3) `list`/`tuple`: If a list or a tuple of exactly 2 elements is assigned, then:
   1) The first element can be either `'process'` or `'thread'` to specify whether processes or threads are used, respectively.
   2) The second element can be:
      1) A positive integer to select the maximum number of processes or threads to be used
      2) `0` to indicate that 0 processes or threads are used. It means no parallel processing. This is identical to setting `parallel_processing=None`.
      3) `None` to use the default value as calculated by the `concurrent.futures` module.

These are examples of the values assigned to the `parallel_processing` parameter:

* `parallel_processing=4`: Because the parameter is assigned a positive integer, this means parallel processing is activated where 4 threads are used.
* `parallel_processing=["thread", 5]`: Use parallel processing with 5 threads. This is identical to `parallel_processing=5`.
* `parallel_processing=["process", 8]`: Use parallel processing with 8 processes.
* `parallel_processing=["process", 0]`: As the second element is given the value 0, this means do not use parallel processing. This is identical to `parallel_processing=None`.

### Choosing Serial, Threads, Processes, or Batches

Start with serial evaluation and measure a complete `run()` using the workload you actually need. A vectorized fitness function with `fitness_batch_size` often helps cheap NumPy calculations more than adding workers.

- **Threads** are useful for I/O waits and native calculations that release Python's GIL. Fitness functions share the GA and other Python objects, so changes to shared state must be synchronized.
- **Processes** are useful for expensive Python CPU work. Each task receives a snapshot of the GA; worker changes do not update the parent instance. Transferring chromosomes, saved histories, and custom attributes costs time and memory, so small fitness calculations can be slower than serial evaluation.
- **Batch fitness** reduces the number of function calls and can use vectorized operations. The last batch can be smaller than `fitness_batch_size`, particularly when cached solutions are skipped. See [Batch Fitness Calculation](#batch-fitness-calculation).

PyGAD uses `concurrent.futures` and reuses one executor during each `run()` call. Ordinary population evaluation and adaptive mutation's offspring evaluation share it. Workers are shut down after completion, early stopping, or an exception. Calling `cal_pop_fitness()` outside a run uses a temporary executor. A population whose fitness is completely cached creates no executor.

Process tasks transport their fitness function and GA state with cloudpickle. This supports local functions, callable instances, bound methods, and loading a saved GA before continuing with process workers. The state is refreshed for every evaluation round, so changes to custom GA attributes or the fitness function in a callback reach the next round. Each process task has its own GA snapshot. Live executors are excluded from checkpoints. External resources attached to the GA must still be serializable; cloudpickle cannot make every file handle, lock, or framework session transferable.

Process tasks are grouped internally to reduce repeated state transfers. This scheduling does not change the scalar fitness signature. `fitness_batch_size`, in contrast, explicitly changes that signature to batches. During adaptive mutation, offspring have no current-population row yet: both scalar and batch fitness calls receive `None` for their index argument in every execution mode. The passed offspring values must be used to evaluate them.

Serial, thread, and process modes use the same cached-fitness rules. `num_fitness_evaluations` counts evaluated solutions, including adaptive offspring; a batch contributes its number of solutions, and cache hits contribute zero. The `evaluations_<N>` stop criterion checks this count at generation boundaries.

Use a main guard when starting a process-based GA, especially on Windows and macOS:

```python
import numpy
import pygad


def fitness_func(ga_instance, solution, solution_idx):
    # Replace this cheap example with your actual expensive CPU workload.
    return -float(numpy.sum(solution * solution))


if __name__ == "__main__":
    ga_instance = pygad.GA(num_generations=20,
                           sol_per_pop=40,
                           num_parents_mating=10,
                           num_genes=10,
                           fitness_func=fitness_func,
                           parallel_processing=["process", 4])
    ga_instance.run()
```

Compare this with `parallel_processing=None` and `["thread", 4]`, keeping the population, seed, stopping criteria, and fitness workload the same. Worker startup is included in the total time of a run. Repeated calls to `run()` create separate pools, so short runs still pay startup costs. If fitness already uses BLAS, OpenMP, TensorFlow, or another parallel library, avoid giving every GA worker another full set of CPU threads.

The repository's `examples/benchmarks/parallel_processing.py` measures complete runs with CPU, simulated I/O, and cheap NumPy workloads. For example, run `python examples/benchmarks/parallel_processing.py --workload cpu`. It reports three timing samples for serial, thread, and process modes; the NumPy workload also includes vectorized batch evaluation. Results are checked against the serial final population.

For Keras, calls to `pygad.kerasga.predict()` sharing one model are synchronized; they preserve each solution's weights but run one at a time. Separate models are needed for concurrent predictions. Direct changes to shared models outside that helper require their own synchronization.

(non-deterministic-fitness)=
## Solve Non-Deterministic Problems

PyGAD can be used to solve both deterministic and non-deterministic problems. Deterministic problems are those that return the same fitness for the same solution. For non-deterministic problems, a different fitness value may be returned for the same solution.

By default, PyGAD settings are set to solve deterministic problems. PyGAD can save the explored solutions and their fitness to reuse them in the future. These instance attributes can save the solutions:

1. `solutions`: Exists if `save_solutions=True`.
2. `best_solutions`: Exists if `save_best_solutions=True`.
3. `last_generation_elitism`: Exists if `keep_elitism` > 0.
4. `last_generation_parents`: Exists if `keep_parents` > 0 or `keep_parents=-1`.

To configure PyGAD for non-deterministic problems, we have to disable saving the previous solutions. This is by setting these parameters:

1. `keep_elitism=0`
2. `keep_parents=0`
3. `save_solutions=False`
4. `save_best_solutions=False`

```python
import pygad
...
ga_instance = pygad.GA(...,
                       keep_elitism=0,
                       keep_parents=0,
                       save_solutions=False,
                       save_best_solutions=False,
                       ...)
```

This way, PyGAD will not save any explored solution, so the fitness function has to be called for each individual solution.

## Reuse the Fitness instead of Calling the Fitness Function

Saved solutions are indexed by their complete gene values to avoid scanning the entire history for every population member. Built-in evolution indexes new snapshots incrementally. Cache precedence remains saved solutions, saved best solutions, retained elites, then retained parents, using the first matching entry in each source. Duplicate solutions that have not been evaluated or saved are still evaluated independently.

Indexes are rebuilt for direct evaluations outside `run()`, at the start of each run, and after user operators or callbacks that may edit the public histories. The indexes are omitted from checkpoints and process-worker snapshots and rebuilt when needed. This preserves history edits and cache behavior without adding configuration parameters.

It may happen that a previously explored solution in generation X is explored again in another generation Y (where Y > X). For some problems, calling the fitness function takes much time. 

For deterministic problems, it is better not to call the fitness function for an already explored solution. Instead, reuse the fitness of the old solution. PyGAD supports some options to help you save the time of calling the fitness function for a previously explored solution.

The parameters explored in this section can be set in the constructor of the `pygad.GA` class.

The `cal_pop_fitness()` method of the `pygad.GA` class checks these parameters to see if there is a possibility of reusing the fitness instead of calling the fitness function.

### 1. `save_solutions`

It defaults to `False`. If set to `True`, then the population of each generation is saved into the `solutions` attribute of the `pygad.GA` instance. In other words, every single solution is saved in the `solutions` attribute.

### 2. `save_best_solutions`

It defaults to `False`. If `True`, then it only saves the best solution in every generation. 

### 3. `keep_elitism`

It accepts an integer and defaults to 1. If set to a positive integer, then it keeps the elitism of one generation available in the next generation. 

### 4. `keep_parents`

It accepts an integer and defaults to -1. If set to `-1` or a positive integer, then it keeps the parents of one generation available in the next generation.

## Why the Fitness Function is not Called for Solution at Index 0?

PyGAD has a parameter called `keep_elitism` which defaults to 1. This parameter defines the number of best solutions in generation **X** to keep in the next generation **X+1**. The best solutions are just copied from generation **X** to generation **X+1** without making any change.

```python
ga_instance = pygad.GA(...,
                       keep_elitism=1,
                       ...)
```

The best solutions are copied at the beginning of the population. If `keep_elitism=1`, this means the best solution in generation X is kept in the next generation X+1 at index 0 of the population. If `keep_elitism=2`, this means the 2 best solutions in generation X are kept in the next generation X+1 at indices 0 and 1 of the population.

Because the fitness values of these best solutions are already calculated in generation X, they are not recalculated at generation X+1 (the fitness function is not called for these solutions again). Instead, their fitness values are reused. This is why no solution with index 0 is passed to the fitness function.

To force calling the fitness function for each solution in every generation, consider setting `keep_elitism` and `keep_parents` to 0. Moreover, keep the 2 parameters `save_solutions` and `save_best_solutions` to their default value `False`.

```python
ga_instance = pygad.GA(...,
                       keep_elitism=0,
                       keep_parents=0,
                       save_solutions=False,
                       save_best_solutions=False,
                       ...)
```

(batch-fitness-calculation)=
## Batch Fitness Calculation

In [PyGAD 2.19.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-19-0), a new optional parameter called `fitness_batch_size` is supported to calculate the fitness function in batches. Thanks to [Linan Qiu](https://github.com/linanqiu) for opening the [GitHub issue #136](https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/136).

Its values can be:

* `1` or `None`: If the `fitness_batch_size` parameter is assigned the value `1` or `None` (default), then the normal flow is used where the fitness function is called for each individual solution. That is if there are 15 solutions, then the fitness function is called 15 times.
* `1 < fitness_batch_size <= sol_per_pop`: If the `fitness_batch_size` parameter is assigned a value satisfying this condition `1 < fitness_batch_size <= sol_per_pop`, then the solutions are grouped into batches of size `fitness_batch_size` and the fitness function is called once for each batch. In this case, the fitness function must return a list/tuple/numpy.ndarray with a length equal to the number of solutions passed.

(short-fitness-batches)=
### Why a Fitness Batch Can Be Smaller

`fitness_batch_size` is the maximum number of solutions passed in one call. The final batch is smaller when the number of solutions needing evaluation is not a multiple of that size. Cached parents, elites, and previously saved solutions can also reduce the number of rows to evaluate.

For example, a population of 100 solutions with `fitness_batch_size=10` initially has ten batches of ten. If one elite's fitness is reused in the next generation and all other rows need evaluation, the remaining 99 solutions form nine batches of ten and a final batch of nine. This is expected behavior.

Use the actual input shape in the fitness function, and return exactly one fitness value for every supplied solution:

```python
def fitness_func(ga_instance, solutions, solutions_indices):
    # Works for a full batch and for a shorter final batch.
    return numpy.sum(solutions, axis=1)
```

Avoid allocating results or reshaping model inputs with a fixed `fitness_batch_size`. Use `len(solutions)` or `solutions.shape[0]` instead. With multiple objectives, return one objective vector per supplied solution. The ordinary index argument lists the evaluated population rows, which can be non-contiguous when cached rows are skipped. Adaptive offspring evaluation receives `None` for the index argument.

The following example prints `10`, `10`, `10`, and `9` as its batch sizes. The initial population requires 20 evaluations, and the next population reuses one elite's fitness:

```python
import numpy
import pygad


def fitness_func(ga_instance, solutions, solutions_indices):
    print(f"Batch size: {len(solutions)}")
    return numpy.sum(solutions, axis=1)


initial_population = numpy.arange(40, dtype=float).reshape(20, 2)
ga_instance = pygad.GA(num_generations=1,
                       num_parents_mating=5,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       fitness_batch_size=10,
                       keep_elitism=1,
                       mutation_num_genes=1,
                       random_seed=17)
ga_instance.run()
```

The runnable script is [`examples/example_fitness_batch_size.py`](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/master/examples/example_fitness_batch_size.py). The same variable-batch-size contract applies to serial, thread, and process evaluation.

### Example without `fitness_batch_size` Parameter

This is an example where the `fitness_batch_size` parameter is given the value `None` (which is the default value). This is equivalent to using the value `1`. In this case, the fitness function will be called for each solution. This means the fitness function `fitness_func` will receive only a single solution. This is an example of the passed arguments to the fitness function:

```
solution: [ 2.52860734, -0.94178795, 2.97545704, 0.84131987, -3.78447118, 2.41008358]
solution_idx: 3
```

The fitness function also must return a single numeric value as the fitness for the passed solution.

As we have a population of `20` solutions, then the fitness function is called 20 times per generation. For 5 generations, then the fitness function is called `20*5 = 100` times. In PyGAD, the fitness function is called after the last generation too and this adds additional 20 times. So, the total number of calls to the fitness function is `20*5 + 20 = 120`.

Note that the `keep_elitism` and `keep_parents` parameters are set to `0` to make sure no fitness values are reused and to force calling the fitness function for each individual solution.

```python
import pygad
import numpy

function_inputs = [4,-2,3.5,5,-11,-4.7]
desired_output = 44

number_of_calls = 0

def fitness_func(ga_instance, solution, solution_idx):
    global number_of_calls
    number_of_calls = number_of_calls + 1
    output = numpy.sum(solution*function_inputs)
    fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
    return fitness

ga_instance = pygad.GA(num_generations=5,
                       num_parents_mating=10,
                       sol_per_pop=20,
                       fitness_func=fitness_func,
                       fitness_batch_size=None,
                       # fitness_batch_size=1,
                       num_genes=len(function_inputs),
                       keep_elitism=0,
                       keep_parents=0)

ga_instance.run()
print(number_of_calls)
```

```
120
```

### Example with `fitness_batch_size` Parameter

This is an example where the `fitness_batch_size` parameter is used and assigned the value `4`. This means the solutions will be grouped into batches of `4` solutions. The fitness function will be called once for each batch (called once for every 4 solutions).

This is an example of the arguments passed to it:

```python
solutions:
    [[ 3.1129432  -0.69123589  1.93792414  2.23772968 -1.54616001 -0.53930799]
     [ 3.38508121  0.19890812  1.93792414  2.23095014 -3.08955597  3.10194128]
     [ 2.37079504 -0.88819803  2.97545704  1.41742256 -3.95594055  2.45028256]
     [ 2.52860734 -0.94178795  2.97545704  0.84131987 -3.78447118  2.41008358]]
solutions_indices:
    [16, 17, 18, 19]
```

As we have 20 solutions, then there are `20/4 = 5` batches. As a result, the fitness function is called only 5 times per generation instead of 20. For each call, the fitness function receives a batch of 4 solutions.

As we have 5 generations, then the function will be called `5*5 = 25` times. Given the call to the fitness function after the last generation, then the total number of calls is `5*5 + 5 = 30`.

```python
import pygad
import numpy

function_inputs = [4,-2,3.5,5,-11,-4.7]
desired_output = 44

number_of_calls = 0

def fitness_func_batch(ga_instance, solutions, solutions_indices):
    global number_of_calls
    number_of_calls = number_of_calls + 1
    batch_fitness = []
    for solution in solutions:
        output = numpy.sum(solution*function_inputs)
        fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
        batch_fitness.append(fitness)
    return batch_fitness

ga_instance = pygad.GA(num_generations=5,
                       num_parents_mating=10,
                       sol_per_pop=20,
                       fitness_func=fitness_func_batch,
                       fitness_batch_size=4,
                       num_genes=len(function_inputs),
                       keep_elitism=0,
                       keep_parents=0)

ga_instance.run()
print(number_of_calls)
```

```
30
```

When batch fitness calculation is used, then we saved `120 - 30 = 90` calls to the fitness function. 
