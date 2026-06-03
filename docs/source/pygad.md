# `pygad` Module

This section of the documentation discusses the `pygad` module.

With the `pygad` module, you can create, run, save, and load instances of the genetic algorithm. It solves both single-objective and multi-objective optimization problems.

## `pygad.GA` Class

The `pygad` module has a class named `GA` for building the genetic algorithm. This section explains the class constructor, its methods, functions, and attributes.

### `__init__()`

To create an instance of the `pygad.GA` class, the constructor accepts several parameters. These let you adjust the genetic algorithm for different types of applications.

The `pygad.GA` class constructor supports the parameters below, grouped by purpose. Click a parameter to expand its full description.

#### Population and Generations

:::{dropdown} `num_generations`: Number of generations to run.
:animate: fade-in-slide-down

Number of generations.
:::

:::{dropdown} `num_parents_mating`: How many solutions are selected as parents.
:animate: fade-in-slide-down

Number of solutions to be selected as parents.
:::

:::{dropdown} `sol_per_pop`: Number of solutions in the population.
:animate: fade-in-slide-down

Number of solutions (i.e. chromosomes) within the population. This parameter has no action if `initial_population` parameter exists.
:::

:::{dropdown} `num_genes`: Number of genes in each solution.
:animate: fade-in-slide-down

Number of genes in the solution/chromosome. This parameter is not needed if the user feeds the initial population to the `initial_population` parameter.
:::

:::{dropdown} `initial_population`: Start from your own population.
:animate: fade-in-slide-down

A population you provide yourself to start the run instead of a random one. It defaults to `None`, in which case PyGAD builds the initial population from the `sol_per_pop` and `num_genes` parameters.

If `initial_population` is `None` and either `sol_per_pop` or `num_genes` is also `None`, an exception is raised.

Introduced in [PyGAD 2.0.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-0-0) and higher.
:::

:::{dropdown} `stop_criteria=None`: Stop early when a condition is met.
:animate: fade-in-slide-down

One or more conditions that stop the evolution early. Each criterion is a string made of a stop word and a number, like `"reach_40"`.

Four stop words are supported:

- `reach`: stop when the fitness is greater than or equal to a given value. Example: `"reach_40"` stops once the fitness is `>= 40`.
- `saturate`: stop when the fitness does not change for a given number of generations. Example: `"saturate_7"` stops if the fitness stays the same for 7 generations in a row.
- `time`: stop when the time spent inside `run()` is at least the given number of seconds. Example: `"time_30"` stops the run after 30 seconds.
- `evaluations`: stop when the number of fitness function calls made inside `run()` reaches the given count. Example: `"evaluations_1000"` stops the run once 1000 calls have been made.

You can also pass a list of criteria; the run stops as soon as any one of them is met.

Added in [PyGAD 2.15.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-15-0). The `time` and `evaluations` keywords were added in PyGAD 3.6.0.
:::

#### Fitness Function

:::{dropdown} `fitness_func`: Function that scores each solution.
:animate: fade-in-slide-down

The function (or method) that calculates the fitness of a solution. This is the one parameter you almost always need to set.

A fitness **function** must accept 3 parameters:

1. The instance of the `pygad.GA` class.
2. A single solution.
3. The index of the solution in the population.

If you pass a **method**, it takes a fourth parameter for the method's class instance.

Return a single number for a single-objective problem, or a `list`, `tuple`, or `numpy.ndarray` for a multi-objective problem (supported since [PyGAD 3.2.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-3-2-0)).

See [Preparing the fitness_func Parameter](https://pygad.readthedocs.io/en/latest/steps_to_use.html#preparing-the-fitness-func-parameter) for how to build one.
:::

:::{dropdown} `fitness_batch_size=None`: Score the solutions in batches.
:animate: fade-in-slide-down

Calculates the fitness in batches instead of one solution at a time.

- `1` or `None` (default): the fitness function is called once per solution.
- An integer where `1 < fitness_batch_size <= sol_per_pop`: solutions are grouped into batches of this size, and the fitness function is called once per batch.

See [Batch Fitness Calculation](https://pygad.readthedocs.io/en/latest/fitness_calculation.html#batch-fitness-calculation) for details and examples. Added in [PyGAD 2.19.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-19-0).
:::

#### Genes: Values and Types

:::{dropdown} `gene_type=float`: Data type (and precision) of the genes.
:animate: fade-in-slide-down

Sets the data type (and optional precision) of the genes. It defaults to `float`, so every gene is a `float`.

You can set it to:

- **One type for all genes:** a numeric type such as `int`, `float`, or any `numpy.int/uint/float(8-64)` type. Example: `gene_type=int`.
- **A type per gene:** a `list`, `tuple`, or `numpy.ndarray` with one type per gene. Example: `gene_type=[int, float, numpy.int8]`.
- **A float precision:** pair a `float` type with the number of decimal places. Example: `gene_type=[float, 2]`.

Version history:

- [PyGAD 2.9.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-9-0): a single numeric type can be used.
- [PyGAD 2.14.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-14-0): a type per gene can be used.
- [PyGAD 2.15.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-15-0): a precision can be set for `float` types.
:::

:::{dropdown} `gene_space=None`: Allowed values or range for each gene.
:animate: fade-in-slide-down

Sets the allowed values for each gene, so you can limit the search space to a range or to a set of discrete values.

You can set it to:

- **The same space for all genes:** a `list`/`tuple`/`range`/`numpy.ndarray`. Example: `gene_space=[0.3, 5.2, -4, 8]` limits every gene to those 4 values.
- **A space per gene:** a nested list/tuple, one sub-list per gene. Example: `gene_space=[[0.4, -5], [0.5, -3.2, 8.2, -9], ...]` (the first sub-list is for the first gene, and so on).
- **A continuous range:** a dictionary with `low` and `high` (and an optional `step`). Example: `{'low': 2, 'high': 4}` limits the gene to the range from 2 to 4.
- **`None` for a gene:** that gene is initialized from `init_range_low`/`init_range_high`, and mutated using `random_mutation_min_val`/`random_mutation_max_val`.

Version history:

- Added in [PyGAD 2.5.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-5-0).
- [PyGAD 2.9.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-9-0): NumPy arrays can be used.
- [PyGAD 2.11.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-11-0): a dictionary can set the low and high limits.
- [PyGAD 2.15.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-15-0): the `"step"` key was added.
:::

:::{dropdown} `gene_constraint=None`: Functions that restrict gene values.
:animate: fade-in-slide-down

A list of callables (functions), one per gene, that restrict the values a gene can take. Before a value is chosen for a gene, its callable checks that the candidate value is valid.

Added in [PyGAD 3.5.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-3-5-0). See the [Gene Constraint](https://pygad.readthedocs.io/en/latest/gene_values.html#gene-constraint) section for more information.
:::

:::{dropdown} `init_range_low=-4`: Lower bound for the initial gene values.
:animate: fade-in-slide-down

The lower value of the random range from which the gene values in the initial population are selected. `init_range_low` defaults to `-4`. Available in [PyGAD 1.0.20](https://pygad.readthedocs.io/en/latest/releases.html#pygad-1-0-20) and higher. This parameter has no action if the `initial_population` parameter exists.
:::

:::{dropdown} `init_range_high=4`: Upper bound for the initial gene values.
:animate: fade-in-slide-down

The upper value of the random range from which the gene values in the initial population are selected. `init_range_high` defaults to `+4`. Available in [PyGAD 1.0.20](https://pygad.readthedocs.io/en/latest/releases.html#pygad-1-0-20) and higher. This parameter has no action if the `initial_population` parameter exists.
:::

:::{dropdown} `allow_duplicate_genes=True`: Allow repeated values within a solution.
:animate: fade-in-slide-down

Added in [PyGAD 2.13.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-13-0). If `True`, then a solution/chromosome may have duplicate gene values. If `False`, then each gene will have a unique value in its solution.
:::

:::{dropdown} `sample_size=100`: Sample size used when searching for a valid value.
:animate: fade-in-slide-down

The size of the sample of candidate values PyGAD draws when it needs to pick a gene value. It defaults to `100`.

It is useful when `allow_duplicate_genes=False` or `gene_constraint` is used. If PyGAD cannot find a unique value or a value that meets a constraint, increase this parameter.

Added in [PyGAD 3.5.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-3-5-0). See the [sample_size Parameter](https://pygad.readthedocs.io/en/latest/gene_values.html#sample-size-parameter) section for more information.
:::

#### Parent Selection

:::{dropdown} `parent_selection_type="sss"`: How the parents are selected.
:animate: fade-in-slide-down

How the parents are selected. It defaults to `"sss"`.

The built-in types are:

- `sss`: steady-state selection.
- `rws`: roulette wheel selection.
- `sus`: stochastic universal selection.
- `rank`: rank selection.
- `random`: random selection.
- `tournament`: tournament selection.
- `nsga2`: NSGA-II selection (multi-objective).
- `tournament_nsga2`: Tournament selection that ranks competitors with NSGA-II non-dominated sorting and crowding distance.
- `nsga3`: NSGA-III selection (multi-objective). Requires the `nsga3_num_divisions` parameter.
- `tournament_nsga3`: Tournament selection that ranks competitors with NSGA-III niche count instead of crowding distance. Requires the `nsga3_num_divisions` parameter.

You can also pass your own parent selection function (since [PyGAD 2.16.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-16-0)). See [User-Defined Crossover, Mutation, and Parent Selection Operators](https://pygad.readthedocs.io/en/latest/user_defined_operators.html#user-defined-crossover-mutation-and-parent-selection-operators).
:::

:::{dropdown} `K_tournament=3`: Contestants per tournament selection.
:animate: fade-in-slide-down

In case that the parent selection type is `tournament`, the `K_tournament` specifies the number of parents participating in the tournament selection. It defaults to `3`.
:::

:::{dropdown} `nsga3_num_divisions=None`: Number of divisions per objective axis for NSGA-III.
:animate: fade-in-slide-down

Only used when `parent_selection_type` is `'nsga3'` or `'tournament_nsga3'`. It is the number of divisions per objective axis used to build the structured reference points (the `p` parameter from Deb & Jain 2014). The total number of reference points is `C(M + p - 1, p)` where `M` is the number of objectives. Must be a positive integer. Defaults to `None`.

If `sol_per_pop` is smaller than the resulting number of reference points, PyGAD raises a warning and grows the population to match before the generational loop starts.
:::

#### Keeping Solutions

:::{dropdown} `keep_elitism=1`: Keep the best solutions each generation.
:animate: fade-in-slide-down

The number of best solutions (the elitism) to keep in the next generation. It defaults to `1`, so only the best solution is kept.

- `0`: elitism is turned off.
- A positive integer `K` (with `0 <= keep_elitism <= sol_per_pop`): the best `K` solutions are kept.

If this parameter is not `0`, then `keep_parents` has no effect.

Added in [PyGAD 2.18.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-18-0). To see how `keep_elitism` and `keep_parents` work together, see [How the Number of Offspring Is Decided](https://pygad.readthedocs.io/en/latest/generations.html#how-the-number-of-offspring-is-decided).
:::

:::{dropdown} `keep_parents=-1`: Keep the parents in the next generation.
:animate: fade-in-slide-down

The number of parents to keep in the next population. It defaults to `-1`.

- `-1`: keep all the parents.
- `0`: keep no parents.
- A positive integer: keep that many parents.

The value cannot be less than `-1` or greater than `sol_per_pop`.

This parameter has an effect only when `keep_elitism=0` (since [PyGAD 2.18.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-18-0)). Since PyGAD 2.20.0, the parents' fitness from the last generation is not re-used if `keep_parents=0`.

To see how `keep_parents` and `keep_elitism` work together, see [How the Number of Offspring Is Decided](https://pygad.readthedocs.io/en/latest/generations.html#how-the-number-of-offspring-is-decided).
:::

#### Crossover

:::{dropdown} `crossover_type="single_point"`: How parents are combined into offspring.
:animate: fade-in-slide-down

The type of crossover. It defaults to `"single_point"`.

The built-in types are:

- `single_point`: single-point crossover.
- `two_points`: two-point crossover.
- `uniform`: uniform crossover.
- `scattered`: scattered crossover (since [PyGAD 2.9.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-9-0)).
- `sbx`: simulated binary crossover. The standard real-coded operator. Requires the `sbx_crossover_eta` parameter.

You can also pass your own crossover function (since [PyGAD 2.16.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-16-0)). See [User-Defined Crossover, Mutation, and Parent Selection Operators](https://pygad.readthedocs.io/en/latest/user_defined_operators.html#user-defined-crossover-mutation-and-parent-selection-operators).

If `crossover_type=None`, the crossover step is skipped and no offspring are created, so the next generation reuses the current population (since [PyGAD 2.2.2](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-2-2)).
:::

:::{dropdown} `sbx_crossover_eta=30`: Distribution index for SBX crossover.
:animate: fade-in-slide-down

Only used when `crossover_type` is `'sbx'`. Sets how close the children stay to the parents. A higher value means children stay closer. Must be a positive number. Defaults to `30`.
:::

:::{dropdown} `crossover_probability=None`: Chance a parent is used for crossover.
:animate: fade-in-slide-down

The probability of selecting a parent for crossover. Its value must be between 0.0 and 1.0.

For each parent, a random value between 0.0 and 1.0 is generated. If that value is less than or equal to `crossover_probability`, the parent is selected.

Added in [PyGAD 2.5.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-5-0) and higher.
:::

#### Mutation

:::{dropdown} `mutation_type="random"`: How offspring genes are mutated.
:animate: fade-in-slide-down

The type of mutation. It defaults to `"random"`.

The built-in types are:

- `random`: random mutation.
- `swap`: swap mutation.
- `inversion`: inversion mutation.
- `scramble`: scramble mutation.
- `adaptive`: adaptive mutation (since [PyGAD 2.10.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-10-0)). See [Adaptive Mutation](https://pygad.readthedocs.io/en/latest/adaptive_mutation.html#adaptive-mutation) and [Use Adaptive Mutation in PyGAD](https://pygad.readthedocs.io/en/latest/adaptive_mutation.html#use-adaptive-mutation-in-pygad).
- `polynomial`: polynomial mutation. The standard real-coded operator used together with SBX. Requires the `polynomial_mutation_eta` parameter.

You can also pass your own mutation function (since [PyGAD 2.16.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-16-0)). See [User-Defined Crossover, Mutation, and Parent Selection Operators](https://pygad.readthedocs.io/en/latest/user_defined_operators.html#user-defined-crossover-mutation-and-parent-selection-operators).

If `mutation_type=None`, the mutation step is skipped and the offspring are used unchanged (since [PyGAD 2.2.2](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-2-2)).
:::

:::{dropdown} `polynomial_mutation_eta=20`: Distribution index for polynomial mutation.
:animate: fade-in-slide-down

Only used when `mutation_type` is `'polynomial'`. Sets the size of the change. A higher value means a smaller change. Must be a positive number. Defaults to `20`.
:::

:::{dropdown} `mutation_probability=None`: Per-gene chance of mutation.
:animate: fade-in-slide-down

The probability of selecting a gene for mutation. Its value must be between 0.0 and 1.0.

For each gene, a random value between 0.0 and 1.0 is generated. If that value is less than or equal to `mutation_probability`, the gene is mutated.

If this parameter is set, you do not need `mutation_percent_genes` or `mutation_num_genes`. Added in [PyGAD 2.5.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-5-0) and higher.
:::

:::{dropdown} `mutation_by_replacement=False`: Replace the gene value instead of adding to it.
:animate: fade-in-slide-down

A bool that controls how `random` mutation changes a gene. It works only when `mutation_type="random"`.

- `True`: replace the gene with the randomly generated value.
- `False` (default): add the random value to the gene.

Supported in [PyGAD 2.2.2](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-2-2) and higher. See the [PyGAD 2.2.2](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-2-2) release notes for an example.
:::

:::{dropdown} `mutation_percent_genes="default"`: Percentage of genes to mutate.
:animate: fade-in-slide-down

The percentage of genes to mutate. It defaults to the string `"default"`, which becomes `10` (10% of the genes). The value must be `> 0` and `<= 100`.

PyGAD uses this percentage to compute `mutation_num_genes`.

This parameter has no effect if `mutation_probability` or `mutation_num_genes` is set, or if `mutation_type` is `None` (since [PyGAD 2.2.2](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-2-2)).
:::

:::{dropdown} `mutation_num_genes=None`: Number of genes to mutate.
:animate: fade-in-slide-down

The number of genes to mutate. It defaults to `None`, meaning no number is set.

This parameter has no effect if `mutation_probability` is set, or if `mutation_type` is `None` (since [PyGAD 2.2.2](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-2-2)).
:::

:::{dropdown} `random_mutation_min_val=-1.0`: Lower bound of the random mutation value.
:animate: fade-in-slide-down

For `random` mutation, the start of the range from which a random value is drawn and added to the gene. It defaults to `-1`.

This parameter has no effect if `mutation_type` is `None` (since [PyGAD 2.2.2](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-2-2)).
:::

:::{dropdown} `random_mutation_max_val=1.0`: Upper bound of the random mutation value.
:animate: fade-in-slide-down

For `random` mutation, the end of the range from which a random value is drawn and added to the gene. It defaults to `+1`.

This parameter has no effect if `mutation_type` is `None` (since [PyGAD 2.2.2](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-2-2)).
:::

#### Lifecycle Callbacks

:::{dropdown} `on_start=None`: Called once before the run starts.
:animate: fade-in-slide-down

A function (or method) called once before the run starts.

- As a **function**, it takes 1 parameter: the instance of the genetic algorithm.
- As a **method**, it takes a second parameter for the method's object.

Added in [PyGAD 2.6.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-6-0).
:::

:::{dropdown} `on_fitness=None`: Called after the fitness is calculated.
:animate: fade-in-slide-down

A function (or method) called after the fitness of all solutions is calculated.

- As a **function**, it takes 2 parameters: a list of all the solutions' fitness values, and the instance of the genetic algorithm.
- As a **method**, it takes a third parameter for the method's object.

Added in [PyGAD 2.6.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-6-0).
:::

:::{dropdown} `on_parents=None`: Called after the parents are selected.
:animate: fade-in-slide-down

A function (or method) called after the parents are selected.

- As a **function**, it takes 2 parameters: the selected parents, and the instance of the genetic algorithm.
- As a **method**, it takes a third parameter for the method's object.

Added in [PyGAD 2.6.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-6-0).
:::

:::{dropdown} `on_crossover=None`: Called after crossover.
:animate: fade-in-slide-down

A function called each time crossover is applied. It takes 2 parameters: the instance of the genetic algorithm, and the offspring generated by crossover.

Added in [PyGAD 2.6.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-6-0).
:::

:::{dropdown} `on_mutation=None`: Called after mutation.
:animate: fade-in-slide-down

A function called each time mutation is applied. It takes 2 parameters: the instance of the genetic algorithm, and the offspring after mutation.

Added in [PyGAD 2.6.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-6-0).
:::

:::{dropdown} `on_generation=None`: Called after each generation.
:animate: fade-in-slide-down

A function called after each generation. It takes 1 parameter: the instance of the genetic algorithm.

If it returns the string `"stop"`, the `run()` method stops without completing the remaining generations.

Added in [PyGAD 2.6.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-6-0).
:::

:::{dropdown} `on_stop=None`: Called once when the run ends.
:animate: fade-in-slide-down

A function called once just before the run ends (or after the last generation). It takes 2 parameters: the instance of the genetic algorithm, and the list of the last population's fitness values.

Added in [PyGAD 2.6.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-6-0).
:::

#### Saving and Logging

:::{dropdown} `save_best_solutions=False`: Save the best solution of each generation.
:animate: fade-in-slide-down

When `True`, the best solution of each generation is saved into the `best_solutions` attribute. When `False` (default), nothing is saved and `best_solutions` stays empty.

Supported in [PyGAD 2.9.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-9-0).
:::

:::{dropdown} `save_solutions=False`: Save every solution of each generation.
:animate: fade-in-slide-down

If `True`, then all solutions in each generation are appended into an attribute called `solutions` which is NumPy array. Supported in [PyGAD 2.15.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-15-0).
:::

:::{dropdown} `logger=None`: Custom logger for the outputs.
:animate: fade-in-slide-down

An instance of the `logging.Logger` class used to log the outputs. When set, messages are logged instead of printed with `print()`. If `None`, PyGAD creates a logger that uses a `StreamHandler` to write the messages to the console.

Added in [PyGAD 3.0.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-3-0-0). See [Logging Outputs](https://pygad.readthedocs.io/en/latest/logging.html#logging-outputs) for more information.
:::

:::{dropdown} `suppress_warnings=False`: Turn warning messages on or off.
:animate: fade-in-slide-down

A bool parameter to control whether the warning messages are printed or not. It defaults to `False`.
:::

#### Performance and Reproducibility

:::{dropdown} `parallel_processing=None`: Use threads or processes to speed up fitness.
:animate: fade-in-slide-down

Runs the fitness calculation in parallel. It defaults to `None` (no parallel processing).

You can set it to:

- **A positive integer:** the number of threads. Example: `parallel_processing=5` uses 5 threads (the same as `["thread", 5]`).
- **A list/tuple of 2 elements:** the first is `"process"` or `"thread"`; the second is the number of processes or threads. Example: `parallel_processing=["process", 10]` uses 10 processes.

Added in [PyGAD 2.17.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-17-0). See [Parallel Processing in PyGAD](https://pygad.readthedocs.io/en/latest/fitness_calculation.html#parallel-processing-in-pygad) for more information.
:::

:::{dropdown} `random_seed=None`: Seed for reproducible runs.
:animate: fade-in-slide-down

The random seed used by the NumPy and `random` number generators. Setting it makes runs reproducible (for example, `random_seed=2`). It defaults to `None`, which means no seed is used.

Added in [PyGAD 2.18.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-18-0).
:::

You do not have to set all of these parameters when you create an instance of the `GA` class. The most important one is `fitness_func`, which defines the fitness function.

It is OK to set the value of any of the 2 parameters `init_range_low` and `init_range_high` to be equal, higher, or lower than the other parameter (i.e. `init_range_low` is not needed to be lower than `init_range_high`). The same holds for the `random_mutation_min_val` and `random_mutation_max_val` parameters.

If both the `mutation_type` and `crossover_type` parameters are `None`, then the genetic algorithm cannot evolve at all. As a result, it cannot find a solution better than the best solution in the initial population.

The parameters are validated by calling the `validate_parameters()` method of the `utils.validation.Validation` class inside the constructor. If any parameter is not correct, an exception is raised and the `valid_parameters` attribute is set to `False`.

## Extended Classes

To keep the library modular and structured, the code is split into several scripts, where each script has one or more classes. Each class has its own purpose.

Here is the list of scripts and the classes that the `pygad.GA` class extends:

1. `utils/engine.py`:
   1. `utils.engine.GAEngine`: Runs the GA loop and owns the run-time lifecycle helpers.
2. `utils/validation.py`
   1. `utils.validation.Validation`: Validates every constructor parameter and dispatches `parent_selection_type` / `crossover_type` / `mutation_type` to the right method.
3. `utils/parent_selection.py`
   1. `utils.parent_selection.ParentSelection`: All built-in parent selection operators, including `nsga2_selection`, `tournament_selection_nsga2`, `nsga3_selection`, and `tournament_selection_nsga3`.
4. `utils/crossover.py`
   1. `utils.crossover.Crossover`: Built-in crossover operators.
5. `utils/mutation.py`
   1. `utils.mutation.Mutation`: Built-in mutation operators.
6. `utils/nsga.py`
   1. `utils.nsga.NSGA`: Building blocks shared by NSGA-II and NSGA-III (`non_dominated_sorting`, `get_non_dominated_set`).
7. `utils/nsga2.py`
   1. `utils.nsga2.NSGA2`: NSGA-II specific primitives (`crowding_distance`, `sort_solutions_nsga2`).
8. `utils/nsga3.py`
   1. `utils.nsga3.NSGA3`: NSGA-III algorithm primitives (reference points, ideal point, extreme points, intercepts, normalization, association, niching).
9. `utils/report.py`
   1. `utils.report.Report`: Builds a PDF report of the run (`generate_report`).
10. `helper/unique.py`
    1. `helper.unique.Unique`: Routines that resolve duplicate genes inside a solution.
11. `helper/misc.py`
    1. `helper.misc.Helper`: Generic helpers used across the library (population dtype handling, per-gene value generation, constraint sampling, lifecycle summary).
12. `visualize/plot.py`
    1. `visualize.plot.Plot`: All plot methods. See [`pygad.visualize`](https://pygad.readthedocs.io/en/latest/visualize.html).

Since the `pygad.GA` class extends such classes, the attributes and methods inside them can be retrieved by instances of the `pygad.GA` class.

### Class Attributes

* `supported_int_types`: A list of the supported types for the integer numbers.
* `supported_float_types`: A list of the supported types for the floating-point numbers.
* `supported_int_float_types`: A list of the supported types for all numbers. It just concatenates the previous 2 lists.

### Other Instance Attributes & Methods

All the parameters and functions passed to the `pygad.GA` class constructor are used as class attributes and methods in the instances of the `pygad.GA` class. In addition to such attributes, there are other attributes and methods added to the instances of the `pygad.GA` class.

> The `GA` class gains the attributes of its parent classes via inheritance, making them accessible through the `GA` object even if they are defined externally to its specific class body.

> Names that begin with an underscore (for example `_bootstrap_nsga3_reference_points`) are internal helpers. They are listed below for completeness but are not part of the stable API; do not rely on their signature staying the same across releases.

#### Lifecycle

##### Attributes

- `generations_completed`: Number of the last completed generation.
- `run_completed`: Set to `True` only after the `run()` method completes gracefully.
- `valid_parameters`: Set to `True` when all the parameters passed in the `GA` class constructor are valid.
- `run_start_time`: Monotonic clock value captured right before the generation loop starts. Internal.
- `logger`: Logger object from the `logging` module. Supported in [PyGAD 3.0.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-3-0-0).

##### Methods

- `run()`: Runs the generation loop. The main entry point.
- `run_loop_head(best_solution_fitness)`: Per-generation pre-loop bookkeeping. Internal; called from inside `run()`. Added in [PyGAD 3.3.1](https://pygad.readthedocs.io/en/latest/releases.html#pygad-3-3-1).
- `run_select_parents(call_on_parents=True)`: Select parents and call `on_parents` when defined. Internal; called from inside `run()`. Pass `call_on_parents=False` when refreshing the parent set at the end of `run()`. Added in [PyGAD 3.3.1](https://pygad.readthedocs.io/en/latest/releases.html#pygad-3-3-1).
- `run_crossover()`: Apply crossover and call `on_crossover` when defined. Internal. Added in [PyGAD 3.3.1](https://pygad.readthedocs.io/en/latest/releases.html#pygad-3-3-1).
- `run_mutation()`: Apply mutation and call `on_mutation` when defined. Internal. Added in [PyGAD 3.3.1](https://pygad.readthedocs.io/en/latest/releases.html#pygad-3-3-1).
- `run_update_population()`: Replace `self.population` with the crossed-over and mutated offspring. Internal. Added in [PyGAD 3.3.1](https://pygad.readthedocs.io/en/latest/releases.html#pygad-3-3-1).
- `summary(...)`: Prints a Keras-like summary of the PyGAD lifecycle. Added in [PyGAD 2.19.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-19-0). See [Print Lifecycle Summary](https://pygad.readthedocs.io/en/latest/logging.html#print-lifecycle-summary).

#### Population and Initialization

##### Attributes

- `population`: A NumPy array that initially holds the initial population and is later updated after each generation.
- `initial_population`: Frozen copy of the initial population, set after `initialize_population` runs.
- `pop_size`: A `(sol_per_pop, num_genes)` tuple describing the population shape.
- `gene_type_single`: `True` when every gene shares the same dtype; `False` when `gene_type` is a list/tuple/numpy.ndarray. Added in [PyGAD 2.14.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-14-0).
- `gene_space_unpacked`: Unpacked version of `gene_space`. For example, `range(1, 5)` becomes `[1, 2, 3, 4]`; `{'low': 2, 'high': 4}` becomes a finite sample. Added in [PyGAD 3.1.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-3-1-0).

##### Methods

- `initialize_population(allow_duplicate_genes, gene_type, gene_constraint)`: Build the initial population, apply gene types and constraints, resolve duplicates when not allowed.
- `initialize_parents_array(shape)`: Allocate an empty parents (or offspring) array with the right dtype.
- `change_population_dtype_and_round(population)`: Cast a 2D population to the dtype encoded in `self.gene_type` and round non-integer genes.
- `change_gene_dtype_and_round(gene_index, gene_value)`: Same as above, but for a single gene value.
- `round_genes(solutions)`: Round genes in a 2D array according to `self.gene_type` precision.
- `get_initial_population_range(gene_index)`: Return the `[init_range_low, init_range_high]` window for a specific gene.
- `get_random_mutation_range(gene_index)`: Return the `[random_mutation_min_val, random_mutation_max_val]` window for a specific gene.
- `get_gene_dtype(gene_index)`: Return the `(type, precision)` pair for a specific gene.
- `generate_gene_value(...)`: Sample a single gene value from the gene space or from the configured range.
- `generate_gene_value_from_space(...)`: Sample a single gene value from `gene_space`.
- `generate_gene_value_randomly(...)`: Sample a single gene value from the configured numeric range.

#### Fitness

##### Attributes

- `last_generation_fitness`: Fitness values of the solutions in the last generation. Added in [PyGAD 2.12.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-12-0).
- `previous_generation_fitness`: Fitness of the population one step before `last_generation_fitness`. Used to skip re-evaluating solutions PyGAD has already seen. Added in [PyGAD 2.16.2](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-16-2).
- `best_solutions_fitness`: List of best-solution fitness per generation.
- `best_solutions`: A NumPy array of the best solution per generation. Only populated when `save_best_solutions=True`.
- `best_solutions_fitness`: Fitness for every entry in `best_solutions`.
- `solutions`: All visited solutions when `save_solutions=True`.
- `solutions_fitness`: Fitness for every entry in `solutions`.
- `best_solution_generation`: Generation at which the best fitness was reached. `-1` until `run()` completes.

##### Methods

- `cal_pop_fitness()`: Compute the fitness of every solution in the current population, reusing previously calculated values where possible.
- `best_solution(pop_fitness=None)`: Return the best solution, its fitness, and its population index.
- `adaptive_mutation_population_fitness(offspring)`: Average fitness used by adaptive mutation to split solutions into low / high quality.

#### Parent Selection (general)

##### Attributes

- `last_generation_parents`: Parents selected in the last generation. Added in [PyGAD 2.12.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-12-0).
- `last_generation_parents_indices`: Indices of the selected parents in `self.population`. Added in [PyGAD 2.15.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-15-0).

##### Methods

- `select_parents(fitness, num_parents)`: Active parent-selection method. Bound during validation according to `parent_selection_type`.
- `steady_state_selection(fitness, num_parents)`: Steady-state selection.
- `rank_selection(fitness, num_parents)`: Rank-based selection.
- `random_selection(fitness, num_parents)`: Random selection.
- `tournament_selection(fitness, num_parents)`: K-tournament selection.
- `roulette_wheel_selection(fitness, num_parents)`: Roulette-wheel selection.
- `stochastic_universal_selection(fitness, num_parents)`: SUS selection.
- `wheel_cumulative_probs(probs, num_parents)`: Build the `[start, end)` ranges used by RWS and SUS.

#### Multi-Objective Optimization (NSGA-II)

##### Attributes

- `pareto_fronts`: List of the Pareto fronts of the last generation when running a multi-objective problem. Each front is a NumPy array of `(population_index, fitness_vector)` pairs. Added in [PyGAD 3.2.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-3-2-0).

##### Methods

- `non_dominated_sorting(fitness)`: Sort the population into Pareto fronts. Defined in `utils.nsga.NSGA` and shared with NSGA-III.
- `get_non_dominated_set(curr_solutions)`: Split the current set of solutions into a dominated and non-dominated subset. Defined in `utils.nsga.NSGA`.
- `crowding_distance(pareto_front, fitness)`: Per-solution crowding distance inside a Pareto front. Defined in `utils.nsga2.NSGA2`.
- `sort_solutions_nsga2(fitness, find_best_solution=False)`: Sort population indices best-to-worst using Pareto fronts and crowding distance for MOO; descending fitness for SOO. Defined in `utils.nsga2.NSGA2`.
- `nsga2_selection(fitness, num_parents)`: NSGA-II parent selection. Defined in `utils.parent_selection.ParentSelection`.
- `tournament_selection_nsga2(fitness, num_parents)`: K-tournament with non-dominated rank + crowding distance as tiebreakers. Defined in `utils.parent_selection.ParentSelection`.

#### Multi-Objective Optimization (NSGA-III)

##### Attributes

- `nsga3_num_divisions`: Stored value of the `nsga3_num_divisions` constructor parameter. Used when building the reference grid.
- `nsga3_reference_points`: Structured grid of reference points on the unit simplex. A 2D NumPy array of shape `(n_points, num_objectives)` where each row sums to 1. Built once before the generation loop starts (`_bootstrap_nsga3_reference_points`). Re-used for every generation.

##### Methods (algorithm primitives in `utils.nsga3.NSGA3`)

- `nsga3_generate_reference_points(num_objectives, num_divisions)`: Build the Das-Dennis grid.
- `nsga3_compute_ideal_point(fitness)`: Best fitness per objective across the input rows (column max under maximization).
- `nsga3_find_extreme_points(fitness, ideal_point, epsilon=NSGA3_ASF_EPSILON)`: For each objective, find the row that best represents the corner of that axis using the ASF.
- `nsga3_compute_intercepts(extreme_points, ideal_point, fallback_fitness)`: Fit a hyperplane through the extreme points and return per-axis intercepts. Falls back to the nadir on singular systems.
- `nsga3_normalize_fitness(fitness, ideal_point, intercepts)`: Scale each fitness row to the unit hypercube and clip outliers to `[0, 1]`.
- `nsga3_associate_to_reference_points(normalized, reference_points)`: For every normalized row, find the closest reference line and the perpendicular distance to it.
- `nsga3_niching_select(critical_front_indices, critical_front_associations, critical_front_distances, accepted_associations, num_reference_points, num_to_select)`: Niching loop that picks survivors from the critical front to preserve diversity across reference points.

##### Methods (selection in `utils.parent_selection.ParentSelection`)

- `nsga3_selection(fitness, num_parents)`: NSGA-III parent selection.
- `tournament_selection_nsga3(fitness, num_parents)`: K-tournament with niche count + perpendicular distance as tiebreakers.
- `_nsga3_pick_critical_front_survivors(...)`: Run normalization and niching on `P_next ∪ critical_front` and return the picked survivors. Internal.
- `_nsga3_pick_tournament_winner(...)`: Decide the winner of one K-tournament round under NSGA-III rules. Internal.
- `_nsga3_build_parents(final_indices, num_parents)`: Copy the chosen rows out of the population into a new parents array. Internal.

##### Methods (bootstrap and population growth in `utils.engine.GAEngine`)

- `_bootstrap_nsga3_reference_points()`: Build the reference-point grid once, right after the first fitness evaluation. Calls `_nsga3_grow_population` when `sol_per_pop` is smaller than the reference count.
- `_nsga3_grow_population(required_size, num_objectives)`: Append random solutions to `self.population`, update `sol_per_pop` / `pop_size` / `num_offspring`, re-evaluate fitness.
- `_nsga3_generate_extra_random_solutions(count)`: Build `count` random solutions respecting the gene space, init range, gene type, gene constraints, and `allow_duplicate_genes` rules.
- `_nsga3_generate_single_random_gene(gene_idx, partial_solution)`: Sample a single gene value using initial-population settings (not mutation settings).
- `_nsga3_apply_gene_constraints(population)`: Enforce `gene_constraint` on the new rows.
- `_nsga3_resolve_duplicate_genes(population)`: Resolve duplicate genes in the new rows when `allow_duplicate_genes=False`.

##### Module-level helpers (in `pygad.utils.nsga3`)

- `NSGA3_ASF_EPSILON`: Off-axis weight used by the ASF inside `nsga3_find_extreme_points`.
- `NSGA3_INTERCEPT_NEAR_ZERO`: Threshold under which an intercept gap is treated as zero.
- `_nsga3_pick_target_reference_point(niche_counts, critical_front_associations, remaining_positions)`: Choose the next reference point for the niching loop. Internal.
- `_nsga3_pick_candidate_at_reference(candidates_at_target, critical_front_distances, niche_count_at_target)`: Choose a candidate at a given reference point. Internal.
- `_nsga3_enumerate_compositions(num_objectives, num_divisions)`: Yield every non-negative integer tuple summing to `num_divisions`. Internal.

##### Module-level helpers (in `pygad.utils.parent_selection`)

- `_nsga3_validate_multi_objective_fitness(fitness, supported_int_float_types, method_name)`: Raise when the GA was set to NSGA-III but the fitness function returned scalars.
- `_nsga3_accumulate_fronts(pareto_fronts, num_parents)`: Walk the Pareto fronts and split them into the accepted set + the critical front.

#### Crossover

##### Attributes

- `last_generation_offspring_crossover`: Offspring after crossover. Added in [PyGAD 2.12.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-12-0).

##### Methods

- `crossover()`: Active crossover operator. Bound during validation according to `crossover_type`.
- `single_point_crossover(parents, offspring_size)`: Single-point crossover.
- `two_points_crossover(parents, offspring_size)`: Two-point crossover.
- `uniform_crossover(parents, offspring_size)`: Uniform crossover.
- `scattered_crossover(parents, offspring_size)`: Scattered crossover.
- `sbx_crossover(parents, offspring_size)`: Simulated binary crossover. Uses `self.sbx_crossover_eta`.

#### Mutation

##### Attributes

- `last_generation_offspring_mutation`: Offspring after mutation. Added in [PyGAD 2.12.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-12-0).
- `last_generation_offspring_mutation_indices`: Indices of mutated offspring inside `self.population`.

##### Methods

- `mutation()`: Active mutation operator. Bound during validation according to `mutation_type`.
- `random_mutation(offspring)`: Random mutation (replaces or adds a uniform random value).
- `swap_mutation(offspring)`: Swap mutation.
- `inversion_mutation(offspring)`: Inversion mutation.
- `scramble_mutation(offspring)`: Scramble mutation.
- `adaptive_mutation(offspring)`: Adaptive mutation. Uses `adaptive_mutation_population_fitness`.
- `polynomial_mutation(offspring)`: Polynomial mutation. Uses `self.polynomial_mutation_eta`.
- `mutation_change_gene_dtype_and_round(...)`: Round and re-cast a mutated gene to the configured dtype/precision.

#### Elitism

##### Attributes

- `last_generation_elitism`: Elitism solutions from the last generation. Added in [PyGAD 2.18.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-18-0).
- `last_generation_elitism_indices`: Population indices of `last_generation_elitism`. Added in [PyGAD 2.19.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-19-0).

#### Gene Constraints and Duplicate Resolution

##### Methods

- `validate_gene_constraint_callable_output(selected_values, values)`: Sanity-check the return value of a user-defined `gene_constraint`.
- `filter_gene_values_by_constraint(values, solution, gene_idx)`: Run `gene_constraint[gene_idx]` and return the filtered list.
- `get_valid_gene_constraint_values(...)`: Sample candidate values until one satisfies the gene constraint.
- `solve_duplicate_genes_randomly(...)`: Resolve duplicate genes by sampling new values from the random range.
- `solve_duplicate_genes_by_space(...)`: Resolve duplicate genes by sampling new values from `gene_space`.
- `solve_duplicates_deeply(...)`: Slow, exhaustive fallback for duplicate resolution.
- `unique_int_gene_from_range(...)`: Pick an integer gene that does not already appear in the solution.
- `unique_float_gene_from_range(...)`: Pick a float gene that does not already appear in the solution.
- `unique_gene_by_space(...)`: Pick a unique value from `gene_space`.
- `unique_genes_by_space(...)`: Pick unique values for several genes from `gene_space`.
- `select_unique_value(...)`: Sample one value uniformly at random from a list of candidates.
- `find_two_duplicates(solution)`: Locate the first pair of duplicated indices in a solution.
- `unpack_gene_space(...)`: Materialize the unpacked `gene_space` (used to build `gene_space_unpacked`).

#### Saving, Loading, and Reporting

##### Methods

- `save(filename)`: Pickle the GA instance to disk (uses `cloudpickle`).
- `generate_report(filename, ...)`: Build a PDF report of the run. See [`generate_report()`](#generate-report) below.
- `push_to_vilvik(...)`: Optional convenience wrapper around the Vilvik SDK.

Note that the attributes with names starting with `last_generation_` are updated after each generation.

The next sections discuss the methods available in the `pygad.GA` class.

### `save()`

The `save()` method in the `pygad.GA` class saves the genetic algorithm instance as a pickled object.

Accepts the following parameter:

* `filename`: Name of the file to save the instance. No extension is needed.

### `generate_report()`

Builds a PDF report of the current GA run. It bundles the configuration table, a run-summary table, the best solution, and every applicable plot. Requires the optional `report` extra:

```
pip install pygad[report]
```

Call it after `run()` finishes. A minimal example:

```python
ga_instance.run()
ga_instance.generate_report("my_run")  # writes my_run.pdf next to the script
```

Parameters:

- `filename` (`str`, required): Output path. `.pdf` is appended automatically if missing.
- `title` (`str` or `None`, default `None`): Title shown on the first page. Defaults to `"PyGAD run report"`.
- `sections` (iterable of `str` or `None`, default `None`): Sections to include and their order. Valid entries are `"title"`, `"configuration"`, `"run_summary"`, `"best_solution"`, `"plots"`, and `"notes"`. When `None`, every section is included in their default order.
- `include_plots` (iterable of `str`, `"all"`, or `None`, default `None`): Plots to embed under the `"plots"` section. `None` or `"all"` auto-selects every plot whose preconditions are met by this run. Pass a list of plot method names to include only those.
- `figure_size_inches` (`(float, float)`, default `(7.0, 4.5)`): Width and height (in inches) used when each plot is drawn for the report.
- `notes` (`str` or `None`, default `None`): Free-form text rendered in the optional `"notes"` section.
- `page_size` (`str`, default `"letter"`): Either `"letter"` or `"A4"`.

The report skips any plot whose preconditions are not met. For example, `plot_pareto_front_curve` is included only for multi-objective runs with 2 or 3 objectives; `plot_non_dominated_hypervolume` is included only when `save_solutions=True` is set on the GA. A full example lives at [`examples/example_generate_report.py`](https://github.com/ahmedfgad/GeneticAlgorithmPython/tree/master/examples/example_generate_report.py).

## Functions in `pygad`

Besides the methods available in the `pygad.GA` class, this section discusses the functions available in `pygad`. Up to this time, there is only a single function named `load()`.

### `pygad.load()`

Reads a saved instance of the genetic algorithm. This is not a method but a function that is indented under the `pygad` module. So, it could be called by the pygad module as follows: `pygad.load(filename)`.

Accepts the following parameter:

* `filename`: Name of the file holding the saved instance of the genetic algorithm. No extension is needed.

Returns the genetic algorithm instance.

## Using PyGAD

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Steps to Use PyGAD
:link: steps_to_use
:link-type: doc

A step-by-step walkthrough to build and run the genetic algorithm.
:::

:::{grid-item-card} Life Cycle of PyGAD
:link: lifecycle
:link-type: doc

How a generation runs and where each callback is called.
:::

::::

## Examples

This section gives the complete code of some examples that use `pygad`. Each subsection builds a different example.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Linear Model - Single Objective
:link: pygad_example_linear
:link-type: doc
:::

:::{grid-item-card} Linear Model - Multi-Objective
:link: pygad_example_multi_objective
:link-type: doc
:::

:::{grid-item-card} Reproducing Images
:link: pygad_example_reproducing_images
:link-type: doc
:::

::::

### Clustering

For a 2-cluster problem, the code is available [here](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/master/example_clustering_2.py). For a 3-cluster problem, the code is [here](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/master/example_clustering_3.py). The 2 examples are using artificial samples.

Soon a tutorial will be published at [Paperspace](https://blog.paperspace.com/author/ahmed) to explain how clustering works using the genetic algorithm with examples in PyGAD.

### CoinTex Game Playing using PyGAD

The code is available at the [CoinTex GitHub project](https://github.com/ahmedfgad/CoinTex/tree/master/PlayerGA). CoinTex is an Android game written in Python using the Kivy framework. Find CoinTex at [Google Play](https://play.google.com/store/apps/details?id=coin.tex.cointexreactfast): https://play.google.com/store/apps/details?id=coin.tex.cointexreactfast

Check this [Paperspace tutorial](https://blog.paperspace.com/building-agent-for-cointex-using-genetic-algorithm) for how the genetic algorithm plays CoinTex: https://blog.paperspace.com/building-agent-for-cointex-using-genetic-algorithm. Check also this [YouTube video](https://youtu.be/Sp_0RGjaL-0) showing the genetic algorithm while playing CoinTex.

:::{toctree}
:hidden:

steps_to_use
lifecycle
pygad_example_linear
pygad_example_multi_objective
pygad_example_reproducing_images
:::
