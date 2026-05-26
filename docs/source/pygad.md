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

Two stop words are supported:

- `reach`: stop when the fitness is greater than or equal to a given value. Example: `"reach_40"` stops once the fitness is `>= 40`.
- `saturate`: stop when the fitness does not change for a given number of generations. Example: `"saturate_7"` stops if the fitness stays the same for 7 generations in a row.

Added in [PyGAD 2.15.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-15-0).
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

You can also pass your own crossover function (since [PyGAD 2.16.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-16-0)). See [User-Defined Crossover, Mutation, and Parent Selection Operators](https://pygad.readthedocs.io/en/latest/user_defined_operators.html#user-defined-crossover-mutation-and-parent-selection-operators).

If `crossover_type=None`, the crossover step is skipped and no offspring are created, so the next generation reuses the current population (since [PyGAD 2.2.2](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-2-2)).
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

You can also pass your own mutation function (since [PyGAD 2.16.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-16-0)). See [User-Defined Crossover, Mutation, and Parent Selection Operators](https://pygad.readthedocs.io/en/latest/user_defined_operators.html#user-defined-crossover-mutation-and-parent-selection-operators).

If `mutation_type=None`, the mutation step is skipped and the offspring are used unchanged (since [PyGAD 2.2.2](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-2-2)).
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
   1. `utils.engine.GAEngine`: 
2. `utils/validation.py`
   1. `utils.validation.Validation`
3. `utils/parent_selection.py`
   1. `utils.parent_selection.ParentSelection`
4. `utils/crossover.py`
   1. `utils.crossover.Crossover`
5. `utils/mutation.py`
   1. `utils.mutation.Mutation`
6. `utils/nsga2.py`
   1. `utils.nsga2.NSGA2`
7. `utils/nsga3.py`
   1. `utils.nsga3.NSGA3`
8. `helper/unique.py`
   1. `helper.unique.Unique` 
9. `helper/misc.py`
   1. `helper.misc.Helper`
10. `visualize/plot.py`
    1. `visualize.plot.Plot` 

Since the `pygad.GA` class extends such classes, the attributes and methods inside them can be retrieved by instances of the `pygad.GA` class.

### Class Attributes

* `supported_int_types`: A list of the supported types for the integer numbers.
* `supported_float_types`: A list of the supported types for the floating-point numbers.
* `supported_int_float_types`: A list of the supported types for all numbers. It just concatenates the previous 2 lists.

### Other Instance Attributes & Methods

All the parameters and functions passed to the `pygad.GA` class constructor are used as class attributes and methods in the instances of the `pygad.GA` class. In addition to such attributes, there are other attributes and methods added to the instances of the `pygad.GA` class:

The next 2 subsections list such attributes and methods.

> The `GA` class gains the attributes of its parent classes via inheritance, making them accessible through the `GA` object even if they are defined externally to its specific class body.

#### Other Attributes

- `generations_completed`:  Holds the number of the last completed generation.
- `population`: A NumPy array that initially holds the initial population and is later updated after each generation.
- `valid_parameters`: Set to `True` when all the parameters passed in the `GA` class constructor are valid.
- `run_completed`: Set to `True` only after the `run()` method completes gracefully.
- `pop_size`: The population size.
- `best_solutions_fitness`: A list holding the fitness values of the best solutions for all generations.
- `best_solution_generation`: The generation number at which the best fitness value is reached. It is only assigned the generation number after the `run()` method completes. Otherwise, its value is -1.
- `best_solutions`: A NumPy array holding the best solution per each generation. It only exists when the `save_best_solutions` parameter in the `pygad.GA` class constructor is set to `True`.
- `last_generation_fitness`: The fitness values of the solutions in the last generation. [Added in PyGAD 2.12.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-12-0).
- `previous_generation_fitness`: At the end of each generation, the fitness of the most recent population is saved in the `last_generation_fitness` attribute. The fitness of the population exactly preceding this most recent population is saved in the `previous_generation_fitness` attribute. This `previous_generation_fitness` attribute is used to fetch the pre-calculated fitness instead of calling the fitness function for already explored solutions. [Added in PyGAD 2.16.2](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-16-2).
- `last_generation_parents`: The parents selected from the last generation. [Added in PyGAD 2.12.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-12-0).
- `last_generation_offspring_crossover`: The offspring generated after applying the crossover in the last generation. [Added in PyGAD 2.12.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-12-0).
- `last_generation_offspring_mutation`: The offspring generated after applying the mutation in the last generation. [Added in PyGAD 2.12.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-12-0).
- `gene_type_single`: A flag that is set to `True` if the `gene_type` parameter is assigned to a single data type that is applied to all genes. If `gene_type` is assigned a `list`, `tuple`, or `numpy.ndarray`, then the value of `gene_type_single` will be `False`. [Added in PyGAD 2.14.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-14-0).
- `last_generation_parents_indices`: This attribute holds the indices of the selected parents in the last generation. Supported in [PyGAD 2.15.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-15-0). 
- `last_generation_elitism`: This attribute holds the elitism of the last generation. It is effective only if the `keep_elitism` parameter has a non-zero value. Supported in [PyGAD 2.18.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-18-0). 
- `last_generation_elitism_indices`: This attribute holds the indices of the elitism of the last generation. It is effective only if the `keep_elitism` parameter has a non-zero value. Supported in [PyGAD 2.19.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-19-0). 
- `logger`: This attribute holds the logger from the `logging` module. Supported in [PyGAD 3.0.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-3-0-0). 
- `gene_space_unpacked`: This is the unpacked version of the `gene_space` parameter. For example, `range(1, 5)` is unpacked to `[1, 2, 3, 4]`. For an infinite range like `{'low': 2, 'high': 4}`, then it is unpacked to a limited number of values (e.g. 100). Supported in [PyGAD 3.1.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-3-1-0). 
- `pareto_fronts`: A new instance attribute named `pareto_fronts` added to the `pygad.GA` instances that holds the pareto fronts when solving a multi-objective problem. Supported in [PyGAD 3.2.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-3-2-0). 

Note that the attributes with names starting with `last_generation_` are updated after each generation.

#### Other Methods

- `cal_pop_fitness()`: A method that calculates the fitness values for all solutions within the population by calling the function passed to the `fitness_func` parameter for each solution.
- `crossover()`: Refers to the method that applies the crossover operator based on the selected type of crossover in the `crossover_type` property.
- `mutation()`: Refers to the method that applies the mutation operator based on the selected type of mutation in the `mutation_type` property.
- `select_parents()`: Refers to a method that selects the parents based on the parent selection type specified in the `parent_selection_type` attribute.
- `adaptive_mutation_population_fitness()`: Returns the average fitness value used in the adaptive mutation to filter the solutions.
- `summary()`: Prints a Keras-like summary of the PyGAD lifecycle. This helps to have an overview of the architecture. Supported in [PyGAD 2.19.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-19-0). Check the [Print Lifecycle Summary](https://pygad.readthedocs.io/en/latest/logging.html#print-lifecycle-summary) section for more details and examples.
- 5 methods with names starting with `run_`. Their purpose is to keep the main loop inside the `run()` method clean. The details inside the loop are moved to 4 individual methods. Generally, any method with a name starting with `run_` is meant to be called by PyGAD from inside the `run()` method. Supported in [PyGAD 3.3.1](https://pygad.readthedocs.io/en/latest/releases.html#pygad-3-3-1).
  1. `run_loop_head()`: The code before the loop starts.
  2. `run_select_parents(call_on_parents=True)`: Select the parents and call the callable `on_parents()` if defined. If `call_on_parents` is `True`, then the callable `on_parents()` is called. It must be `False` when the `run_select_parents()` method is called to update the parents at the end of the `run()` method.
  3. `run_crossover()`: Apply crossover and call the callable `on_crossover()` if defined.
  4. `run_mutation()`: Apply mutation and call the callable `on_mutation()` if defined.
  5. `run_update_population()`: Update the `population` attribute after completing the processes of crossover and mutation.

There are many methods that are not designed for user usage. Some of them are listed above but this is not a comprehensive list. The [release history](https://pygad.readthedocs.io/en/latest/releases.html) section usually covers them. Moreover, you can check the [PyGAD GitHub repository](https://github.com/ahmedfgad/GeneticAlgorithmPython) to find more.

The next sections discuss the methods available in the `pygad.GA` class.

### `save()`

The `save()` method in the `pygad.GA` class saves the genetic algorithm instance as a pickled object.

Accepts the following parameter:

* `filename`: Name of the file to save the instance. No extension is needed.

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
