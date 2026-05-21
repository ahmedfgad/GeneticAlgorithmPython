# `pygad` Module

This section of the documentation discusses the `pygad` module.

With the `pygad` module, you can create, run, save, and load instances of the genetic algorithm. It solves both single-objective and multi-objective optimization problems.

## `pygad.GA` Class

The `pygad` module has a class named `GA` for building the genetic algorithm. This section explains the class constructor, its methods, functions, and attributes.

### `__init__()`

To create an instance of the `pygad.GA` class, the constructor accepts several parameters. These let you adjust the genetic algorithm for different types of applications.

The `pygad.GA` class constructor supports the following parameters:

- `num_generations`: Number of generations.
- `num_parents_mating`: Number of solutions to be selected as parents.
- `fitness_func`: Accepts a function/method and returns the fitness value(s) of the solution. If a function is passed, then it must accept 3 parameters (1. the instance of the `pygad.GA` class, 2. a single solution, and 3. its index in the population). If method, then it accepts a fourth parameter representing the method's class instance. Check the [Preparing the fitness_func Parameter](https://pygad.readthedocs.io/en/latest/steps_to_use.html#preparing-the-fitness-func-parameter) section for information about creating such a function. In [PyGAD 3.2.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-3-2-0), multi-objective optimization is supported. To consider the problem as multi-objective, just return a `list`, `tuple`, or `numpy.ndarray` from the fitness function.
- `fitness_batch_size=None`: A new optional parameter called `fitness_batch_size` is supported to calculate the fitness function in batches. If it is assigned the value `1` or `None` (default), then the normal flow is used where the fitness function is called for each individual solution. If the `fitness_batch_size` parameter is assigned a value satisfying this condition `1 < fitness_batch_size <= sol_per_pop`, then the solutions are grouped into batches of size `fitness_batch_size` and the fitness function is called once for each batch. Check the [Batch Fitness Calculation](https://pygad.readthedocs.io/en/latest/fitness_calculation.html#batch-fitness-calculation) section for more details and examples. Added in from [PyGAD 2.19.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-19-0).
- `initial_population`: A user-defined initial population. It is useful when the user wants to start the generations with a custom initial population. It defaults to `None` which means no initial population is specified by the user. In this case, [PyGAD](https://pypi.org/project/pygad) creates an initial population using the `sol_per_pop` and `num_genes` parameters. An exception is raised if the `initial_population` is `None` while any of the 2 parameters (`sol_per_pop` or `num_genes`) is also `None`. Introduced in [PyGAD 2.0.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-0-0) and higher.
- `sol_per_pop`: Number of solutions (i.e. chromosomes) within the population. This parameter has no action if `initial_population` parameter exists.
- `num_genes`: Number of genes in the solution/chromosome. This parameter is not needed if the user feeds the initial population to the `initial_population` parameter.
- `gene_type=float`: Controls the gene type. It can be assigned to a single data type that is applied to all genes or can specify the data type of each individual gene. It defaults to `float` which means all genes are of `float` data type. Starting from [PyGAD 2.9.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-9-0), the `gene_type` parameter can be assigned to a numeric value of any of these types: `int`, `float`, and `numpy.int/uint/float(8-64)`. Starting from [PyGAD 2.14.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-14-0), it can be assigned to a `list`, `tuple`, or a `numpy.ndarray` which hold a data type for each gene (e.g. `gene_type=[int, float, numpy.int8]`).  This helps to control the data type of each individual gene. In [PyGAD 2.15.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-15-0), a precision for the `float` data types can be specified (e.g. `gene_type=[float, 2]`.
- `init_range_low=-4`: The lower value of the random range from which the gene values in the initial population are selected. `init_range_low` defaults to `-4`. Available in [PyGAD 1.0.20](https://pygad.readthedocs.io/en/latest/releases.html#pygad-1-0-20) and higher. This parameter has no action if the `initial_population` parameter exists.
- `init_range_high=4`: The upper value of the random range from which the gene values in the initial population are selected. `init_range_high` defaults to `+4`. Available in [PyGAD 1.0.20](https://pygad.readthedocs.io/en/latest/releases.html#pygad-1-0-20) and higher. This parameter has no action if the `initial_population` parameter exists.
- `parent_selection_type="sss"`: The parent selection type. Supported types are `sss` (for steady-state selection), `rws` (for roulette wheel selection), `sus` (for stochastic universal selection), `rank` (for rank selection), `random` (for random selection), and `tournament` (for tournament selection). A custom parent selection function can be passed starting from [PyGAD 2.16.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-16-0). Check the [User-Defined Crossover, Mutation, and Parent Selection Operators](https://pygad.readthedocs.io/en/latest/user_defined_operators.html#user-defined-crossover-mutation-and-parent-selection-operators) section for more details about building a user-defined parent selection function.
- `keep_parents=-1`: The number of parents to keep in the next population. `-1` (default) means keep all the parents. `0` means keep no parents. A value greater than `0` means keep that number of parents. The value of `keep_parents` cannot be less than `-1` or greater than the number of solutions in the population (`sol_per_pop`). Starting from [PyGAD 2.18.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-18-0), this parameter has an effect only when the `keep_elitism` parameter is `0`. Starting from PyGAD 2.20.0, the parents' fitness from the last generation is not re-used if `keep_parents=0`. To see how `keep_parents` and `keep_elitism` work together, check the [How the Number of Offspring Is Decided](https://pygad.readthedocs.io/en/latest/generations.html#how-the-number-of-offspring-is-decided) section.
- `keep_elitism=1`: Added in [PyGAD 2.18.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-18-0). It takes the value `0` or a positive integer that meets the condition `0 <= keep_elitism <= sol_per_pop`. It defaults to `1`, which means only the best solution in the current generation is kept in the next generation. If set to `0`, it has no effect. If set to a positive integer `K`, then the best `K` solutions are kept in the next generation. It cannot be greater than the value of the `sol_per_pop` parameter. If this parameter is not `0`, then the `keep_parents` parameter has no effect. To see how `keep_elitism` and `keep_parents` work together, check the [How the Number of Offspring Is Decided](https://pygad.readthedocs.io/en/latest/generations.html#how-the-number-of-offspring-is-decided) section.
- `K_tournament=3`: In case that the parent selection type is `tournament`, the `K_tournament` specifies the number of parents participating in the tournament selection. It defaults to `3`.
- `crossover_type="single_point"`: Type of the crossover operation. Supported types are `single_point` (for single-point crossover), `two_points` (for two points crossover), `uniform` (for uniform crossover), and `scattered` (for scattered crossover). Scattered crossover is supported from PyGAD [2.9.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-9-0) and higher. It defaults to `single_point`. A custom crossover function can be passed starting from [PyGAD 2.16.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-16-0). Check the [User-Defined Crossover, Mutation, and Parent Selection Operators](https://pygad.readthedocs.io/en/latest/user_defined_operators.html#user-defined-crossover-mutation-and-parent-selection-operators) section for more details about creating a user-defined crossover function. Starting from [PyGAD 2.2.2](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-2-2) and higher, if  `crossover_type=None`, then the crossover step is bypassed which means no crossover is applied and thus no offspring will be created in the next generations. The next generation will use the solutions in the current population.
- `crossover_probability=None`: The probability of selecting a parent for applying the crossover operation. Its value must be between 0.0 and 1.0 inclusive. For each parent, a random value between 0.0 and 1.0 is generated. If this random value is less than or equal to the value assigned to the `crossover_probability` parameter, then the parent is selected. Added in [PyGAD 2.5.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-5-0) and higher.
- `mutation_type="random"`: Type of the mutation operation. Supported types are `random` (for random mutation), `swap` (for swap mutation), `inversion` (for inversion mutation), `scramble` (for scramble mutation), and `adaptive` (for adaptive mutation). It defaults to `random`. A custom mutation function can be passed starting from [PyGAD 2.16.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-16-0). Check the [User-Defined Crossover, Mutation, and Parent Selection Operators](https://pygad.readthedocs.io/en/latest/user_defined_operators.html#user-defined-crossover-mutation-and-parent-selection-operators) section for more details about creating a user-defined mutation function. Starting from [PyGAD 2.2.2](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-2-2) and higher, if `mutation_type=None`, then the mutation step is bypassed which means no mutation is applied and thus no changes are applied to the offspring created using the crossover operation. The offspring will be used unchanged in the next generation. `Adaptive` mutation is supported starting from [PyGAD 2.10.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-10-0). For more information about adaptive mutation, go to the [Adaptive Mutation](https://pygad.readthedocs.io/en/latest/adaptive_mutation.html#adaptive-mutation) section. For example about using adaptive mutation, check the [Use Adaptive Mutation in PyGAD](https://pygad.readthedocs.io/en/latest/adaptive_mutation.html#use-adaptive-mutation-in-pygad) section.
- `mutation_probability=None`: The probability of selecting a gene for applying the mutation operation. Its value must be between 0.0 and 1.0 inclusive. For each gene in a solution, a random value between 0.0 and 1.0 is generated. If this random value is less than or equal to the value assigned to the `mutation_probability` parameter, then the gene is selected. If this parameter exists, then there is no need for the 2 parameters `mutation_percent_genes` and `mutation_num_genes`. Added in [PyGAD 2.5.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-5-0) and higher.
- `mutation_by_replacement=False`: An optional bool parameter. It works only when the selected type of mutation is random (`mutation_type="random"`). In this case, `mutation_by_replacement=True` means replace the gene by the randomly generated value. If False, then it has no effect and random mutation works by adding the random value to the gene. Supported in [PyGAD 2.2.2](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-2-2) and higher. Check the changes in [PyGAD 2.2.2](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-2-2) under the Release History section for an example.
- `mutation_percent_genes="default"`: Percentage of genes to mutate. It defaults to the string `"default"` which is later translated into the integer `10` which means 10% of the genes will be mutated. It must be `>0` and `<=100`. Out of this percentage, the number of genes to mutate is deduced which is assigned to the `mutation_num_genes` parameter. The `mutation_percent_genes` parameter has no action if `mutation_probability` or `mutation_num_genes` exist. Starting from [PyGAD 2.2.2](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-2-2) and higher, this parameter has no action if `mutation_type` is `None`.
- `mutation_num_genes=None`: Number of genes to mutate which defaults to `None` meaning that no number is specified. The `mutation_num_genes` parameter has no action if the parameter `mutation_probability` exists. Starting from [PyGAD 2.2.2](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-2-2) and higher, this parameter has no action if `mutation_type` is `None`.
- `random_mutation_min_val=-1.0`: For `random` mutation, the `random_mutation_min_val` parameter specifies the start value of the range from which a random value is selected to be added to the gene. It defaults to `-1`. Starting from [PyGAD 2.2.2](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-2-2) and higher, this parameter has no action if `mutation_type` is `None`.
- `random_mutation_max_val=1.0`: For `random` mutation, the `random_mutation_max_val` parameter specifies the end value of the range from which a random value is selected to be added to the gene. It defaults to `+1`. Starting from [PyGAD 2.2.2](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-2-2) and higher, this parameter has no action if `mutation_type` is `None`.
- `gene_space=None`: It is used to specify the possible values for each gene in case the user wants to restrict the gene values. It is useful if the gene space is restricted to a certain range or to discrete values. It accepts a `list`, `range`, or `numpy.ndarray`. When all genes have the same global space, specify their values as a `list`/`tuple`/`range`/`numpy.ndarray`. For example, `gene_space = [0.3, 5.2, -4, 8]` restricts the gene values to the 4 specified values. If each gene has its own space, then the `gene_space` parameter can be nested like `[[0.4, -5], [0.5, -3.2, 8.2, -9], ...]` where the first sublist determines the values for the first gene, the second sublist for the second gene, and so on. If the nested list/tuple has a `None` value, then the gene's initial value is selected randomly from the range specified by the 2 parameters `init_range_low` and `init_range_high` and its mutation value is selected randomly from the range specified by the 2 parameters `random_mutation_min_val` and `random_mutation_max_val`. `gene_space` is added in [PyGAD 2.5.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-5-0). Check the [Release History of PyGAD 2.5.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-5-0) section of the documentation for more details. In [PyGAD 2.9.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-9-0), NumPy arrays can be assigned to the `gene_space` parameter. In [PyGAD 2.11.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-11-0), the `gene_space` parameter itself or any of its elements can be assigned to a dictionary to specify the lower and upper limits of the genes. For example, `{'low': 2, 'high': 4}` means the minimum and maximum values are 2 and 4, respectively. In [PyGAD 2.15.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-15-0), a new key called `"step"` is supported to specify the step of moving from the start to the end of the range specified by the 2 existing keys `"low"` and `"high"`.  
- `gene_constraint=None`: A list of callables (i.e. functions) acting as constraints for the gene values. Before selecting a value for a gene, the callable is called to ensure the candidate value is valid. Added in [PyGAD 3.5.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-3-5-0). Check the [Gene Constraint](https://pygad.readthedocs.io/en/latest/gene_values.html#gene-constraint) section for more information.
- `sample_size=100`: In some cases where a gene value is to be selected, this variable defines the size of the sample from which a value is selected randomly. Useful if either `allow_duplicate_genes` or `gene_constraint` is used. If PyGAD failed to find a unique value or a value that meets a gene constraint, it is recommended to increases this parameter's value. Added in [PyGAD 3.5.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-3-5-0). Check the [sample_size Parameter](https://pygad.readthedocs.io/en/latest/gene_values.html#sample-size-parameter) section for more information.
- `on_start=None`: Accepts a function/method to be called only once before the genetic algorithm starts its evolution. If function, then it must accept a single parameter representing the instance of the genetic algorithm. If method, then it must accept 2 parameters where the second one refers to the method's object. Added in [PyGAD 2.6.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-6-0).
- `on_fitness=None`: Accepts a function/method to be called after calculating the fitness values of all solutions in the population. If function, then it must accept 2 parameters: 1) a list of all solutions' fitness values 2) the instance of the genetic algorithm. If method, then it must accept 3 parameters where the third one refers to the method's object. Added in [PyGAD 2.6.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-6-0).
- `on_parents=None`: Accepts a function/method to be called after selecting the parents that mates. If function, then it must accept 2 parameters: 1) the selected parents 2) the instance of the genetic algorithm  If method, then it must accept 3 parameters where the third one refers to the method's object. Added in [PyGAD 2.6.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-6-0).
- `on_crossover=None`: Accepts a function to be called each time the crossover operation is applied. This function must accept 2 parameters: the first one represents the instance of the genetic algorithm and the second one represents the offspring generated using crossover. Added in [PyGAD 2.6.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-6-0).
- `on_mutation=None`: Accepts a function to be called each time the mutation operation is applied. This function must accept 2 parameters: the first one represents the instance of the genetic algorithm and the second one represents the offspring after applying the mutation. Added in [PyGAD 2.6.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-6-0).
- `on_generation=None`: Accepts a function to be called after each generation. This function must accept a single parameter representing the instance of the genetic algorithm. If the function returned the string `stop`, then the `run()` method stops without completing the other generations. Added in [PyGAD 2.6.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-6-0).
- `on_stop=None`: Accepts a function to be called only once exactly before the genetic algorithm stops or when it completes all the generations. This function must accept 2 parameters: the first one represents the instance of the genetic algorithm and the second one is a list of fitness values of the last population's solutions. Added in [PyGAD 2.6.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-6-0). 
- `save_best_solutions=False`: When `True`, then the best solution after each generation is saved into an attribute named `best_solutions`. If `False` (default), then no solutions are saved and the `best_solutions` attribute will be empty. Supported in [PyGAD 2.9.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-9-0).
- `save_solutions=False`: If `True`, then all solutions in each generation are appended into an attribute called `solutions` which is NumPy array. Supported in [PyGAD 2.15.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-15-0).
- `suppress_warnings=False`: A bool parameter to control whether the warning messages are printed or not. It defaults to `False`.
- `allow_duplicate_genes=True`: Added in [PyGAD 2.13.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-13-0). If `True`, then a solution/chromosome may have duplicate gene values. If `False`, then each gene will have a unique value in its solution.
- `stop_criteria=None`: Some criteria to stop the evolution. Added in [PyGAD 2.15.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-15-0). Each criterion is passed as `str` which has a stop word. The current 2 supported words are `reach` and `saturate`. `reach` stops the `run()` method if the fitness value is equal to or greater than a given fitness value. An example for `reach` is `"reach_40"` which stops the evolution if the fitness is >= 40. `saturate` means stop the evolution if the fitness saturates for a given number of consecutive generations. An example for `saturate` is `"saturate_7"` which means stop the `run()` method if the fitness does not change for 7 consecutive generations. 
- `parallel_processing=None`: Added in [PyGAD 2.17.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-17-0). If `None` (Default), this means no parallel processing is applied. It can accept a list/tuple of 2 elements [1) Can be either `'process'` or `'thread'` to indicate whether processes or threads are used, respectively., 2) The number of processes or threads to use.]. For example, `parallel_processing=['process', 10]` applies parallel processing with 10 processes. If a positive integer is assigned, then it is used as the number of threads. For example, `parallel_processing=5` uses 5 threads which is equivalent to `parallel_processing=["thread", 5]`. For more information, check the [Parallel Processing in PyGAD](https://pygad.readthedocs.io/en/latest/fitness_calculation.html#parallel-processing-in-pygad) section.
- `random_seed=None`: Added in [PyGAD 2.18.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-18-0). It defines the random seed to be used by the random function generators (we use random functions in the NumPy and random modules). This helps to reproduce the same results by setting the same random seed (e.g. `random_seed=2`). If given the value `None`, then it has no effect. 
- `logger=None`: Accepts an instance of the `logging.Logger` class to log the outputs. Any message is no longer printed using `print()` but logged. If `logger=None`, then a logger is created that uses `StreamHandler` to logs the messages to the console. Added in [PyGAD 3.0.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-3-0-0). Check the [Logging Outputs](https://pygad.readthedocs.io/en/latest/logging.html#logging-outputs) for more information.

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
7. `helper/unique.py`
   1. `helper.unique.Unique` 
8. `helper/misc.py`
   1. `helper.misc.Helper`
9. `visualize/plot.py`
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
