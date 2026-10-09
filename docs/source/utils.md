# `pygad.utils` Module

This section of the documentation discusses the **pygad.utils** module.

PyGAD supports different types of operators for selecting the parents, applying the crossover, and mutation. More features will be added in the future. To ask for a new feature, please check the [Ask for Feature](https://pygad.readthedocs.io/en/latest/help_support.html#ask-for-feature) section.

The submodules in the `pygad.utils` module are:

1. `engine`: Has the `GAEngine` class implementing the main loop and related functions.
2. `parallel`: Has the `FitnessEvaluation` class implementing serial, thread, and process fitness dispatch and worker lifecycle management.
3. `validation`: Has the `Validation` class validating constructor parameters and selecting operators.
4. `crossover`: Has the `Crossover` class implementing crossover operators.
5. `mutation`: Has the `Mutation` class implementing mutation operators.
6. `parent_selection`: Has the `ParentSelection` class implementing parent selection operators.
7. `nsga`: Has the `NSGA` class implementing shared non-dominated sorting operations.
8. `nsga2`: Has the `NSGA2` class implementing the Non-Dominated Sorting Genetic Algorithm II (NSGA-II).
9. `nsga3`: Has the `NSGA3` class implementing the Non-Dominated Sorting Genetic Algorithm III (NSGA-III).
10. `report`: Has the `Report` class generating PDF reports.
11. `quality_indicators`: Has functions measuring the quality of a Pareto front: `hypervolume`, `inverted_generational_distance`, `generational_distance`, and `spacing`.

The `pygad.GA` class inherits the classes listed above, including `FitnessEvaluation` through `GAEngine`. Their methods are accessible through a GA instance. The functions in `quality_indicators` are standalone functions called through that module.

The next sections discuss each submodule.

## `pygad.utils.engine` Submodule

The `pygad.utils.engine` module has the `GAEngine` class that implements the engine of the library. It inherits fitness dispatch and serialization methods from {ref}`FitnessEvaluation <fitness-evaluation>`. The main methods defined in `GAEngine` are:

1. `initialize_population()`
2. `cal_pop_fitness()`
3. `run()`
   1. `run_loop_head()`
   2. `run_select_parents()`
   3. `run_crossover()`
   4. `run_mutation()`
   5. `run_update_population()`
4. `best_solution()`
5. `round_genes()`

### `initialize_population()`

It creates an initial population randomly as a NumPy array. The array is saved in the instance attribute named `population`.

Accepts the following parameters:

- `low`: The lower value of the random range from which the gene values in the initial population are selected. It defaults to -4. Available in PyGAD 1.0.20 and higher.
- `high`: The upper value of the random range from which the gene values in the initial population are selected. It defaults to +4. Available in PyGAD 1.0.20 and higher.

This method assigns the values of the following 3 instance attributes:

1. `pop_size`: Size of the population.
2. `population`: Initially, it holds the initial population and is later updated after each generation.
3. `initial_population`: Holds the initial population.

### `cal_pop_fitness()`

`cal_pop_fitness()` accepts no arguments and returns a NumPy array in current-population order: shape `(sol_per_pop,)` for single-objective fitness, or `(sol_per_pop, num_objectives)` for multi-objective fitness. It returns the values without assigning them to `last_generation_fitness`; `run()` performs that assignment.

For each solution, it checks the following sources in order:

1. `solutions` and `solutions_fitness`, when `save_solutions=True`.
2. `best_solutions` and `best_solutions_fitness`, when `save_best_solutions=True`.
3. Retained elites, when `keep_elitism > 0`. `last_generation_elitism_indices` maps each elite back to its value in `previous_generation_fitness`.
4. Retained parents, when `keep_parents != 0`. `last_generation_parents_indices` maps each parent back to its value in `previous_generation_fitness`.
5. The fitness function, for solutions with no cached value.

These cache rules apply in serial, thread, and process modes. Only uncached rows are passed to `_evaluate_fitness()`, which respects `fitness_batch_size`, validates returned values, and increments `num_fitness_evaluations` by the number of evaluated solutions. Cached rows contribute zero to that counter. These rules assume that a solution's fitness can be reused; see {ref}`non-deterministic problems <non-deterministic-fitness>` for settings that disable reuse.

During `run()`, evaluations reuse the run's worker pool. Outside `run()`, parallel evaluation creates and closes a temporary pool. If all fitness values are cached, no pool is created. Fitness-function exceptions and invalid return values propagate to the caller after being logged. See the {ref}`fitness dispatch reference <fitness-evaluation>` for return-value validation.

### `run()`

Runs the genetic algorithm. This is the main method in which the genetic algorithm is evolved through some generations. It accepts no parameters as it uses the instance to access all of its requirements.

The initial population and each updated population are evaluated using `cal_pop_fitness()`, which reuses cached values and evaluates remaining solutions individually or in batches. Adaptive mutation also evaluates offspring before mutating them.

Each call resets `num_fitness_evaluations` after `on_start` and captures `run_start_time` before initial fitness evaluation. A worker pool is created only when parallel work is needed, shared with adaptive evaluation, and shut down in a `finally` block on completion, early stopping, or an exception. A later `run()` call creates a new pool as needed.

According to the fitness values of all solutions, the parents are selected using the `select_parents()` method. This method's behavior is determined by the parent selection type in the `parent_selection_type` parameter in the `pygad.GA` class constructor.

Based on the selected parents, offspring are generated by applying the crossover and mutation operations using the `crossover()` and `mutation()` methods. The behavior of such 2 methods is defined according to the `crossover_type` and `mutation_type` parameters in the `pygad.GA` class constructor.

After the generation completes, the following takes place:

- The `population` attribute is updated by the new population.
- The `generations_completed` attribute is assigned the number of the last completed generation.
- If there is a callback function assigned to the `on_generation` attribute, then it will be called.

After the `run()` method completes, the following takes place:

- The `best_solution_generation` is assigned the generation number at which the best fitness value is reached.
- The `run_completed` attribute is set to `True`.

Note that the `run()` method is calling 5 different methods during the loop:

1. `run_loop_head()`
2. `run_select_parents()`
3. `run_crossover()`
4. `run_mutation()`
5. `run_update_population()`

(current-population-best-solution)=
### `best_solution()`

Returns information about the best solution in the **current population**. Single-objective problems use the maximum fitness; multi-objective problems use the first solution in the NSGA-II ordering (non-dominated front, then crowding distance). This method does not search all saved generations.

It accepts the following parameters:

* `pop_fitness=None`: Optional `list`, `tuple`, or NumPy array of fitness values matching the current population's length and row order. If `None`, `cal_pop_fitness()` is called, potentially performing additional evaluations. Passing an incompatible type or length raises `ValueError`.

It returns the following:

* `best_solution`: Best solution in the current population.

* `best_solution_fitness`: Fitness value of the best solution.

* `best_match_idx`: Index of the best solution in the current population.

### `round_genes()`

A method to round the genes in the passed solutions. It loops through each gene across all the passed solutions and rounds their values if applicable. 

(fitness-evaluation)=
## `pygad.utils.parallel` Submodule

This module contains the `FitnessEvaluation` mixin and the process-worker function `_process_fitness_chunk()`. A mixin is a class that provides methods for another class to inherit. `GAEngine` inherits this mixin, so `pygad.GA` inherits its methods indirectly. Configure evaluation through `fitness_func`, `fitness_batch_size`, and `parallel_processing`; see the {ref}`parallel processing guide <parallel-processing-guide>` for examples and performance tradeoffs.

The methods and worker attributes below are internal implementation details, documented for completeness. Their signatures and lifetime are not a stable public API.

### `FitnessEvaluation` Methods

#### `__getstate__()`

Returns a shallow copy of the instance's attribute dictionary, excluding `_fitness_executor`, `_fitness_executor_config`, and `_fitness_run_active`. Cloudpickle uses this state for `GA.save()` and for GA snapshots sent to process workers. Live locks and worker handles are therefore not included in a checkpoint. Other instance attributes must still be serializable. Loading the saved GA preserves its optimization state; the next run creates workers as needed.

#### `_shutdown_fitness_executor()`

Detaches the current run's executor, sets `_fitness_executor` and `_fitness_executor_config` to `None`, and calls `executor.shutdown(wait=True)` when an executor exists. It returns `None` and can be called when no pool exists. Shutdown waits for submitted work to finish.

#### `_fitness_pool()`

A context manager yielding a `concurrent.futures.ThreadPoolExecutor` or `ProcessPoolExecutor`, selected by the normalized `parallel_processing` value `(mode, max_workers)`. A worker count of `None` uses the executor's default.

Outside a run, the pool exists only within the context and is closed when the context exits. During a run, it is stored on the GA and reused by later evaluations. If the normalized mode or worker count changes, the previous pool is shut down and a new one is created when parallel evaluation next needs it. This helper expects parallel processing to be enabled.

#### `_map_fitness(tasks)`

Accepts a list of `(solution, index_argument)` pairs and yields fitness-function results in task order. A solution is either a single chromosome or a batch, as prepared by `_evaluate_fitness()`. An empty task list performs no evaluations and creates no pool.

Serial evaluation calls `fitness_func(self, solution, index_argument)` directly and closes any existing run pool when parallelism has been disabled. Thread tasks share the parent GA instance. Process evaluation serializes `(fitness_func, self)` with cloudpickle once per evaluation round, groups tasks to reduce repeated state transfers, and calls `_process_fitness_chunk()` in the workers. Each process task receives a separate snapshot; worker changes do not update the parent GA or other tasks. Updates to the fitness function or custom GA attributes before the next round are included in its new snapshot.

Internal process grouping preserves the scalar fitness signature. Only `fitness_batch_size` changes the function's input to a batch. Fitness-function and serialization exceptions propagate while results are consumed.

(evaluate-selected-fitness)=
#### `_evaluate_fitness(population, indices, adaptive=False)`

Parameters:

- `population`: A two-dimensional NumPy array of chromosomes to evaluate. For adaptive mutation, this is the temporary population containing retained solutions and actual offspring.
- `indices`: A list of row indices to evaluate, in the desired result order. An empty list returns `[]` without creating a pool.
- `adaptive=False`: With `False`, the fitness function receives each row's index, or a list of row indices for a batch. With `True`, it receives `None` in both scalar and batch modes because the offspring do not yet have indices in the current GA population.

Returns a list containing one fitness value per requested row, in `indices` order. Each value is a supported numeric scalar or a `list`, `tuple`, or NumPy array of objective values. With `fitness_batch_size=None` or `1`, calls are scalar. Larger batch sizes group only the requested rows; the final batch can be smaller. A batch call must return a `list`, `tuple`, or NumPy array containing one fitness value per solution.

For each returned task result, `num_fitness_evaluations` increases by the number of solutions in that task before return-value validation. This counter measures solutions, not function calls, and is not a count of every task submitted to a worker. A failed fitness call has no returned result to count.

Batch return types outside `list`, `tuple`, and NumPy array raise `TypeError`; a batch length mismatch or unsupported individual fitness type raises `ValueError`. Fitness-function exceptions propagate. The result generator is closed even if validation fails, ensuring temporary pools are cleaned up. Run-owned pools are cleaned up by `run()`.

Serial scalar calls receive a row view of the passed population. Parallel scalar calls receive row copies, and batch calls use NumPy indexing to create copies in every mode. Fitness functions should treat the supplied solutions as inputs and avoid mutating them.

### Process-Worker Function

`_process_fitness_chunk(payload, tasks)` is a module-level internal worker entry point. `payload` is cloudpickle-serialized bytes containing `(fitness_func, ga_instance)`; `tasks` is a list of `(solution, index_argument)` pairs. It deserializes a fresh function and GA snapshot for each task, calls the fitness function, and returns a list of results in task order. Fitness-function and deserialization exceptions propagate through the executor. This function performs neither caching nor fitness validation.

### Runtime Instance Attributes

These attributes belong to the GA instance through `FitnessEvaluation`; they are omitted from serialized state and may be absent before first use or immediately after loading a checkpoint.

| Attribute | Value and lifetime |
| --- | --- |
| `_fitness_run_active` | Set to `True` when `run()` starts and to `False` in its `finally` block. Selects whether pools are run-owned or temporary. |
| `_fitness_executor` | The lazily created executor for the current run, or `None` after shutdown. Temporary executors used outside `run()` are not stored here. |
| `_fitness_executor_config` | Tuple `(mode, max_workers)` for the stored executor, or `None` after shutdown. Used to detect configuration changes between evaluation rounds. |

`FitnessEvaluation` also uses the existing GA attributes `fitness_func`, `fitness_batch_size`, `parallel_processing`, `supported_int_float_types`, and `num_fitness_evaluations`. The counter is reset by `run()`; direct evaluations outside a run increment its current value.

## `pygad.utils.validation` Submodule

The `pygad.utils.validation` module has the `Validation` class that validates the arguments passed while instantiating the `pygad.GA` class. The methods in this class are:

1. `validate_parameters()`: A method that accepts the same list of arguments accepted by the constructor of the `pygad.GA` class. It validates all the parameters. If everything is validated, the instance attribute `valid_parameters` will be set to `True`. Otherwise, it will be `False` and an exception is raised indicating the invalid criteria.

An inner method called `validate_multi_stop_criteria()` exists to validate the `stop_criteria` argument.

## `pygad.utils.crossover` Submodule

The `pygad.utils.crossover` module has a class named `Crossover` with the supported crossover operations:

1. Single point: Implemented using the `single_point_crossover()` method.
2. Two points: Implemented using the `two_points_crossover()` method.
3. Uniform: Implemented using the `uniform_crossover()` method.
4. Scattered: Implemented using the `scattered_crossover()` method.
5. Simulated binary: Implemented using the `sbx_crossover()` method.

Crossover takes two parents and builds a child by mixing their genes. The next figure shows how single-point, two-point, and uniform crossover do this.

:::{figure} images/crossover_types.*
:alt: Single-point, two-point, and uniform crossover
:width: 560px
:align: center

How single-point, two-point, and uniform crossover build a child from two parents.
:::

All crossover methods accept these parameters:

1. `parents`: The parents to mate for producing the offspring.
2. `offspring_size`: The size of the offspring to produce.

### Crossover Methods

The `Crossover` class in the `pygad.utils.crossover` module supports several methods for applying crossover between the selected parents. All of these methods accept the same parameters which are:

* `parents`: The parents to mate for producing the offspring.
* `offspring_size`: The size of the offspring to produce.

All of such methods return an array of the produced offspring.

The next subsections list the supported methods for crossover.

#### `single_point_crossover()`

Applies the single-point crossover. It selects a point randomly at which crossover takes place between the pairs of parents.

(two-points-crossover)=
#### `two_points_crossover()`

Applies the 2 points crossover. It selects the 2 points randomly at which crossover takes place between the pairs of parents.

The two distinct cut points are selected from `0` through `num_genes`, including both ends. Every pair is equally likely, and the segment copied from the second parent can contain between one and all genes. With a single gene, that gene is copied from the second parent.

The corrected two-point crossover, swap mutation, SBX crossover, and permutation mutation fallback use different random draws from earlier versions. Runs remain reproducible with the same `random_seed` within the same version and environment, but their results can differ from earlier versions.

#### `uniform_crossover()`

Applies the uniform crossover. For each gene, a parent out of the 2 mating parents is selected randomly and the gene is copied from it.

#### `scattered_crossover()`

Applies the scattered crossover. It randomly selects the gene from one of the 2 parents. 

(sbx-crossover)=
#### `sbx_crossover()`

Applies simulated binary crossover for numeric genes. The `sbx_crossover_eta` parameter controls the spread: larger values keep children closer to their parents. Bounds come from `init_range_low` and `init_range_high`, which can specify a separate range for each gene.

For each crossed gene, SBX produces two possible values on opposite sides of the parents' midpoint. PyGAD selects either value with probability `0.5`, avoiding the downward bias from always selecting the lower child. Equal parent values are copied unchanged.

## `pygad.utils.mutation` Submodule

The `pygad.utils.mutation` module has a class named `Mutation` with the supported mutation operations:

1. Random: Implemented using the `random_mutation()` method.
2. Swap: Implemented using the `swap_mutation()` method.
3. Inversion: Implemented using the `inversion_mutation()` method.
4. Scramble: Implemented using the `scramble_mutation()` method.
5. Adaptive: Implemented using the `adaptive_mutation()` method.
6. Polynomial: Implemented using the `polynomial_mutation()` method.

Mutation makes small random changes to the offspring so the search can explore new values. The next figure shows random mutation, where a few genes are picked at random and their values are changed.

:::{figure} images/mutation.*
:alt: Random mutation changes a few genes
:width: 560px
:align: center

Random mutation changes the values of a few genes that are picked at random.
:::

All mutation methods accept this parameter:

1. `offspring`: The offspring to mutate.

(mutation-methods)=
### Mutation Methods

The `Mutation` class in the `pygad.utils.mutation` module supports several methods for applying mutation. All of these methods accept the same parameter which is:

* `offspring`: The offspring to mutate.

All of such methods return an array of the mutated offspring.

The next subsections list the supported methods for mutation.

#### `random_mutation()`

Applies the random mutation which changes the values of some genes randomly. The number of genes is specified according to either the `mutation_num_genes` or the `mutation_percent_genes` attributes.

When `allow_duplicate_genes=False` and every value in a gene's space is already used, a replacement cannot be selected. Random mutation then tries a compatible swap instead. This allows permutation encodings, such as TSP tours, to change even when there are no unused values.

Both swapped values must keep their numeric values after conversion to their destination gene types and rounding, and must belong to both destination gene spaces. Genes with a `gene_constraint` are excluded from swaps, and any constraints on other genes must remain satisfied. If no compatible partner exists, the gene stays unchanged.

Each gene participates in at most one fallback swap per offspring per mutation pass, so selecting both genes of a two-gene permutation does not undo the swap. A fallback swap changes two positions; `mutation_num_genes` or `mutation_probability` selects the genes that can initiate mutation, rather than guaranteeing the number of changed positions. A swap partner can be outside the selected mutation indices.

For each gene, a random value is selected according to the range specified by the 2 attributes `random_mutation_min_val` and `random_mutation_max_val`. The random value is added to the selected gene.

(swap-mutation)=
#### `swap_mutation()`

Applies the swap mutation which interchanges the values of 2 randomly selected genes.

Any pair of distinct positions can be selected. An offspring with only one gene is returned unchanged because there is no second gene to swap.

#### `inversion_mutation()`

Applies the inversion mutation which selects a subset of genes and inverts them.

(scramble-mutation)=
#### `scramble_mutation()`

Applies the scramble mutation which selects a subset of genes and shuffles their order randomly.

The selected contiguous segment has `num_genes // 2` genes. Its values are shuffled directly, preserving the segment's values and leaving genes outside it unchanged. A shuffle can produce the original order; segments with fewer than two genes stay unchanged. The offspring array is modified in place and returned. The simplified implementation can produce every permutation of the selected segment and changes the random draws compared with earlier versions.

#### `adaptive_mutation()`

Applies the adaptive mutation, which selects the number/percentage of genes to mutate based on the solution's fitness. If the fitness is high (the solution quality is high), then a smaller number/percentage of genes is mutated compared to a solution with low fitness.

The count-based and probability-based adaptive mutation methods use the same compatible-swap fallback for permutations as random mutation. Their fitness-based controls select which genes can initiate a mutation; swapped partners are not mutated again in the same pass.

(polynomial-mutation)=
#### `polynomial_mutation(offspring)`

Applies polynomial mutation to the passed two-dimensional offspring array in place and returns it. Each gene is selected with `mutation_probability`, or with probability `1 / num_genes` when that parameter is `None`. `polynomial_mutation_eta` controls the size of the change; higher values favor smaller changes. Bounds come from `init_range_low` and `init_range_high` for each gene, and mutated values are clipped to those bounds. Genes whose range has effectively zero width are skipped. When `allow_duplicate_genes=False`, the existing random duplicate-resolution helper is applied after changing a gene.

### Mutation Helper Methods

The `pygad.utils.mutation` module has some helper methods to assist applying the mutation operation:

1. `mutation_by_space()`: Applies the mutation using the `gene_space` parameter.
2. `mutation_probs_by_space()`: Uses the mutation probabilities in the `mutation_probabilities` instance attribute to apply the mutation using the `gene_space` parameter. For each gene, if its probability is <= the mutation probability, then it will be mutated based on the gene space.
3. `mutation_process_gene_value()`: Generate/select values for the gene that satisfy the constraint. The values could be generated randomly or from the gene space. 
4. `mutation_randomly()`: Applies the random mutation.
5. `mutation_probs_randomly()`: Uses the mutation probabilities in the `mutation_probabilities` instance attribute to apply the random mutation. For each gene, if its probability is <= the mutation probability, then it will be mutated randomly.
6. `adaptive_mutation_population_fitness(offspring)`: Calculate average population fitness and offspring fitness before applying adaptive mutation. See the detailed reference below.
7. `adaptive_mutation_by_space()`: Applies the adaptive mutation based on the `gene_space` parameter. A number of genes are selected randomly for mutation. This number depends on the fitness of the solution. The random values are selected from the `gene_space` parameter.
8. `adaptive_mutation_probs_by_space()`: Uses the mutation probabilities to decide which genes to apply the adaptive mutation by space.
9. `adaptive_mutation_randomly()`: Applies the adaptive mutation randomly. A number of genes are selected randomly for mutation. This number depends on the fitness of the solution. The random values are selected based on the 2 parameters `random_mutation_min_val` and `random_mutation_max_val`.
10. `adaptive_mutation_probs_randomly()`: Uses the mutation probabilities to decide which genes to apply the adaptive mutation randomly.
11. `swap_gene_by_space(solution, gene_idx, swapped_genes=None)`: Swap one gene with a compatible partner while preserving gene types, numeric values, gene spaces, uniqueness, and constraints. The solution is modified in place. The optional `swapped_genes` set tracks both positions already swapped in the same offspring's mutation pass; start with a new set for each pass.

(adaptive-offspring-fitness)=
#### `adaptive_mutation_population_fitness(offspring)`

Accepts a two-dimensional NumPy array of offspring before mutation, with one chromosome per row. It builds a temporary population containing retained solutions followed by these actual offspring, without replacing `self.population`. The number of offspring must match the available rows after retention, as prepared by PyGAD's crossover step.

Retention follows the GA configuration: positive `keep_elitism` selects the best elites; otherwise `keep_parents=-1` retains all selected parents, a positive `keep_parents` selects that many best solutions, and `0` retains none. Retained fitness is read from `last_generation_fitness` using the corresponding original population indices. Only offspring are evaluated through `_evaluate_fitness(..., adaptive=True)`.

Returns `(average_fitness, offspring_fitness)`. For a single objective, the average is a scalar and offspring fitness is a one-dimensional NumPy array. For multiple objectives, the average is a vector and offspring fitness has one row per offspring and one column per objective. The average includes both retained solutions and offspring. NumPy infers a common fitness dtype, preserving fractional offspring values when earlier population fitness was integer-valued.

All execution modes and batch sizes evaluate the same offspring values. The fitness function receives `None` as its index argument in both scalar and batch calls. It must use the supplied chromosomes rather than indexing the current `ga_instance.population`. Every evaluated offspring contributes to `num_fitness_evaluations`; retained fitness does not. During a run, these calls reuse the same executor as ordinary population evaluation. Return validation and propagated exceptions are described in {ref}`_evaluate_fitness() <evaluate-selected-fitness>`.

#### `swap_gene_by_space(solution, gene_idx, swapped_genes=None)`

This helper provides a permutation-preserving fallback for random and adaptive mutation when unique replacement values are unavailable in `gene_space`. `solution` is a chromosome modified in place, and `gene_idx` is the position initiating the swap. The optional `swapped_genes` set records both positions after a successful swap; reuse it within one offspring's mutation pass and start a fresh set for each new pass.

The partner is selected randomly from compatible, not-yet-swapped genes. Both values must remain numerically unchanged after conversion and rounding for their destination gene types, and must belong to the destination gene spaces. Genes with their own constraint are excluded from swapping, and constraints on other genes are checked against the complete proposed chromosome. If no compatible partner exists, the solution is left unchanged. The method returns the solution in either case. Invalid constraint output raises an exception through the existing constraint validator.

## `pygad.utils.parent_selection` Submodule

The `pygad.utils.parent_selection` module has a class named `ParentSelection` with the supported parent selection operations:

1. Steady-state: Implemented using the `steady_state_selection()` method.
2. Roulette wheel: Implemented using the `roulette_wheel_selection()` method.
3. Stochastic universal: Implemented using the `stochastic_universal_selection()` method.
4. Rank: Implemented using the `rank_selection()` method.
5. Random: Implemented using the `random_selection()` method.
6. Tournament: Implemented using the `tournament_selection()` method.
7. NSGA-II: Implemented using the `nsga2_selection()` method.
8. NSGA-II Tournament: Implemented using the `tournament_selection_nsga2()` method.
9. NSGA-III: Implemented using the `nsga3_selection()` method.
10. NSGA-III Tournament: Implemented using the `tournament_selection_nsga3()` method.

All parent selection methods accept these parameters:

1. `fitness`: The fitness of the entire population.
2. `num_parents`: The number of parents to select.

It has the following helper methods:

1. `wheel_cumulative_probs()`: A helper function to calculate the wheel probabilities for these 2 methods: 1) `roulette_wheel_selection()` 2) `rank_selection()`

### Parent Selection Methods

The `ParentSelection` class in the `pygad.utils.parent_selection` module has several methods for selecting the parents that will mate to produce the offspring. All of such methods accept the same parameters which are:

* `fitness`: The fitness values of the solutions in the current population.
* `num_parents`: The number of parents to be selected.

All of such methods return an array of the selected parents.

The next subsections list the supported methods for parent selection.

#### `steady_state_selection()`

Selects the parents using the steady-state selection technique.

(rank-selection)=
#### `rank_selection()`

Selects the parents using the rank selection technique.

Solutions are sorted from best to worst. For a population of `N` solutions, the selection weights in that order are `N, N-1, ..., 1`, and the probabilities are those weights divided by their sum. For example, fitness values `[1, 2, 3, 4]` give the corresponding population rows probabilities `[0.1, 0.2, 0.3, 0.4]`. Larger fitness is favored even when the values are negative. For multiple objectives, the order comes from non-dominated sorting and crowding distance. Returned indices refer to the original population rows. Selection remains random and can select the same row more than once; it does not guarantee selecting the best solution every time.

#### `random_selection()`

Selects the parents randomly.

#### `tournament_selection()`

Selects the parents using the tournament selection technique.

#### `roulette_wheel_selection()`

Selects the parents using the roulette wheel selection technique.

(stochastic-universal-selection)=
#### `stochastic_universal_selection()`

Selects the parents using the stochastic universal selection technique.

`stochastic_universal_selection(fitness, num_parents)` spaces its pointers according to the requested `num_parents`, even when this differs from `num_parents_mating`. It returns the selected parent array and its population indices. Sampling can select the same row more than once; with equal fitness, the evenly spaced pointers distribute selections as evenly as possible. For multiple objectives, the fitness values are summed per solution before constructing the wheel.

#### `nsga2_selection()`

Selects the parents for the NSGA-II algorithm to solve multi-objective optimization problems. It selects the parents by ranking them based on non-dominated sorting and crowding distance.

#### `tournament_selection_nsga2()`

Selects the parents for the NSGA-II algorithm to solve multi-objective optimization problems. It selects the parents using the tournament selection technique applied based on non-dominated sorting and crowding distance.

#### `nsga3_selection()`

Selects the parents for the NSGA-III algorithm to solve multi-objective optimization problems. It accepts whole Pareto fronts in order until adding the next front would overflow the requested parent count, then picks the remaining survivors from that critical front using niching against the structured reference points stored on the GA instance. Requires the `nsga3_num_divisions` parameter to be set when constructing the `pygad.GA` instance.

#### `tournament_selection_nsga3()`

Selects the parents for the NSGA-III algorithm to solve multi-objective optimization problems. It selects the parents using the tournament selection technique where the within-front comparison is based on the niche count (instead of the crowding distance used by `tournament_selection_nsga2()`). Requires the `nsga3_num_divisions` parameter to be set.

## `pygad.utils.nsga` Submodule

The `pygad.utils.nsga` module has a class named `NSGA` that holds the building blocks shared by NSGA-II and NSGA-III. The methods inside this class are:

1. `non_dominated_sorting()`: Returns all the Pareto fronts by applying non-dominated sorting over the solutions.
2. `get_non_dominated_set()`: Returns the two sets of non-dominated and dominated solutions from the passed solutions. The Pareto front is the non-dominated set.

## `pygad.utils.nsga2` Submodule

The `pygad.utils.nsga2` module has a class named `NSGA2` that implements the NSGA-II-specific primitives. The methods inside this class are:

1. `crowding_distance()`: Calculates the crowding distance for all solutions in the current Pareto front.
2. `sort_solutions_nsga2()`: Sort the solutions. If the problem is single-objective, the solutions are sorted by their fitness values. If it is multi-objective, non-dominated sorting and crowding distance are applied to sort the solutions.

## `pygad.utils.nsga3` Submodule

The `pygad.utils.nsga3` module has a class named `NSGA3` that implements the NSGA-III algorithm primitives. NSGA-III novel names start with `nsga3_` to make the algorithm surface easy to spot.

1. `nsga3_generate_reference_points()`: Build the structured grid of reference points on the unit simplex using the Das-Dennis (stars-and-bars) method.
2. `nsga3_compute_ideal_point()`: Return the ideal point (column maximum under PyGAD's maximization convention).
3. `nsga3_find_extreme_points()`: For each objective axis, return the solution that best represents the corner of that axis based on the Achievement Scalarizing Function (ASF).
4. `nsga3_compute_intercepts()`: Fit a hyperplane through the M extreme points and return the per-axis intercept point used as the normalization denominator. Falls back to the nadir (worst per objective) when the hyperplane cannot be fitted or when the intercept is degenerate.
5. `nsga3_normalize_fitness()`: Scale each fitness row to the `[0, 1]` range using the ideal point and the intercepts.
6. `nsga3_associate_to_reference_points()`: For every normalized solution, return the nearest reference index and the perpendicular distance to that reference line.
7. `nsga3_niching_select()`: Pick `num_to_select` survivors from the critical front using niche counts and per-niche tie-breaking rules.

The selection methods `nsga3_selection()` and `tournament_selection_nsga3()` live in `pygad.utils.parent_selection`. The engine-time helpers (`_bootstrap_nsga3_reference_points()`, `_nsga3_grow_population()`, `_nsga3_generate_extra_random_solutions()`, `_nsga3_generate_single_random_gene()`) live in `pygad.utils.engine`.

Two module-level constants in `pygad.utils.nsga3` control the numerical safeguards: `NSGA3_ASF_EPSILON` (default `1e-6`) and `NSGA3_INTERCEPT_NEAR_ZERO` (default `1e-12`).

## `pygad.utils.report` Submodule

The `pygad.utils.report` module has a class named `Report` that adds the `generate_report()` method to the `pygad.GA` class. It builds a PDF report of the GA run, bundling the configuration table, a run summary, the best solution, and every applicable plot. The title page shows the PyGAD logo, which ships with the package. Requires the optional dependencies `reportlab` and `matplotlib`:

```
pip install pygad[report]
```

See [`generate_report()`](https://pygad.readthedocs.io/en/latest/pygad.html#generate-report).

:::{python-examples}
example_generate_report.py
:::

(quality-indicators)=
## `pygad.utils.quality_indicators` Submodule

The `pygad.utils.quality_indicators` module has functions to measure the quality of a Pareto front. All functions take fitness values in PyGAD's maximization format. The functions are:

1. `hypervolume(fitness, reference_point)`: Volume of the objective space dominated by the front. The reference point must be worse than every solution on every objective. A larger value is better.
2. `inverted_generational_distance(fitness, reference_front)`: Mean distance from each reference-front point to its nearest approximation point. Reports both convergence and diversity. A smaller value is better.
3. `generational_distance(fitness, reference_front)`: Mean distance from each approximation point to its nearest reference point. Reports convergence only. A smaller value is better.
4. `spacing(fitness)`: Standard deviation of the distance from each solution to its nearest neighbour. A smaller value means the solutions are spread more evenly.

Example:

```python
from pygad.utils.quality_indicators import hypervolume, inverted_generational_distance

# After ga.run()
fitness = ga.last_generation_fitness
reference_point = [-10.0, -10.0]   # worse than every solution
hv = hypervolume(fitness, reference_point)

# If the true Pareto front is known
from pygad.benchmarks.zdt import ZDT1
problem = ZDT1()
true_front = problem.pareto_front(num_points=100)
igd = inverted_generational_distance(fitness, true_front)
```

:::{python-examples}
quality_indicators/example_hypervolume.py
quality_indicators/example_inverted_generational_distance.py
quality_indicators/example_generational_distance.py
quality_indicators/example_spacing.py
:::

## More about the Operators

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Adaptive Mutation
:link: adaptive_mutation
:link-type: doc

Change the mutation rate per solution based on its fitness.
:::

:::{grid-item-card} User-Defined Operators
:link: user_defined_operators
:link-type: doc

Plug in your own crossover, mutation, and parent selection.
:::

::::

:::{toctree}
:hidden:

adaptive_mutation
user_defined_operators
:::
