# Release History

![PYGAD-LOGO](https://user-images.githubusercontent.com/16560492/101267295-c74c0180-375f-11eb-9ad0-f8e37bd796ce.png)

## PyGAD 1.0.17

Release Date: 15 April 2020

1. The **pygad.GA** class accepts a new argument named `fitness_func` which accepts a function to be used for calculating the fitness values for the solutions. This allows the project to be customized to any problem by building the right fitness function.

## PyGAD 1.0.20 

Release Date: 4 May 2020

1. The **pygad.GA** attributes are moved from the class scope to the instance scope.
2. Raising an exception for incorrect values of the passed parameters.
3. Two new parameters are added to the **pygad.GA** class constructor (`init_range_low` and `init_range_high`) allowing the user to customize the range from which the genes values in the initial population are selected. 
4. The code object `__code__` of the passed fitness function is checked to ensure it has the right number of parameters.

## PyGAD 2.0.0 

Release Date: 13 May 2020

1. The fitness function accepts a new argument named `sol_idx` representing the index of the solution within the population.
2. A new parameter to the **pygad.GA** class constructor named `initial_population` is supported to allow the user to use a custom initial population to be used by the genetic algorithm. If not None, then the passed population will be used. If `None`, then the genetic algorithm will create the initial population using the `sol_per_pop` and `num_genes` parameters. 
3. The parameters `sol_per_pop` and `num_genes` are optional and set to `None` by default.
4. A new parameter named `callback_generation` is introduced in the **pygad.GA** class constructor. It accepts a function with a single parameter representing the **pygad.GA** class instance. This function is called after each generation. This helps the user to do post-processing or debugging operations after each generation.

## PyGAD 2.1.0

Release Date: 14 May 2020

1. The `best_solution()` method in the **pygad.GA** class returns a new output representing the index of the best solution within the population. Now, it returns a total of 3 outputs and their order is: best solution, best solution fitness, and best solution index. Here is an example: 
```python
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution :", solution)
print("Fitness value of the best solution :", solution_fitness, "\n")
print("Index of the best solution :", solution_idx, "\n")
```

2. A new attribute named `best_solution_generation` is added to the instances of the **pygad.GA** class. it holds the generation number at which the best solution is reached. It is only assigned the generation number after the `run()` method completes. Otherwise, its value is -1.
Example:
```python
print("Best solution reached after {best_solution_generation} generations.".format(best_solution_generation=ga_instance.best_solution_generation))
```

3. The `best_solution_fitness` attribute is renamed to `best_solutions_fitness` (plural solution).
4. Mutation is applied independently for the genes.

## PyGAD 2.2.1

Release Date: 17 May 2020

1. Adding 2 extra modules (pygad.nn and pygad.gann) for building and training neural networks with the genetic algorithm.

## PyGAD 2.2.2

Release Date: 18 May 2020
1. The initial value of the `generations_completed` attribute of instances from the pygad.GA class is `0` rather than `None`.

2. An optional bool parameter named `mutation_by_replacement` is added to the constructor of the pygad.GA class. It works only when the selected type of mutation is random (`mutation_type="random"`). In this case, setting `mutation_by_replacement=True` means replace the gene by the randomly generated value. If `False`, then it has no effect and random mutation works by adding the random value to the gene. This parameter should be used when the gene falls within a fixed range and its value must not go out of this range. Here are some examples:

  Assume there is a gene with the value 0.5.

  If `mutation_type="random"` and `mutation_by_replacement=False`, then the generated random value (e.g. 0.1) will be added to the gene value. The new gene value is **0.5+0.1=0.6**.  

  If `mutation_type="random"` and `mutation_by_replacement=True`, then the generated random value (e.g. 0.1) will replace the gene value. The new gene value is **0.1**.

3. `None` value could be assigned to the `mutation_type` and `crossover_type` parameters of the pygad.GA class constructor. When `None`, this means the step is bypassed and has no action.

## PyGAD 2.3.0

Release date: 1 June 2020

1. A new module named `pygad.cnn` is supported for building convolutional neural networks.
2. A new module named `pygad.gacnn` is supported for training convolutional neural networks using the genetic algorithm.
3. The `pygad.plot_result()` method has 3 optional parameters named `title`, `xlabel`, and `ylabel` to customize the plot title, x-axis label, and y-axis label, respectively.
4. The `pygad.nn` module supports the softmax activation function.
5. The name of the `pygad.nn.predict_outputs()` function is changed to `pygad.nn.predict()`.
6. The name of the `pygad.nn.train_network()` function is changed to `pygad.nn.train()`.

## PyGAD 2.4.0

Release date: 5 July 2020

1. A new parameter named `delay_after_gen` is added which accepts a non-negative number specifying the time in seconds to wait after a generation completes and before going to the next generation. It defaults to `0.0` which means no delay after the generation.

2. The passed function to the `callback_generation` parameter of the pygad.GA class constructor can terminate the execution of the genetic algorithm if it returns the string `stop`. This causes the `run()` method to stop.

One important use case for that feature is to stop the genetic algorithm when a condition is met before passing though all the generations. The user may assigned a value of 100 to the `num_generations` parameter of the pygad.GA class constructor. Assuming that at generation 50, for example, a condition is met and the user wants to stop the execution before waiting the remaining 50 generations. To do that, just make the function passed to the `callback_generation` parameter to return the string `stop`.

Here is an example of a function to be passed to the `callback_generation` parameter which stops the execution if the fitness value 70 is reached. The value 70 might be the best possible fitness value. After being reached, then there is no need to pass through more generations because no further improvement is possible.

   ```python
   def func_generation(ga_instance):
    if ga_instance.best_solution()[1] >= 70:
        return "stop"
   ```

## PyGAD 2.5.0

Release date: 19 July 2020

1. 2 new optional parameters added to the constructor of the `pygad.GA` class which are `crossover_probability` and `mutation_probability`. 
   While applying the crossover operation, each parent has a random value generated between 0.0 and 1.0. If this random value is less than or equal to the value assigned to the `crossover_probability` parameter, then the parent is selected for the crossover operation.
   For the mutation operation, a random value between 0.0 and 1.0 is generated for each gene in the solution. If this value is less than or equal to the value assigned to the `mutation_probability`, then this gene is selected for mutation.
2. A new optional parameter named `linewidth` is added to the `plot_result()` method to specify the width of the curve in the plot. It defaults to 3.0.
3. Previously, the indices of the genes selected for mutation was randomly generated once for all solutions within the generation. Currently, the genes' indices are randomly generated for each solution in the population. If the population has 4 solutions, the indices are randomly generated 4 times inside the single generation, 1 time for each solution.
4. Previously, the position of the point(s) for the single-point and two-points crossover was(were) randomly selected once for all solutions within the generation. Currently, the position(s) is(are) randomly selected for each solution in the population. If the population has 4 solutions, the position(s) is(are) randomly generated 4 times inside the single generation, 1 time for each solution.
5. A new optional parameter named `gene_space` as added to the `pygad.GA` class constructor. It is used to specify the possible values for each gene in case the user wants to restrict the gene values. It is useful if the gene space is restricted to a certain range or to discrete values. For more information, check the [More about the `gene_space` Parameter](https://pygad.readthedocs.io/en/latest/gene_values.html#more-about-the-gene-space-parameter) section. Thanks to [Prof. Tamer A. Farrag](https://github.com/tfarrag2000) for requesting this useful feature.

## PyGAD 2.6.0 

Release Date: 6 August 2020

1. A bug fix in assigning the value to the `initial_population` parameter.
2. A new parameter named `gene_type` is added to control the gene type. It can be either `int` or `float`. It has an effect only when the parameter `gene_space` is `None`.
3. 7 new parameters that accept callback functions: `on_start`, `on_fitness`, `on_parents`, `on_crossover`, `on_mutation`, `on_generation`, and `on_stop`.

## PyGAD 2.7.0

Release Date: 11 September 2020
1. The `learning_rate` parameter in the `pygad.nn.train()` function defaults to **0.01**.
2. Added support of building neural networks for regression using the new parameter named `problem_type`. It is added as a parameter to both `pygad.nn.train()` and `pygad.nn.predict()` functions. The value of this parameter can be either **classification** or **regression** to define the problem type. It defaults to **classification**.
3. The activation function for a layer can be set to the string `"None"` to refer that there is no activation function at this layer. As a result, the supported values for the activation function are `"sigmoid"`, `"relu"`, `"softmax"`, and `"None"`.

To build a regression network using the `pygad.nn` module, just do the following:
1. Set the `problem_type` parameter in the `pygad.nn.train()` and `pygad.nn.predict()` functions to the string `"regression"`.
2. Set the activation function for the output layer to the string `"None"`. This sets no limits on the range of the outputs as it will be from `-infinity` to `+infinity`. If you are sure that all outputs will be nonnegative values, then use the ReLU function.

Check the documentation of the `pygad.nn` module for an example that builds a neural network for regression. The regression example is also available at [this GitHub project](https://github.com/ahmedfgad/NumPyANN): https://github.com/ahmedfgad/NumPyANN

To build and train a regression network using the `pygad.gann` module, do the following:

1. Set the `problem_type` parameter in the `pygad.nn.train()` and `pygad.nn.predict()` functions to the string `"regression"`.
2. Set the `output_activation` parameter in the constructor of the `pygad.gann.GANN` class to `"None"`.

Check the documentation of the `pygad.gann` module for an example that builds and trains a neural network for regression. The regression example is also available at [this GitHub project](https://github.com/ahmedfgad/NeuralGenetic): https://github.com/ahmedfgad/NeuralGenetic

To build a classification network, either ignore the `problem_type` parameter or set it to `"classification"` (default value). In this case, the activation function of the last layer can be set to any type (e.g. softmax).

## PyGAD 2.7.1

Release Date: 11 September 2020

1. A bug fix when the `problem_type` argument is set to `regression`.

## PyGAD 2.7.2

Release Date: 14 September 2020

1. Bug fix to support building and training regression neural networks with multiple outputs.

## PyGAD 2.8.0

Release Date: 20 September 2020

1. Support of a new module named `kerasga` so that the Keras models can be trained by the genetic algorithm using PyGAD. 

## PyGAD 2.8.1

Release Date: 3 October 2020

1. Bug fix in applying the crossover operation when the `crossover_probability` parameter is used. Thanks to [Eng. Hamada Kassem, Research and Teaching Assistant, Construction Engineering and Management, Faculty of Engineering, Alexandria University, Egypt](https://www.linkedin.com/in/hamadakassem).

## PyGAD 2.9.0 

Release Date: 06 December 2020

1. The fitness values of the initial population are considered in the `best_solutions_fitness` attribute.
2. An optional parameter named `save_best_solutions` is added. It defaults to `False`. When it is `True`, then the best solution after each generation is saved into an attribute named `best_solutions`. If `False`, then no solutions are saved and the `best_solutions` attribute will be empty.
3. Scattered crossover is supported. To use it, assign the `crossover_type` parameter the value `"scattered"`.
4. NumPy arrays are now supported by the `gene_space` parameter.
5. The following parameters (`gene_type`, `crossover_probability`, `mutation_probability`, `delay_after_gen`) can be assigned to a numeric value of any of these data types: `int`, `float`, `numpy.int`, `numpy.int8`, `numpy.int16`, `numpy.int32`, `numpy.int64`, `numpy.float`, `numpy.float16`, `numpy.float32`, or `numpy.float64`. 

## PyGAD 2.10.0

Release Date: 03 January 2021

1. Support of a new module `pygad.torchga` to train PyTorch models using PyGAD. Check [its documentation](https://pygad.readthedocs.io/en/latest/torchga.html).
2. Support of adaptive mutation where the mutation rate is determined by the fitness value of each solution. Read the [Adaptive Mutation](https://pygad.readthedocs.io/en/latest/adaptive_mutation.html#adaptive-mutation) section for more details. Also, read this paper: [Libelli, S. Marsili, and P. Alba. "Adaptive mutation in genetic algorithms." Soft computing 4.2 (2000): 76-80.](https://www.researchgate.net/publication/225642916_Adaptive_mutation_in_genetic_algorithms)
3. Before the `run()` method completes or exits, the fitness value of the best solution in the current population is appended to the `best_solution_fitness` list attribute. Note that the fitness value of the best solution in the initial population is already saved at the beginning of the list. So, the fitness value of the best solution is saved before the genetic algorithm starts and after it ends.
4. When the parameter `parent_selection_type` is set to `sss` (steady-state selection), then a warning message is printed if the value of the `keep_parents` parameter is set to 0.
5. More validations to the user input parameters.
6. The default value of the `mutation_percent_genes` is set to the string `"default"` rather than the integer 10. This change helps to know whether the user explicitly passed a value to the `mutation_percent_genes` parameter or it is left to its default one. The `"default"` value is later translated into the integer 10. 
7. The `mutation_percent_genes` parameter is no longer accepting the value 0. It must be `>0` and `<=100`.
8. The built-in `warnings` module is used to show warning messages rather than just using the `print()` function.
9. A new `bool` parameter called `suppress_warnings` is added to the constructor of the `pygad.GA` class. It allows the user to control whether the warning messages are printed or not. It defaults to `False` which means the messages are printed.
10. A helper method called `adaptive_mutation_population_fitness()` is created to calculate the average fitness value used in adaptive mutation to filter the solutions.
11. The `best_solution()` method accepts a new optional parameter called `pop_fitness`. It accepts a list of the fitness values of the solutions in the population. If `None`, then the `cal_pop_fitness()` method is called to calculate the fitness values of the population.

## PyGAD 2.10.1

Release Date: 10 January 2021

1. In the `gene_space` parameter, any `None` value (regardless of its index or axis), is replaced by a randomly generated number based on the 3 parameters `init_range_low`, `init_range_high`, and `gene_type`. So, the `None` value in `[..., None, ...]` or `[..., [..., None, ...], ...]` are replaced with random values. This gives more freedom in building the space of values for the genes.
2. All the numbers passed to the `gene_space` parameter are casted to the type specified in the `gene_type` parameter.
3. The `numpy.uint` data type is supported for the parameters that accept integer values.
4. In the `pygad.kerasga` module, the `model_weights_as_vector()` function uses the `trainable` attribute of the model's layers to only return the trainable weights in the network. So, only the trainable layers with their `trainable` attribute set to `True` (`trainable=True`), which is the default value, have their weights evolved. All non-trainable layers with the `trainable` attribute set to `False` (`trainable=False`) will not be evolved. Thanks to [Prof. Tamer A. Farrag](https://github.com/tfarrag2000) for pointing about that at [GitHub](https://github.com/ahmedfgad/KerasGA/issues/1).

## PyGAD 2.10.2

Release Date: 15 January 2021

1. A bug fix when `save_best_solutions=True`. Refer to this issue for more information: https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/25

## PyGAD 2.11.0

Release Date: 16 February 2021

1. In the `gene_space` argument, the user can use a dictionary to specify the lower and upper limits of the gene. This dictionary must have only 2 items with keys `low` and `high` to specify the low and high limits of the gene, respectively. This way, PyGAD takes care of not exceeding the value limits of the gene. For a problem with only 2 genes, then using `gene_space=[{'low': 1, 'high': 5}, {'low': 0.2, 'high': 0.81}]` means the accepted values in the first gene start from 1 (inclusive) to 5 (exclusive) while the second one has values between 0.2 (inclusive) and 0.85 (exclusive). For more information, please check the [Limit the Gene Value Range](https://pygad.readthedocs.io/en/latest/gene_values.html#limit-the-gene-value-range-using-the-gene-space-parameter) section of the documentation.
2. The `plot_result()` method returns the figure so that the user can save it.
3. Bug fixes in copying elements from the gene space.
4. For a gene with a set of discrete values (more than 1 value) in the `gene_space` parameter like `[0, 1]`, it was possible that the gene value may not change after mutation. That is if the current value is 0, then the randomly selected value could also be 0. Now, it is verified that the new value is changed. So, if the current value is 0, then the new value after mutation will not be 0 but 1.

## PyGAD 2.12.0

Release Date: 20 February 2021

1. 4 new instance attributes are added to hold temporary results after each generation: `last_generation_fitness` holds the fitness values of the solutions in the last generation, `last_generation_parents` holds the parents selected from the last generation, `last_generation_offspring_crossover` holds the offspring generated after applying the crossover in the last generation, and `last_generation_offspring_mutation` holds the offspring generated after applying the mutation in the last generation. You can access these attributes inside the `on_generation()` method for example.
2. A bug fixed when the `initial_population` parameter is used. The bug occurred due to a mismatch between the data type of the array assigned to `initial_population` and the gene type in the `gene_type` attribute. Assuming that the array assigned to the `initial_population` parameter is `((1, 1), (3, 3), (5, 5), (7, 7))` which has type `int`. When `gene_type` is set to `float`, then the genes will not be float but casted to `int` because the defined array has `int` type. The bug is fixed by forcing the array assigned to `initial_population` to have the data type in the `gene_type` attribute. Check the [issue at GitHub](https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/27): https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/27

Thanks to Andrei Rozanski [PhD Bioinformatics Specialist, Department of Tissue Dynamics and Regeneration, Max Planck Institute for Biophysical Chemistry, Germany] for opening my eye to the first change.

Thanks to [Marios Giouvanakis](https://www.researchgate.net/profile/Marios-Giouvanakis), a PhD candidate in Electrical & Computer Engineer, [Aristotle University of Thessaloniki (Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης), Greece](https://www.auth.gr/en), for emailing me about the second issue.

## PyGAD 2.13.0 

Release Date: 12 March 2021

1. A new `bool` parameter called `allow_duplicate_genes` is supported. If `True`, which is the default, then a solution/chromosome may have duplicate gene values. If `False`, then each gene will have a unique value in its solution. Check the [Prevent Duplicates in Gene Values](https://pygad.readthedocs.io/en/latest/gene_values.html#prevent-duplicates-in-gene-values) section for more details.
2. The `last_generation_fitness` is updated at the end of each generation not at the beginning. This keeps the fitness values of the most up-to-date population assigned to the `last_generation_fitness` parameter.

## PyGAD 2.14.0

PyGAD 2.14.0 has an issue that is solved in PyGAD 2.14.1. Please consider using 2.14.1 not 2.14.0.

Release Date: 19 May 2021

1. [Issue #40](https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/40) is solved. Now, the `None` value works with the `crossover_type` and `mutation_type` parameters: https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/40
2. The `gene_type` parameter supports accepting a `list/tuple/numpy.ndarray` of numeric data types for the genes. This helps to control the data type of each individual gene. Previously, the `gene_type` can be assigned only to a single data type that is applied for all genes. For more information, check the [More about the `gene_type` Parameter](https://pygad.readthedocs.io/en/latest/gene_values.html#more-about-the-gene-type-parameter) section. Thanks to [Rainer Engel](https://www.linkedin.com/in/rainer-matthias-engel-5ba47a9) for asking about this feature in [this discussion](https://github.com/ahmedfgad/GeneticAlgorithmPython/discussions/43): https://github.com/ahmedfgad/GeneticAlgorithmPython/discussions/43
3. A new `bool` attribute named `gene_type_single` is added to the `pygad.GA` class. It is `True` when there is a single data type assigned to the `gene_type` parameter. When the `gene_type` parameter is assigned a `list/tuple/numpy.ndarray`, then `gene_type_single` is set to `False`.
4. The `mutation_by_replacement` flag now has no effect if `gene_space` exists except for the genes with `None` values. For example, for `gene_space=[None, [5, 6]]` the `mutation_by_replacement` flag affects only the first gene which has `None` for its value space.
5. When an element has a value of `None` in the `gene_space` parameter (e.g. `gene_space=[None, [5, 6]]`), then its value will be randomly generated for each solution rather than being generate once for all solutions. Previously, the gene with `None` value in `gene_space` is the same across all solutions
6. Some changes in the documentation according to [issue #32](https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/32): https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/32 

## PyGAD 2.14.2

Release Date: 27 May 2021

1. Some bug fixes when the `gene_type` parameter is nested. Thanks to [Rainer Engel](https://www.linkedin.com/in/rainer-matthias-engel-5ba47a9) for opening [a discussion](https://github.com/ahmedfgad/GeneticAlgorithmPython/discussions/43#discussioncomment-763342) to report this bug: https://github.com/ahmedfgad/GeneticAlgorithmPython/discussions/43#discussioncomment-763342

[Rainer Engel](https://www.linkedin.com/in/rainer-matthias-engel-5ba47a9) helped a lot in suggesting new features and suggesting enhancements in 2.14.0 to 2.14.2 releases. 

## PyGAD 2.14.3

Release Date: 6 June 2021

1. Some bug fixes when setting the `save_best_solutions` parameter to `True`. Previously, the best solution for generation `i` was added into the `best_solutions` attribute at generation `i+1`. Now, the `best_solutions` attribute is updated by each best solution at its exact generation.

## PyGAD 2.15.0

Release Date: 17 June 2021

1. Control the precision of all genes/individual genes. Thanks to [Rainer](https://github.com/rengel8) for asking about this feature: https://github.com/ahmedfgad/GeneticAlgorithmPython/discussions/43#discussioncomment-763452
2. A new attribute named `last_generation_parents_indices` holds the indices of the selected parents in the last generation.
3. In adaptive mutation, no need to recalculate the fitness values of the parents selected in the last generation as these values can be returned based on the `last_generation_fitness` and `last_generation_parents_indices` attributes. This speeds-up the adaptive mutation.
4. When a sublist has a value of `None` in the `gene_space` parameter (e.g. `gene_space=[[1, 2, 3], [5, 6, None]]`), then its value will be randomly generated for each solution rather than being generated once for all solutions. Previously, a value of `None` in a sublist of the `gene_space` parameter was identical across all solutions.
5. The dictionary assigned to the `gene_space` parameter itself or one of its elements has a new key called `"step"` to specify the step of moving from the start to the end of the range specified by the 2 existing keys `"low"` and `"high"`.  An example is `{"low": 0, "high": 30, "step": 2}` to have only even values for the gene(s) starting from 0 to 30. For more information, check the [More about the `gene_space` Parameter](https://pygad.readthedocs.io/en/latest/gene_values.html#more-about-the-gene-space-parameter) section. https://github.com/ahmedfgad/GeneticAlgorithmPython/discussions/48
6. A new function called `predict()` is added in both the `pygad.kerasga` and `pygad.torchga` modules to make predictions. This makes it easier than using custom code each time a prediction is to be made.
7. A new parameter called `stop_criteria` allows the user to specify one or more stop criteria to stop the evolution based on some conditions. Each criterion is passed as `str` which has a stop word. The current 2 supported words are `reach` and `saturate`. `reach` stops the `run()` method if the fitness value is equal to or greater than a given fitness value. An example for `reach` is `"reach_40"` which stops the evolution if the fitness is >= 40. `saturate` means stop the evolution if the fitness saturates for a given number of consecutive generations. An example for `saturate` is `"saturate_7"` which means stop the `run()` method if the fitness does not change for 7 consecutive generations. Thanks to [Rainer](https://github.com/rengel8) for asking about this feature: https://github.com/ahmedfgad/GeneticAlgorithmPython/discussions/44
8. A new bool parameter, defaults to `False`, named `save_solutions` is added to the constructor of the `pygad.GA` class. If `True`, then all solutions in each generation are appended into an attribute called `solutions` which is NumPy array.
9. The `plot_result()` method is renamed to `plot_fitness()`. The users should migrate to the new name as the old name will be removed in the future.
10. Four new optional parameters are added to the `plot_fitness()` function in the `pygad.GA` class which are `font_size=14`, `save_dir=None`, `color="#3870FF"`, and `plot_type="plot"`. Use `font_size` to change the font of the plot title and labels. `save_dir` accepts the directory to which the figure is saved. It defaults to `None` which means do not save the figure. `color` changes the color of the plot. `plot_type` changes the plot type which can be either `"plot"` (default), `"scatter"`, or `"bar"`. https://github.com/ahmedfgad/GeneticAlgorithmPython/pull/47
11. The default value of the `title` parameter in the `plot_fitness()` method is `"PyGAD - Generation vs. Fitness"` rather than `"PyGAD - Iteration vs. Fitness"`.
12. A new method named `plot_new_solution_rate()` creates, shows, and returns a figure showing the rate of new/unique solutions explored in each generation. It accepts the same parameters as in the `plot_fitness()` method. This method only works when `save_solutions=True` in the `pygad.GA` class's constructor.
13. A new method named `plot_genes()` creates, shows, and returns a figure to show how each gene changes per each generation. It accepts similar parameters like the `plot_fitness()` method in addition to the `graph_type`, `fill_color`, and `solutions` parameters. The `graph_type` parameter can be either `"plot"` (default), `"boxplot"`, or `"histogram"`. `fill_color` accepts the fill color which works when `graph_type` is either `"boxplot"` or `"histogram"`. `solutions` can be either `"all"` or `"best"` to decide whether all solutions or only best solutions are used.
14. The `gene_type` parameter now supports controlling the precision of `float` data types. For a gene, rather than assigning just the data type like `float`, assign a `list`/`tuple`/`numpy.ndarray` with 2 elements where the first one is the type and the second one is the precision. For example, `[float, 2]` forces a gene with a value like `0.1234` to be `0.12`. For more information, check the [More about the `gene_type` Parameter](https://pygad.readthedocs.io/en/latest/gene_values.html#more-about-the-gene-type-parameter) section.

## PyGAD 2.15.1

Release Date: 18 June 2021

1. Fix a bug when `keep_parents` is set to a positive integer. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/49

## PyGAD 2.15.2

Release Date: 18 June 2021

1. Fix a bug when using the `kerasga` or `torchga` modules. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/51

## PyGAD 2.16.0

Release Date: 19 June 2021

1. A user-defined function can be passed to the `mutation_type`, `crossover_type`, and `parent_selection_type` parameters in the `pygad.GA` class to create a custom mutation, crossover, and parent selection operators. Check the [User-Defined Crossover, Mutation, and Parent Selection Operators](https://pygad.readthedocs.io/en/latest/user_defined_operators.html#user-defined-crossover-mutation-and-parent-selection-operators) section for more details. https://github.com/ahmedfgad/GeneticAlgorithmPython/discussions/50

## PyGAD 2.16.1

Release Date: 28 September 2021

1. The user can use the `tqdm` library to show a progress bar. https://github.com/ahmedfgad/GeneticAlgorithmPython/discussions/50.  

```python
import pygad
import numpy
import tqdm

equation_inputs = [4,-2,3.5]
desired_output = 44

def fitness_func(ga_instance, solution, solution_idx):
    output = numpy.sum(solution * equation_inputs)
    fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
    return fitness

num_generations = 10000
with tqdm.tqdm(total=num_generations) as pbar:
    ga_instance = pygad.GA(num_generations=num_generations,
                           sol_per_pop=5,
                           num_parents_mating=2,
                           num_genes=len(equation_inputs),
                           fitness_func=fitness_func,
                           on_generation=lambda _: pbar.update(1))
    
    ga_instance.run()

ga_instance.plot_result()
```
But this work does not work if the `ga_instance` will be pickled (i.e. the `save()` method will be called. 

```python
ga_instance.save("test")
```

To solve this issue, define a function and pass it to the `on_generation` parameter. In the next code, the `on_generation_progress()` function is defined which updates the progress bar.

```python
import pygad
import numpy
import tqdm

equation_inputs = [4,-2,3.5]
desired_output = 44

def fitness_func(ga_instance, solution, solution_idx):
    output = numpy.sum(solution * equation_inputs)
    fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
    return fitness

def on_generation_progress(ga):
    pbar.update(1)

num_generations = 100
with tqdm.tqdm(total=num_generations) as pbar:
    ga_instance = pygad.GA(num_generations=num_generations,
                           sol_per_pop=5,
                           num_parents_mating=2,
                           num_genes=len(equation_inputs),
                           fitness_func=fitness_func,
                           on_generation=on_generation_progress)

    ga_instance.run()

ga_instance.plot_result()

ga_instance.save("test")
```

2. Solved the issue of unequal length between the `solutions` and `solutions_fitness` when the `save_solutions` parameter is set to `True`. Now, the fitness of the last population is appended to the `solutions_fitness` array. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/64

3. There was an issue of getting the length of these 4 variables (`solutions`, `solutions_fitness`, `best_solutions`, and `best_solutions_fitness`) doubled after each call of the `run()` method. This is solved by resetting these variables at the beginning of the `run()` method. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/62
4. Bug fixes when adaptive mutation is used (`mutation_type="adaptive"`). https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/65

## PyGAD 2.16.2

Release Date: 2 February 2022

1. A new instance attribute called `previous_generation_fitness` added in the `pygad.GA` class. It holds the fitness values of one generation before the fitness values saved in the `last_generation_fitness`.
3. Issue in the `cal_pop_fitness()` method in getting the correct indices of the previous parents. This is solved by using the previous generation's fitness saved in the new attribute `previous_generation_fitness` to return the parents' fitness values. Thanks to Tobias Tischhauser (M.Sc. - [Mitarbeiter Institut EMS, Departement Technik, OST – Ostschweizer Fachhochschule, Switzerland](https://www.ost.ch/de/forschung-und-dienstleistungen/technik/systemtechnik/ems/team)) for detecting this bug.

## PyGAD 2.16.3

Release Date: 2 February 2022

1. Validate the fitness value returned from the fitness function. An exception is raised if something is wrong. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/67

## PyGAD 2.17.0

Release Date: 8 July 2022

1. An issue is solved when the `gene_space` parameter is given a fixed value. e.g. gene_space=[range(5), 4]. The second gene's value is static (4) which causes an exception.
2. Fixed the issue where the `allow_duplicate_genes` parameter did not work when mutation is disabled (i.e. `mutation_type=None`). This is by checking for duplicates after crossover directly. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/39
3. Solve an issue in the `tournament_selection()` method as the indices of the selected parents were incorrect. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/89
4. Reuse the fitness values of the previously explored solutions rather than recalculating them. This feature only works if `save_solutions=True`.
4. Parallel processing is supported. This is by the introduction of a new parameter named `parallel_processing` in the constructor of the `pygad.GA` class. Thanks to [@windowshopr](https://github.com/windowshopr) for opening the issue [#78](https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/78) at GitHub. Check the [Parallel Processing in PyGAD](https://pygad.readthedocs.io/en/latest/fitness_calculation.html#parallel-processing-in-pygad) section for more information and examples.

## PyGAD 2.18.0
Release Date: 9 September 2022

1. Raise an exception if the sum of fitness values is zero while either roulette wheel or stochastic universal parent selection is used. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/129
2. Initialize the value of the `run_completed` property to `False`. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/122
3. The values of these properties are no longer reset with each call to the `run()` method `self.best_solutions, self.best_solutions_fitness, self.solutions, self.solutions_fitness`: https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/123. Now, the user can have the flexibility of calling the `run()` method more than once while extending the data collected after each generation. Another advantage happens when the instance is loaded and the `run()` method is called, as the old fitness value are shown on the graph alongside with the new fitness values. Read more in this section: [Continue without Losing Progress](https://pygad.readthedocs.io/en/latest/generations.html#continue-without-losing-progress)
4. Thanks [Prof. Fernando Jiménez Barrionuevo](http://webs.um.es/fernan) (Dept. of Information and Communications Engineering, University of Murcia, Murcia, Spain) for editing this [comment](https://github.com/ahmedfgad/GeneticAlgorithmPython/blob/5315bbec02777df96ce1ec665c94dece81c440f4/pygad.py#L73) in the code. https://github.com/ahmedfgad/GeneticAlgorithmPython/commit/5315bbec02777df96ce1ec665c94dece81c440f4
5. A bug fixed when `crossover_type=None`.
6. Support of elitism selection through a new parameter named `keep_elitism`. It defaults to 1 which means for each generation keep only the best solution in the next generation. If assigned 0, then it has no effect. Read more in this section: [Elitism Selection](https://pygad.readthedocs.io/en/latest/generations.html#elitism-selection). https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/74
7. A new instance attribute named `last_generation_elitism` added to hold the elitism in the last generation.
8. A new parameter called `random_seed` added to accept a seed for the random function generators. Credit to this issue https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/70 and [Prof. Fernando Jiménez Barrionuevo](http://webs.um.es/fernan). Read more in this section: [Random Seed](https://pygad.readthedocs.io/en/latest/generations.html#random-seed).
9. Editing the `pygad.TorchGA` module to make sure the tensor data is moved from GPU to CPU. Thanks to Rasmus Johansson for opening this pull request: https://github.com/ahmedfgad/TorchGA/pull/2

## PyGAD 2.18.1

Release Date: 19 September 2022

1. A big fix when `keep_elitism` is used. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/132

## PyGAD 2.18.2
Release Date: 14 February 2023

1. Remove `numpy.int` and `numpy.float` from the list of supported data types. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/151 https://github.com/ahmedfgad/GeneticAlgorithmPython/pull/152
2. Call the `on_crossover()` callback function even if `crossover_type` is `None`. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/138
3. Call the `on_mutation()` callback function even if `mutation_type` is `None`. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/138

## PyGAD 2.18.3

Release Date: 14 February 2023

1. Bug fixes.

## PyGAD 2.19.0

Release Date: 22 February 2023
1. A new `summary()` method is supported to return a Keras-like summary of the PyGAD lifecycle.
2. A new optional parameter called `fitness_batch_size` is supported to calculate the fitness in batches. If it is assigned the value `1` or `None` (default), then the normal flow is used where the fitness function is called for each individual solution. If the `fitness_batch_size` parameter is assigned a value satisfying this condition `1 < fitness_batch_size <= sol_per_pop`, then the solutions are grouped into batches of size `fitness_batch_size` and the fitness function is called once for each batch. In this case, the fitness function must return a list/tuple/numpy.ndarray with a length equal to the number of solutions passed. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/136.
3. The `cloudpickle` library (https://github.com/cloudpipe/cloudpickle) is used instead of the `pickle` library to pickle the `pygad.GA` objects. This solves the issue of having to redefine the functions (e.g. fitness function). The `cloudpickle` library is added as a dependency in the `requirements.txt` file. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/159
4. Support of assigning methods to these parameters: `fitness_func`, `crossover_type`, `mutation_type`, `parent_selection_type`, `on_start`, `on_fitness`, `on_parents`, `on_crossover`, `on_mutation`, `on_generation`, and `on_stop`. https://github.com/ahmedfgad/GeneticAlgorithmPython/pull/92 https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/138
5. Validating the output of the parent selection, crossover, and mutation functions.
6. The built-in parent selection operators return the parent's indices as a NumPy array.
7. The outputs of the parent selection, crossover, and mutation operators must be NumPy arrays.
8. Fix an issue when `allow_duplicate_genes=True`. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/39
9. Fix an issue creating scatter plots of the solutions' fitness.
10. Sampling from a `set()` is no longer supported in Python 3.11. Instead, sampling happens from a `list()`. Thanks `Marco Brenna` for pointing to this issue.
11. The lifecycle is updated to reflect that the new population's fitness is calculated at the end of the lifecycle not at the beginning. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/154#issuecomment-1438739483
12. There was an issue when `save_solutions=True` that causes the fitness function to be called for solutions already explored and have their fitness pre-calculated. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/160
13. A new instance attribute named `last_generation_elitism_indices` added to hold the indices of the selected elitism. This attribute helps to re-use the fitness of the elitism instead of calling the fitness function.
14. Fewer calls to the `best_solution()` method which in turns saves some calls to the fitness function.
15. Some updates in the documentation to give more details about the `cal_pop_fitness()` method. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/79#issuecomment-1439605442

## PyGAD 2.19.1

Release Date: 22 February 2023

1. Add the [cloudpickle](https://github.com/cloudpipe/cloudpickle) library as a dependency.

## PyGAD 2.19.2

Release Date 23 February 2023

1. Fix an issue when parallel processing was used where the elitism solutions' fitness values are not re-used. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/160#issuecomment-1441718184

## PyGAD 3.0.0

Release Date 8 April 2023

1. The structure of the library is changed and some methods defined in the `pygad.py` module are moved to the `pygad.utils`, `pygad.helper`, and `pygad.visualize` submodules.
  2. The `pygad.utils.parent_selection` module has a class named `ParentSelection` where all the parent selection operators exist. The `pygad.GA` class extends this class.
  3. The `pygad.utils.crossover` module has a class named `Crossover` where all the crossover operators exist. The `pygad.GA` class extends this class.
  4. The `pygad.utils.mutation` module has a class named `Mutation` where all the mutation operators exist. The `pygad.GA` class extends this class.
  5. The `pygad.helper.unique` module has a class named `Unique` some helper methods exist to solve duplicate genes and make sure every gene is unique. The `pygad.GA` class extends this class.
  6. The `pygad.visualize.plot` module has a class named `Plot` where all the methods that create plots exist. The `pygad.GA` class extends this class.
  7. Support of using the `logging` module to log the outputs to both the console and text file instead of using the `print()` function. This is by assigning the `logging.Logger` to the new `logger` parameter. Check the [Logging Outputs](https://pygad.readthedocs.io/en/latest/logging.html#logging-outputs) for more information.
  8. A new instance attribute called `logger` to save the logger. 
  9. The function/method passed to the `fitness_func` parameter accepts a new parameter that refers to the instance of the `pygad.GA` class. Check this for an example: [Use Functions and Methods to Build Fitness Function and Callbacks](https://pygad.readthedocs.io/en/latest/custom_functions.html#use-functions-methods-and-classes-to-build-fitness-and-callbacks). https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/163
  10. Update the documentation to include an example of using functions and methods to calculate the fitness and build callbacks. Check this for more details: [Use Functions and Methods to Build Fitness Function and Callbacks](https://pygad.readthedocs.io/en/latest/custom_functions.html#use-functions-methods-and-classes-to-build-fitness-and-callbacks). https://github.com/ahmedfgad/GeneticAlgorithmPython/pull/92#issuecomment-1443635003
  11. Validate the value passed to the `initial_population` parameter.
  12. Validate the type and length of the `pop_fitness` parameter of the `best_solution()` method.
  13. Some edits in the documentation. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/106
  14. Fix an issue when building the initial population as (some) genes have their value taken from the mutation range (defined by the parameters `random_mutation_min_val` and `random_mutation_max_val`) instead of using the parameters `init_range_low` and `init_range_high`.
  15. The `summary()` method returns the summary as a single-line string. Just log/print the returned string it to see it properly.
  16. The `callback_generation` parameter is removed. Use the `on_generation` parameter instead.
  17. There was an issue when using the `parallel_processing` parameter with Keras and PyTorch. As Keras/PyTorch are not thread-safe, the `predict()` method gives incorrect and weird results when more than 1 thread is used. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/145 https://github.com/ahmedfgad/TorchGA/issues/5 https://github.com/ahmedfgad/KerasGA/issues/6. Thanks to this [StackOverflow answer](https://stackoverflow.com/a/75606666/5426539).
  18. Replace `numpy.float` by `float` in the 2 parent selection operators roulette wheel and stochastic universal. https://github.com/ahmedfgad/GeneticAlgorithmPython/pull/168

## PyGAD 3.0.1

Release Date 20 April 2023

1. Fix an issue with passing user-defined function/method for parent selection. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/179

## PyGAD 3.1.0

Release Date 20 June 2023

1. Fix a bug when the initial population has duplciate genes if a nested gene space is used.
2. The `gene_space` parameter can no longer be assigned a tuple.
3. Fix a bug when the `gene_space` parameter has a member of type `tuple`.
4. A new instance attribute called `gene_space_unpacked` which has the unpacked `gene_space`. It is used to solve duplicates. For infinite ranges in the `gene_space`, they are unpacked to a limited number of values (e.g. 100).
5. Bug fixes when creating the initial population using `gene_space` attribute.
6. When a `dict` is used with the `gene_space` attribute, the new gene value was calculated by summing 2 values: 1) the value sampled from the `dict` 2) a random value returned from the random mutation range defined by the 2 parameters `random_mutation_min_val` and `random_mutation_max_val`. This might cause the gene value to exceed the range limit defined in the `gene_space`. To respect the `gene_space` range, this release only returns the value from the `dict` without summing it to a random value.
7. Formatting the strings using f-string instead of the `format()` method. https://github.com/ahmedfgad/GeneticAlgorithmPython/pull/189
8. In the `__init__()` of the `pygad.GA` class, the logged error messages are handled using a `try-except` block instead of repeating the `logger.error()` command. https://github.com/ahmedfgad/GeneticAlgorithmPython/pull/189
9. A new class named `CustomLogger` is created in the `pygad.cnn` module to create a default logger using the `logging` module assigned to the `logger` attribute. This class is extended in all other classes in the module. The constructors of these classes have a new parameter named `logger` which defaults to `None`. If no logger is passed, then the default logger in the `CustomLogger` class is used.
10. Except for the `pygad.nn` module, the `print()` function in all other modules are replaced by the `logging` module to log messages.
11. The callback functions/methods `on_fitness()`, `on_parents()`, `on_crossover()`, and `on_mutation()` can return values. These returned values override the corresponding properties. The output of `on_fitness()` overrides the population fitness. The `on_parents()` function/method must return 2 values representing the parents and their indices. The output of `on_crossover()` overrides the crossover offspring. The output of `on_mutation()` overrides the mutation offspring.
12. Fix a bug when adaptive mutation is used while `fitness_batch_size`>1. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/195
13. When `allow_duplicate_genes=False` and a user-defined `gene_space` is used, it sometimes happen that there is no room to solve the duplicates between the 2 genes by simply replacing the value of one gene by another gene. This release tries to solve such duplicates by looking for a third gene that will help in solving the duplicates. Check [this section](https://pygad.readthedocs.io/en/latest/gene_values.html#prevent-duplicates-in-gene-values) for more information.
14. Use probabilities to select parents using the rank parent selection method. https://github.com/ahmedfgad/GeneticAlgorithmPython/discussions/205
15. The 2 parameters `random_mutation_min_val` and `random_mutation_max_val` can accept iterables (list/tuple/numpy.ndarray) with length equal to the number of genes. This enables customizing the mutation range for each individual gene.  https://github.com/ahmedfgad/GeneticAlgorithmPython/discussions/198
16. The 2 parameters `init_range_low` and `init_range_high` can accept iterables (list/tuple/numpy.ndarray) with length equal to the number of genes. This enables customizing the initial range for each individual gene when creating the initial population. 
17. The `data` parameter in the `predict()` function of the `pygad.kerasga` module can be assigned a data generator. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/115 https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/207
18. The `predict()` function of the `pygad.kerasga` module accepts 3 optional parameters: 1) `batch_size=None`, `verbose=0`, and `steps=None`. Check documentation of the [Keras Model.predict()](https://keras.io/api/models/model_training_apis) method for more information. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/207
19. The documentation is updated to explain how mutation works when `gene_space` is used with `int` or `float` data types. Check [this section](https://pygad.readthedocs.io/en/latest/gene_values.html#limit-the-gene-value-range-using-the-gene-space-parameter). https://github.com/ahmedfgad/GeneticAlgorithmPython/discussions/198

## PyGAD 3.2.0

Release Date 7 September 2023

1. A new module `pygad.utils.nsga2` is created that has the `NSGA2` class that includes the functionalities of NSGA-II. The class has these methods: 1) `get_non_dominated_set()` 2) `non_dominated_sorting()` 3) `crowding_distance()` 4) `sort_solutions_nsga2()`. Check [this section](https://pygad.readthedocs.io/en/latest/multi_objective.html#multi-objective-optimization) for an example. 
2. Support of multi-objective optimization using Non-Dominated Sorting Genetic Algorithm II (NSGA-II) using the `NSGA2` class in the `pygad.utils.nsga2` module. Just return a `list`, `tuple`, or `numpy.ndarray` from the fitness function and the library will consider the problem as multi-objective optimization. All the objectives are expected to be maximization. Check [this section](https://pygad.readthedocs.io/en/latest/multi_objective.html#multi-objective-optimization) for an example. 
3. The parent selection methods and adaptive mutation are edited to support multi-objective optimization.
4. Two new NSGA-II parent selection methods are supported in the `pygad.utils.parent_selection` module: 1) Tournament selection for NSGA-II 2) NSGA-II selection.
5. The `plot_fitness()` method in the `pygad.plot` module has a new optional parameter named `label` to accept the label of the plots. This is only used for multi-objective problems. Otherwise, it is ignored. It defaults to `None` and accepts a `list`, `tuple`, or `numpy.ndarray`. The labels are used in a legend inside the plot.
6. The default color in the methods of the `pygad.plot` module is changed to the greenish `#64f20c` color.
7. A new instance attribute named `pareto_fronts` added to the `pygad.GA` instances that holds the pareto fronts when solving a multi-objective problem. 
8. The `gene_type` accepts a `list`, `tuple`, or `numpy.ndarray` for integer data types given that the precision is set to `None` (e.g. `gene_type=[float, [int, None]]`).
9. In the `cal_pop_fitness()` method, the fitness value is re-used if `save_best_solutions=True` and the solution is found in the `best_solutions` attribute. These parameters also can help re-using the fitness of a solution instead of calling the fitness function: `keep_elitism`, `keep_parents`, and `save_solutions`.
10. The value `99999999999` is replaced by `float('inf')` in the 2 methods `wheel_cumulative_probs()` and `stochastic_universal_selection()` inside the `pygad.utils.parent_selection.ParentSelection` class.
11. The `plot_result()` method in the `pygad.visualize.plot.Plot` class is removed. Instead, please use the `plot_fitness()` if you did not upgrade yet.

## PyGAD 3.3.0

Release Date 29 January 2024

1. Solve bugs when multi-objective optimization is used. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/238
2. When the `stop_ciiteria` parameter is used with the `reach` keyword, then multiple numeric values can be passed when solving a multi-objective problem. For example, if a problem has 3 objective functions, then `stop_criteria="reach_10_20_30"` means the GA stops if the fitness of the 3 objectives are at least 10, 20, and 30, respectively. The number values must match the number of objective functions. If a single value found (e.g. `stop_criteria=reach_5`) when solving a multi-objective problem, then it is used across all the objectives. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/238
3. The `delay_after_gen` parameter is now deprecated and will be removed in a future release. If it is necessary to have a time delay after each generation, then assign a callback function/method to the `on_generation` parameter to pause the evolution.
4. Parallel processing now supports calculating the fitness during adaptive mutation. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/201
5. The population size can be changed during runtime by changing all the parameters that would affect the size of any thing used by the GA. For more information, check the [Change Population Size during Runtime](https://pygad.readthedocs.io/en/latest/generations.html#change-population-size-during-runtime) section. https://github.com/ahmedfgad/GeneticAlgorithmPython/discussions/234
6. When a dictionary exists in the `gene_space` parameter without a step, then mutation occurs by adding a random value to the gene value. The random vaue is generated based on the 2 parameters `random_mutation_min_val` and `random_mutation_max_val`. For more information, check the [How Mutation Works with the gene_space Parameter?](https://pygad.readthedocs.io/en/latest/gene_values.html#how-mutation-works-with-the-gene-space-parameter) section. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/229
7. Add `object` as a supported data type for int (GA.supported_int_types) and float (GA.supported_float_types). https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/174
8. Use the `raise` clause instead of the `sys.exit(-1)` to terminate the execution. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/213
9. Fix a bug when multi-objective optimization is used with batch fitness calculation (e.g. `fitness_batch_size` set to a non-zero number).
10. Fix a bug in the `pygad.py` script when finding the index of the best solution. It does not work properly with multi-objective optimization where `self.best_solutions_fitness` have multiple columns.

   ```python
               self.best_solution_generation = numpy.where(numpy.array(
                   self.best_solutions_fitness) == numpy.max(numpy.array(self.best_solutions_fitness)))[0][0]
   ```

## PyGAD 3.3.1

Release Date 17 February 2024

1. After the last generation and before the `run()` method completes, update the 2 instance attributes: 1) `last_generation_parents` 2) `last_generation_parents_indices`. This is to keep the list of parents up-to-date with the latest population fitness `last_generation_fitness`.  https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/275
2. 5 methods with names starting with `run_`. Their purpose is to keep the main loop inside the `run()` method clean. Check the [Other Methods](https://pygad.readthedocs.io/en/latest/pygad.html#other-methods) section for more information.
   1. `run_loop_head()`: The code before the loop starts.
   2. `run_select_parents()`: The parent selection-related code.
   3. `run_crossover()`: The crossover-related code.
   4. `run_mutation()`: The mutation-related code.
   5. `run_update_population()`: Update the `population` instance attribute after completing the processes of crossover and mutation.


## PyGAD 3.4.0

Release Date 07 January 2025

1. The `delay_after_gen` parameter is removed from the `pygad.GA` class constructor. As a result, it is no longer an attribute of the `pygad.GA` class instances. To add a delay after each generation, apply it inside the `on_generation` callback. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/283
2. In the `single_point_crossover()` method of the `pygad.utils.crossover.Crossover` class, all the random crossover points are returned before the `for` loop. This is by calling the `numpy.random.randint()` function only once before the loop to generate all the K points (where K is the offspring size). This is compared to calling the `numpy.random.randint()` function inside the `for` loop K times, once for each individual offspring.
3. Bug fix in the `examples/example_custom_operators.py` script. https://github.com/ahmedfgad/GeneticAlgorithmPython/pull/285
4. While making prediction using the `pygad.torchga.predict()` function, no gradients are calculated.
5. The `gene_type` parameter of the `pygad.helper.unique.Unique.unique_int_gene_from_range()` method accepts the type of the current gene only instead of the full gene_type list.
6. Created a new method called `unique_float_gene_from_range()` inside the `pygad.helper.unique.Unique` class to find a unique floating-point number from a range.
7. Fix a bug in the `pygad.helper.unique.Unique.unique_gene_by_space()` method to return the numeric value only instead of a NumPy array.
8. Refactoring the `pygad/helper/unique.py` script to remove duplicate codes and reformatting the docstrings.
9. The `plot_pareto_front_curve()` method added to the pygad.visualize.plot.Plot class to visualize the Pareto front for multi-objective problems. It only supports 2 objectives. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/279
11. Fix a bug converting a nested NumPy array to a nested list. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/300
12. The `Matplotlib` library is only imported when a method inside the `pygad/visualize/plot.py` script is used. This is more efficient than using `import matplotlib.pyplot` at the module level as this causes it to be imported when `pygad` is imported even when it is not needed. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/292
13. Fix a bug when minus sign (-) is used inside the `stop_criteria` parameter (e.g. `stop_criteria=["saturate_10", "reach_-0.5"]`). https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/296
14. Make sure `self.best_solutions` is a list of lists inside the `cal_pop_fitness` method. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/293
15. Fix a bug where the `cal_pop_fitness()` method was using the `previous_generation_fitness` attribute to return the parents fitness. This instance attribute was not using the fitness of the latest population, instead the fitness of the population before the last one. The issue is solved by updating the `previous_generation_fitness` attribute to the latest population fitness before the GA completes. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/291

## PyGAD 3.5.0

Release Date 08 July 2025

1. Fix a bug when minus sign (-) is used inside the `stop_criteria` parameter for multi-objective problems. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/314 https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/323
2. Fix a bug when the `stop_criteria` parameter is passed as an iterable (e.g. list) for multi-objective problems (e.g. `['reach_50_60', 'reach_20, 40']`). https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/314
3. Call the `get_matplotlib()` function from the `plot_genes()` method inside the `pygad.visualize.plot.Plot` class to import the matplotlib library. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/315
4. Create a new helper method called `select_unique_value()` inside the `pygad/helper/unique.py` script to select a unique gene from an array of values.
5. Create a new helper method called `get_random_mutation_range()` inside the `pygad/utils/mutation.py` script that returns the random mutation range (min and max) for a single gene by its index.
6. Create a new helper method called `change_random_mutation_value_dtype` inside the `pygad/utils/mutation.py` script that changes the data type of the value used to apply random mutation.
7. Create a new helper method called  `round_random_mutation_value()` inside the `pygad/utils/mutation.py` script that rounds the value used to apply random mutation. 
8. Create the `pygad/helper/misc.py` script with a class called `Helper` that has the following helper methods:
   1. `change_population_dtype_and_round()`: For each gene in the population, round the gene value and change the data type.
   2. `change_gene_dtype_and_round()`: Round the change the data type of a single gene.
   3. `mutation_change_gene_dtype_and_round()`: Decides whether mutation is done by replacement or not. Then it rounds and change the data type of the new gene value.
   4. `validate_gene_constraint_callable_output()`: Validates the output of the user-defined callable/function that checks whether the gene constraint defined in the `gene_constraint` parameter is satisfied or not.
   5. `get_gene_dtype()`: Returns the gene data type from the `gene_type` instance attribute.
   6. `get_random_mutation_range()`: Returns the random mutation range using the `random_mutation_min_val` and `random_mutation_min_val` instance attributes.
   7. `get_initial_population_range()`: Returns the initial population values range using the `init_range_low` and `init_range_high` instance attributes.
   8. `generate_gene_value_from_space()`: Generates/selects a value for a gene using the `gene_space` instance attribute.
   9. `generate_gene_value_randomly()`: Generates a random value for the gene. Only used if `gene_space` is `None`.
   10. `generate_gene_value()`: Generates a value for the gene. It checks whether `gene_space` is `None` and calls either `generate_gene_value_randomly()` or `generate_gene_value_from_space()`.
   11. `filter_gene_values_by_constraint()`: Receives a list of values for a gene. Then it filters such values using the gene constraint.
   12. `get_valid_gene_constraint_values()`: Selects one valid gene value that satisfy the gene constraint. It simply calls `generate_gene_value()` to generate some gene values then it filters such values using `filter_gene_values_by_constraint()`.
9. Create a new helper method called `mutation_process_random_value()` inside the `pygad/utils/mutation.py` script that generates constrained random values for mutation. It calls either `generate_gene_value()` or `get_valid_gene_constraint_values()` based on whether the `gene_constraint` parameter is used or not.
10. A new parameter called `gene_constraint` is added. It accepts a list of callables (i.e. functions) acting as constraints for the gene values. Before selecting a value for a gene, the callable is called to ensure the candidate value is valid. Check the [Gene Constraint](https://pygad.readthedocs.io/en/latest/gene_values.html#gene-constraint) section for more information. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/119
11. A new parameter called `sample_size` is added. To select a gene value that respects a constraint, this variable defines the size of the sample from which a value is selected randomly. Useful if either `allow_duplicate_genes` or `gene_constraint` is used. An instance attribute of the same name is created in the instances of the `pygad.GA` class. Check the [sample_size Parameter](https://pygad.readthedocs.io/en/latest/gene_values.html#sample-size-parameter) section for more information.
12. Use the `sample_size` parameter instead of `num_trials` in the methods `solve_duplicate_genes_randomly()` and `unique_float_gene_from_range()` inside the `pygad/helper/unique.py` script. It is the maximum number of values to generate as the search space when looking for a unique float value out of a range.
13. Fixed a bug in population initialization when `allow_duplicate_genes=False`. Previously, gene values were checked for duplicates before rounding, which could allow near-duplicates like 7.61 and 7.62 to pass. After rounding (e.g., both becoming 7.6), this resulted in unintended duplicates. The fix ensures gene values are now rounded before duplicate checks, preventing such cases.
14. More tests are created.
15. More examples are created.
16. Edited the `sort_solutions_nsga2()` method in the `pygad/utils/nsga2.py` script to accept an optional parameter called `find_best_solution` when calling this method just to find the best solution.
17. Fixed a bug while applying the non-dominated sorting in the `get_non_dominated_set()` method inside the `pygad/utils/nsga2.py` script. It was swapping the non-dominated and dominated sets. In other words, it used the non-dominated set as if it is the dominated set and vice versa. All the calls to this method were edited accordingly. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/320.
18. Fix a bug retrieving in the `best_solution()` method when retrieving the best solution for multi-objective problems. https://github.com/ahmedfgad/GeneticAlgorithmPython/pull/331

## PyGAD 3.6.0

Release Date April 8, 2026

1. Support passing a class to the fitness, crossover, and mutation. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/342
2. A new class called `Validation` is created in the new `pygad/utils/validation.py` script. It has a method called `validate_parameters()` to validate all the parameters passed while instantiating the `pygad.GA` class.
3. Refactoring the `pygad.py` script by moving a lot of functions and methods to other classes in other scripts. 
  4. The `summary()` method was moved to `Helper` class in the `pygad/helper/misc.py` script.
  5. The validation code in the `__init__()` method of the `pygad.GA` class is moved to the new `validate_parameters()` method in the new `Validation` class in the new `pygad/utils/validation.py` script. Moreover, the `validate_multi_stop_criteria()` method is also moved to the same class.
  6. The GA main workflow is moved into the new `GAEngine` class in the new `pygad/utils/engine.py` script. Specifically, these methods are moved from the `pygad.GA` class to the new `GAEngine` class:
          1. `run()`
               1. `run_loop_head()`
               2. `run_select_parents()`
               3. `run_crossover()`
               4. `run_mutation()`
               5. `run_update_population()`
          2. `initialize_population()`
          3. `cal_pop_fitness()`
          4. `best_solution()`
          5. `round_genes()`
7. The `pygad.GA` class now extends the two new classes `utils.validation.Validation` and `utils.engine.GAEngine`.
8. The version of the `pygad.utils` submodule is upgraded from `1.3.0` to `1.4.0`.
9. The version of the `pygad.helper` submodule is upgraded from `1.2.0` to `1.3.0`.
10. The version of the `pygad.visualize` submodule is upgraded from `1.1.0` to `1.1.1`.
11. The version of the `pygad.nn` submodule is upgraded from `1.2.1` to `1.2.2`.
12. The version of the `pygad.cnn` submodule is upgraded from `1.1.0` to `1.1.1`.
13. The version of the `pygad.torchga` submodule is upgraded from `1.4.0` to `1.4.1`.
14. The version of the `pygad.kerasga` submodule is upgraded from `1.3.0` to `1.3.1`.
15. Update the elitism after the evolution ends to fix issue where the best solution returned by the `best_solution()` method is not correct. https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/337
16. Fix a bug in calling the `numpy.reshape()` function. The parameter `newshape` is removed since it is no longer supported started from NumPy `2.4.0`. https://numpy.org/doc/stable/release/2.4.0-notes.html#removed-newshape-parameter-from-numpy-reshape
17. A minor change in the documentation is made to replace the `newshape` parameter when calling `numpy.reshape()`.
18. Fix a bug in the `visualize/plot.py` script that causes a warning to be given when the plot leged is used with single-objective problems.
19. A new method called `initialize_parents_array()` is added to the `Helper` class in the `pygad/helper/misc.py` script. It is usually called from the methods in the `ParentSelection` class in the `pygad/utils/parent_selection.py` script to initialize the parents array.
20. Add more tests about: 
          1. Operators (crossover, mutation, and parent selection).
                2. The `best_solution()` method.
                3. Parallel processing.
                      4. The `GANN` module.
                      5. The plots created by the `visualize`.

21. Instead of using repeated code for converting the data type and rounding the genes during crossover and mutation, the `change_gene_dtype_and_round()` method is called from the `pygad.helper.misc.Helper` class.
22. Fix some documentation issues. https://github.com/ahmedfgad/GeneticAlgorithmPython/pull/336
23. Update the documentation to reflect the recent additions and changes to the library structure.
