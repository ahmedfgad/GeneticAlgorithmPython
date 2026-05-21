# Controlling Generations

This page covers how PyGAD controls evolution across generations: when to stop, elitism, the random seed, saving and continuing progress, and changing the population size.

## Stop at Any Generation

In [PyGAD 2.4.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-4-0), it is possible to stop the genetic algorithm after any generation. All you need to do is return the string `"stop"` in the `on_generation` callback function. When this callback function is implemented and assigned to the `on_generation` parameter in the constructor of the `pygad.GA` class, the algorithm stops right after it completes its current generation. Here is an example.

Assume the user wants to stop the algorithm either after 100 generations or when a condition is met. The user can assign a value of 100 to the `num_generations` parameter of the `pygad.GA` class constructor.

The condition that stops the algorithm is written in a callback function like the one in the next code. If the fitness value of the best solution exceeds 70, then the string `"stop"` is returned.

```python
def func_generation(ga_instance):
    if ga_instance.best_solution()[1] >= 70:
        return "stop"
```

## Stop Criteria

In [PyGAD 2.15.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-15-0), a new parameter named `stop_criteria` is added to the constructor of the `pygad.GA` class. It helps to stop the evolution based on some criteria. It can be assigned one or more criteria.

Each criterion is passed as `str` that consists of 2 parts:

1.  Stop word.
2.  Number. 

It takes this form:

```python
"word_num"
```

The current 2 supported words are `reach` and `saturate`. 

The `reach` word stops the `run()` method if the fitness value is equal to or greater than a given fitness value. An example for `reach` is `"reach_40"` which stops the evolution if the fitness is >= 40.

`saturate` stops the evolution if the fitness saturates for a given number of consecutive generations. An example for `saturate` is `"saturate_7"` which means stop the `run()` method if the fitness does not change for 7 consecutive generations. 

Here is an example that stops the evolution if either the fitness value reached `127.4` or if the fitness saturates for `15` generations.

```python
import pygad
import numpy

equation_inputs = [4, -2, 3.5, 8, 9, 4]
desired_output = 44

def fitness_func(ga_instance, solution, solution_idx):
    output = numpy.sum(solution * equation_inputs)

    fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)

    return fitness

ga_instance = pygad.GA(num_generations=200,
                       sol_per_pop=10,
                       num_parents_mating=4,
                       num_genes=len(equation_inputs),
                       fitness_func=fitness_func,
                       stop_criteria=["reach_127.4", "saturate_15"])

ga_instance.run()
print(f"Number of generations passed is {ga_instance.generations_completed}")
```

### Multi-Objective Stop Criteria

When multi-objective is used, then there are 2 options to use the `stop_criteria` parameter with the `reach` keyword:

1. Pass a single value to use along the `reach` keyword to use across all the objectives.
2. Pass multiple values along the `reach` keyword. But the number of values must equal the number of objectives.  

For the `saturate` keyword, it is independent of the number of objectives.

Suppose there are 3 objectives. Here is a working example. It stops when the fitness values of the 3 objectives reach or exceed 10, 20, and 30, respectively.

```python
stop_criteria='reach_10_20_30'
```

More than one criterion can be used together. In this case, pass the `stop_criteria` parameter as an iterable. This is an example. It stops when either of these 2 conditions hold:

1. The fitness values of the 3 objectives reach or exceed 10, 20, and 30, respectively.
2. The fitness values of the 3 objectives reach or exceed 90, -5.7, and 10, respectively.

```python
stop_criteria=['reach_10_20_30', 'reach_90_-5.7_10']
```

## Elitism Selection

Starting from [PyGAD 2.18.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-18-0), there is a parameter called `keep_elitism`. It takes an integer that sets how many of the best solutions (the elitism) are kept in the next generation. It defaults to `1`, so only the best solution is kept by default.

The best solutions are copied to the next generation without any change. Crossover and mutation do not touch them. This makes sure the best solutions found so far are never lost.

In the next example, the `keep_elitism` parameter in the constructor of the `pygad.GA` class is set to `2`. So, the best 2 solutions in each generation are kept in the next generation.

```python
import numpy
import pygad

function_inputs = [4,-2,3.5,5,-11,-4.7]
desired_output = 44

def fitness_func(ga_instance, solution, solution_idx):
    output = numpy.sum(solution*function_inputs)
    fitness = 1.0 / numpy.abs(output - desired_output)
    return fitness

ga_instance = pygad.GA(num_generations=2,
                       num_parents_mating=3,
                       fitness_func=fitness_func,
                       num_genes=6,
                       sol_per_pop=5,
                       keep_elitism=2)

ga_instance.run()
```

The value passed to the `keep_elitism` parameter must meet 2 conditions:

1. It must be `>= 0`.
2. It must be `<= sol_per_pop`. Its value cannot be more than the number of solutions in the population.

In the previous example, if `keep_elitism` is set equal to `sol_per_pop` (which is `5`), then there is no evolution at all, as shown in the next figure. This is because all the 5 solutions are kept as elitism in the next generation, so no offspring are created.

```python
...

ga_instance = pygad.GA(...,
                       sol_per_pop=5,
                       keep_elitism=5)

ga_instance.run()
```



![elitism_kills_evolution](https://user-images.githubusercontent.com/16560492/189273225-67ffad41-97ab-45e1-9324-429705e17b20.png)

### How the Number of Offspring Is Decided

PyGAD has two parameters that decide how many solutions are carried over to the next generation:

- `keep_elitism`: keeps the best solutions (the elitism).
- `keep_parents`: keeps the selected parents.

Only one of them is used at a time, and `keep_elitism` has priority. If `keep_elitism` is not zero, then `keep_parents` is ignored. Because `keep_elitism` defaults to `1`, the `keep_parents` parameter has no effect by default. To use `keep_parents`, set `keep_elitism=0`.

The number of kept solutions decides how many offspring are created. The rest of the population is filled with new offspring:

```
number of offspring = sol_per_pop - (number of kept solutions)
```

The next tree shows how the two parameters decide the number of offspring.

:::{figure} images/offspring_decision_tree.*
:alt: Decision tree showing how keep_elitism and keep_parents decide the number of offspring
:width: 680px
:align: center

How `keep_elitism` and `keep_parents` decide the number of offspring.
:::

There are four cases:

| `keep_elitism` | `keep_parents` | What is kept | Number of offspring |
| --- | --- | --- | --- |
| `> 0` | ignored | the best `keep_elitism` solutions | `sol_per_pop - keep_elitism` |
| `0` | `-1` | all the parents | `sol_per_pop - num_parents_mating` |
| `0` | `0` | nothing | `sol_per_pop` |
| `0` | `> 0` | the best `keep_parents` parents | `sol_per_pop - keep_parents` |

The kept solutions are placed at the top of the next population, starting at index 0. The offspring fill the slots that remain.

:::{figure} images/population_assembly.*
:alt: The kept solutions sit at the top of the next population and the offspring fill the rest
:width: 620px
:align: center

The kept solutions are copied to the top of the population. The offspring fill the rest.
:::

## Random Seed

In [PyGAD 2.18.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-18-0), a new parameter called `random_seed` is supported. Its value is used as a seed for the random function generators.

 PyGAD uses random functions in these 2 libraries:

1.  NumPy
2.  random

The `random_seed` parameter defaults to `None` which means no seed is used. As a result, different random numbers are generated for each run of PyGAD.

If this parameter is assigned a proper seed, then the results will be reproducible. In the next example, the integer 2 is used as a random seed. 

```python
import numpy
import pygad

function_inputs = [4,-2,3.5,5,-11,-4.7]
desired_output = 44

def fitness_func(ga_instance, solution, solution_idx):
    output = numpy.sum(solution*function_inputs)
    fitness = 1.0 / numpy.abs(output - desired_output)
    return fitness

ga_instance = pygad.GA(num_generations=2,
                       num_parents_mating=3,
                       fitness_func=fitness_func,
                       sol_per_pop=5,
                       num_genes=6,
                       random_seed=2)

ga_instance.run()
best_solution, best_solution_fitness, best_match_idx = ga_instance.best_solution()
print(best_solution)
print(best_solution_fitness)
```

This is the best solution found and its fitness value.

```
[ 2.77249188 -4.06570662  0.04196872 -3.47770796 -0.57502138 -3.22775267]
0.04872203136549972
```

After running the code again, it will find the same result.

```
[ 2.77249188 -4.06570662  0.04196872 -3.47770796 -0.57502138 -3.22775267]
0.04872203136549972
```

## Continue without Losing Progress

In [PyGAD 2.18.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-18-0), and thanks for [Felix Bernhard](https://github.com/FeBe95) for opening [this GitHub issue](https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/123#issuecomment-1203035106), the values of these 4 instance attributes are no longer reset after each call to the `run()` method.

1. `self.best_solutions`
2. `self.best_solutions_fitness`
3. `self.solutions`
4. `self.solutions_fitness`

This helps the user to continue where the last run stopped without losing the values of these 4 attributes.

Now, the user can save the model by calling the `save()` method.

```python
import pygad

def fitness_func(ga_instance, solution, solution_idx):
    ...
    return fitness

ga_instance = pygad.GA(...)

ga_instance.run()

ga_instance.plot_fitness()

ga_instance.save("pygad_GA")
```

Then the saved model is loaded by calling the `load()` function. After calling the `run()` method over the loaded instance, then the data from the previous 4 attributes are not reset but extended with the new data.

```python
import pygad

def fitness_func(ga_instance, solution, solution_idx):
    ...
    return fitness

loaded_ga_instance = pygad.load("pygad_GA")

loaded_ga_instance.run()

loaded_ga_instance.plot_fitness()
```

The plot created by the `plot_fitness()` method will show the data collected from both the runs. 

Note that the 2 attributes (`self.best_solutions` and `self.best_solutions_fitness`) only work if the `save_best_solutions` parameter is set to `True`. Also, the 2 attributes (`self.solutions` and `self.solutions_fitness`) only work if the `save_solutions` parameter is `True`.

## Change Population Size during Runtime

Starting from [PyGAD 3.3.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-3-3-0), the population size can be changed during runtime. In other words, the number of solutions/chromosomes and the number of genes can be changed.

The user has to carefully arrange the list of *parameters* and *instance attributes* that have to be changed to keep the GA consistent before and after changing the population size. Generally, change everything that would be used during the GA evolution.

> CAUTION: If the user fails to change a parameter or an instance attribute that is needed to keep the GA running after the population size changes, errors will arise.

These are examples of the parameters that the user should decide whether to change. The user should check the [list of parameters](https://pygad.readthedocs.io/en/latest/pygad.html#init) and decide what to change.

1. `population`: The population. It *must* be changed.
2. `num_offspring`: The number of offspring to produce from the crossover and mutation operations. Change this parameter if the number of offspring has to change to match the new population size.
3. `num_parents_mating`: The number of solutions to select as parents. Change this parameter if the number of parents has to change to match the new population size.
4. `fitness_func`: If the way of calculating the fitness changes with the new population size, then the fitness function has to be changed.
5. `sol_per_pop`: The number of solutions per population. It is not critical to change it but it is recommended to keep this number consistent with the number of solutions in the `population` parameter. 

These are examples of the instance attributes that might be changed. The user should check the [list of instance attributes](https://pygad.readthedocs.io/en/latest/pygad.html#other-instance-attributes-methods) and decide what to change.

1. All the `last_generation_*` attributes
   1. `last_generation_fitness`: A 1D NumPy array of fitness values of the population.
   2. `last_generation_parents` and `last_generation_parents_indices`: Two NumPy arrays: 2D array representing the parents and 1D array of the parents indices.
   3. `last_generation_elitism` and `last_generation_elitism_indices`: Must be changed if `keep_elitism != 0`. The default value of `keep_elitism` is 1. Two NumPy arrays: 2D array representing the elitism and 1D array of the elitism indices.
2. `pop_size`: The population size.
