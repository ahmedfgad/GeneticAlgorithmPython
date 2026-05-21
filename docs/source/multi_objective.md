# Multi-Objective Optimization

In [PyGAD 3.2.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-3-2-0), the library added support for multi-objective optimization using the non-dominated sorting genetic algorithm II (NSGA-II). The code is almost the same as the code for single-objective optimization, except for one difference: the return value of the fitness function.

In single-objective optimization, the fitness function returns a single numeric value. In this example, the variable `fitness` is expected to be a numeric value.

```python
def fitness_func(ga_instance, solution, solution_idx):
    ...
    return fitness
```

But in multi-objective optimization, the fitness function returns any of these data types:

1. `list`
2. `tuple`
3. `numpy.ndarray`

```python
def fitness_func(ga_instance, solution, solution_idx):
    ...
    return [fitness1, fitness2, ..., fitnessN]
```

Whenever the fitness function returns an iterable of these data types, then the problem is considered multi-objective. This holds even if there is a single element in the returned iterable.

Other than the fitness function, everything else could be the same in both single and multi-objective problems.

But it is recommended to use one of these 2 parent selection operators to solve multi-objective problems:

1. `nsga2`: This selects the parents based on non-dominated sorting and crowding distance.
2. `tournament_nsga2`: This selects the parents using tournament selection which uses non-dominated sorting and crowding distance to rank the solutions.

This is a multi-objective optimization example that optimizes these 2 linear functions:

1. `y1 = f(w1:w6) = w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + w6x6`
2. `y2 = f(w1:w6) = w1x7 + w2x8 + w3x9 + w4x10 + w5x11 + w6x12`

Where:

1. `(x1,x2,x3,x4,x5,x6)=(4,-2,3.5,5,-11,-4.7)` and `y=50`
2. `(x7,x8,x9,x10,x11,x12)=(-2,0.7,-9,1.4,3,5)` and `y=30`

The 2 functions use the same parameters (weights) `w1` to `w6`.

The goal is to use PyGAD to find the optimal values for such weights that satisfy the 2 functions `y1` and `y2`.

```python
import pygad
import numpy

"""
Given these 2 functions:
    y1 = f(w1:w6) = w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + w6x6
    y2 = f(w1:w6) = w1x7 + w2x8 + w3x9 + w4x10 + w5x11 + w6x12
    where (x1,x2,x3,x4,x5,x6)=(4,-2,3.5,5,-11,-4.7) and y=50
    and   (x7,x8,x9,x10,x11,x12)=(-2,0.7,-9,1.4,3,5) and y=30
What are the best values for the 6 weights (w1 to w6)? We are going to use the genetic algorithm to optimize these 2 functions.
This is a multi-objective optimization problem.

PyGAD considers the problem as multi-objective if the fitness function returns:
    1) List.
    2) Or tuple.
    3) Or numpy.ndarray.
"""

function_inputs1 = [4,-2,3.5,5,-11,-4.7] # Function 1 inputs.
function_inputs2 = [-2,0.7,-9,1.4,3,5] # Function 2 inputs.
desired_output1 = 50 # Function 1 output.
desired_output2 = 30 # Function 2 output.

def fitness_func(ga_instance, solution, solution_idx):
    output1 = numpy.sum(solution*function_inputs1)
    output2 = numpy.sum(solution*function_inputs2)
    fitness1 = 1.0 / (numpy.abs(output1 - desired_output1) + 0.000001)
    fitness2 = 1.0 / (numpy.abs(output2 - desired_output2) + 0.000001)
    return [fitness1, fitness2]

num_generations = 100
num_parents_mating = 10

sol_per_pop = 20
num_genes = len(function_inputs1)

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       fitness_func=fitness_func,
                       parent_selection_type='nsga2')

ga_instance.run()

ga_instance.plot_fitness(label=['Obj 1', 'Obj 2'])

solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
print(f"Parameters of the best solution : {solution}")
print(f"Fitness value of the best solution = {solution_fitness}")

prediction = numpy.sum(numpy.array(function_inputs1)*solution)
print(f"Predicted output 1 based on the best solution : {prediction}")
prediction = numpy.sum(numpy.array(function_inputs2)*solution)
print(f"Predicted output 2 based on the best solution : {prediction}")
```

This is the result of the print statements. The predicted outputs are close to the desired outputs.

```
Parameters of the best solution : [ 0.79676439 -2.98823386 -4.12677662  5.70539445 -2.02797016 -1.07243922]
Fitness value of the best solution = [  1.68090829 349.8591915 ]
Predicted output 1 based on the best solution : 50.59491545442283
Predicted output 2 based on the best solution : 29.99714270722312
```

This is the figure created by the `plot_fitness()` method. The fitness of the first objective has the green color. The blue color is used for the second objective fitness.

![multi-objective-pygad](https://github.com/ahmedfgad/GeneticAlgorithmPython/assets/16560492/7896f8d8-01c5-4ff9-8d15-52191c309b63)
