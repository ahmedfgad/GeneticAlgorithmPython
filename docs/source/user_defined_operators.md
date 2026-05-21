# User-Defined Crossover, Mutation, and Parent Selection Operators

Previously, the user could select the type of the crossover, mutation, and parent selection operators by assigning the name of the operator to the following parameters of the `pygad.GA` class's constructor:

1. `crossover_type`
2. `mutation_type`
3. `parent_selection_type`

This way, the user can only use the built-in functions for each of these operators.

Starting from [PyGAD 2.16.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-16-0), the user can create a custom crossover, mutation, and parent selection operators and assign these functions to the above parameters. Thus, a new operator can be plugged easily into the [PyGAD Lifecycle](https://pygad.readthedocs.io/en/latest/lifecycle.html#life-cycle-of-pygad).

This is a sample code that does not use any custom function.

```python
import pygad
import numpy

equation_inputs = [4,-2,3.5]
desired_output = 44

def fitness_func(ga_instance, solution, solution_idx):
    output = numpy.sum(solution * equation_inputs)
    fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
    return fitness

ga_instance = pygad.GA(num_generations=10,
                       sol_per_pop=5,
                       num_parents_mating=2,
                       num_genes=len(equation_inputs),
                       fitness_func=fitness_func)

ga_instance.run()
ga_instance.plot_fitness()
```

This section describes the expected input parameters and outputs. For simplicity, all of these custom functions accept the instance of the `pygad.GA` class as the last parameter.

## User-Defined Crossover Operator

The user-defined crossover function is a Python function that accepts 3 parameters:

1. The selected parents.
2. The size of the offspring as a tuple of 2 numbers: (the offspring size, number of genes).
3. The instance from the `pygad.GA` class. This instance helps to retrieve any property like `population`, `gene_type`, `gene_space`, etc.

This function should return a NumPy array of shape equal to the value passed to the second parameter.

The next code creates a template for the user-defined crossover operator. You can use any names for the parameters. Note how a NumPy array is returned.

```python
def crossover_func(parents, offspring_size, ga_instance):
    offspring = ...
    ...
    return numpy.array(offspring)
```

As an example, the next code creates a single-point crossover function. By randomly generating a random point (i.e. index of a gene), the function simply uses 2 parents to produce an offspring by copying the genes before the point from the first parent and the remaining from the second parent.

```python
def crossover_func(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        random_split_point = numpy.random.choice(range(offspring_size[1]))

        parent1[random_split_point:] = parent2[random_split_point:]

        offspring.append(parent1)

        idx += 1

    return numpy.array(offspring)
```

To use this user-defined function, simply assign its name to the `crossover_type` parameter in the constructor of the `pygad.GA` class. The next code gives an example. In this case, the custom function will be called in each generation rather than calling the built-in crossover functions defined in PyGAD.

```python
ga_instance = pygad.GA(num_generations=10,
                       sol_per_pop=5,
                       num_parents_mating=2,
                       num_genes=len(equation_inputs),
                       fitness_func=fitness_func,
                       crossover_type=crossover_func)
```

## User-Defined Mutation Operator

A user-defined mutation function/operator can be created the same way a custom crossover operator/function is created. Simply, it is a Python function that accepts 2 parameters:

1. The offspring to be mutated.
2. The instance from the `pygad.GA` class. This instance helps to retrieve any property like `population`, `gene_type`, `gene_space`, etc.

The template for the user-defined mutation function is given in the next code. According to the user preference, the function should make some random changes to the genes.

```python
def mutation_func(offspring, ga_instance):
    ...
    return offspring
```

The next code builds the random mutation where a single gene from each chromosome is mutated by adding a random number between 0 and 1 to the gene's value.

```python
def mutation_func(offspring, ga_instance):

    for chromosome_idx in range(offspring.shape[0]):
        random_gene_idx = numpy.random.choice(range(offspring.shape[1]))

        offspring[chromosome_idx, random_gene_idx] += numpy.random.random()

    return offspring
```

Here is how this function is assigned to the `mutation_type` parameter.

```python
ga_instance = pygad.GA(num_generations=10,
                       sol_per_pop=5,
                       num_parents_mating=2,
                       num_genes=len(equation_inputs),
                       fitness_func=fitness_func,
                       crossover_type=crossover_func,
                       mutation_type=mutation_func)
```

Note that there are other things to take into consideration like:

- Making sure that each gene conforms to the data type(s) listed in the `gene_type` parameter.
- If the `gene_space` parameter is used, then the new value for the gene should conform to the values/ranges listed. 
- Mutating a number of genes that conforms to the parameters `mutation_percent_genes`, `mutation_probability`, and `mutation_num_genes`.
- Whether mutation happens with or without replacement based on the `mutation_by_replacement` parameter.
- The minimum and maximum values from which a random value is generated based on the `random_mutation_min_val` and `random_mutation_max_val` parameters.
- Whether duplicates are allowed or not in the chromosome based on the `allow_duplicate_genes` parameter.

and more.

It all depends on your goal in building the mutation function. You may ignore or apply some of these points depending on your goal.

## User-Defined Parent Selection Operator

There is not much to add about building a user-defined parent selection function, as it is similar to building a crossover or mutation function. Just create a Python function that accepts 3 parameters:

1. The fitness values of the current population.
2. The number of parents needed.
3. The instance from the `pygad.GA` class. This instance helps to retrieve any property like `population`, `gene_type`, `gene_space`, etc.

The function should return 2 outputs:

1. The selected parents as a NumPy array. Its shape is equal to (the number of selected parents, `num_genes`). Note that the number of selected parents is equal to the value assigned to the second input parameter.
2. The indices of the selected parents inside the population. It is a 1D list with length equal to the number of selected parents.

The outputs must be of type `numpy.ndarray`.

Here is a template for building a custom parent selection function.

```python
def parent_selection_func(fitness, num_parents, ga_instance):
    ...
    return parents, fitness_sorted[:num_parents]
```

The next code builds the steady-state parent selection where the best parents are selected. The number of parents is equal to the value in the `num_parents` parameter. 

```python
def parent_selection_func(fitness, num_parents, ga_instance):

    fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
    fitness_sorted.reverse()

    parents = numpy.empty((num_parents, ga_instance.population.shape[1]))

    for parent_num in range(num_parents):
        parents[parent_num, :] = ga_instance.population[fitness_sorted[parent_num], :].copy()

    return parents, numpy.array(fitness_sorted[:num_parents])
```

Finally, the defined function is assigned to the `parent_selection_type` parameter as in the next code.

```python
ga_instance = pygad.GA(num_generations=10,
                       sol_per_pop=5,
                       num_parents_mating=2,
                       num_genes=len(equation_inputs),
                       fitness_func=fitness_func,
                       crossover_type=crossover_func,
                       mutation_type=mutation_func,
                       parent_selection_type=parent_selection_func)
```

## Example

Now that we have seen how to customize the 3 operators, the next code uses the previous 3 user-defined functions instead of the built-in ones.

```python
import pygad
import numpy

equation_inputs = [4,-2,3.5]
desired_output = 44

def fitness_func(ga_instance, solution, solution_idx):
    output = numpy.sum(solution * equation_inputs)

    fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)

    return fitness

def parent_selection_func(fitness, num_parents, ga_instance):

    fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
    fitness_sorted.reverse()

    parents = numpy.empty((num_parents, ga_instance.population.shape[1]))

    for parent_num in range(num_parents):
        parents[parent_num, :] = ga_instance.population[fitness_sorted[parent_num], :].copy()

    return parents, numpy.array(fitness_sorted[:num_parents])

def crossover_func(parents, offspring_size, ga_instance):

    offspring = []
    idx = 0
    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        random_split_point = numpy.random.choice(range(offspring_size[1]))

        parent1[random_split_point:] = parent2[random_split_point:]

        offspring.append(parent1)

        idx += 1

    return numpy.array(offspring)

def mutation_func(offspring, ga_instance):

    for chromosome_idx in range(offspring.shape[0]):
        random_gene_idx = numpy.random.choice(range(offspring.shape[0]))

        offspring[chromosome_idx, random_gene_idx] += numpy.random.random()

    return offspring

ga_instance = pygad.GA(num_generations=10,
                       sol_per_pop=5,
                       num_parents_mating=2,
                       num_genes=len(equation_inputs),
                       fitness_func=fitness_func,
                       crossover_type=crossover_func,
                       mutation_type=mutation_func,
                       parent_selection_type=parent_selection_func)

ga_instance.run()
ga_instance.plot_fitness()
```

This is the same example but using methods instead of functions.

```python
import pygad
import numpy

equation_inputs = [4,-2,3.5]
desired_output = 44

class Test:
    def fitness_func(self, ga_instance, solution, solution_idx):
        output = numpy.sum(solution * equation_inputs)
    
        fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
    
        return fitness
    
    def parent_selection_func(self, fitness, num_parents, ga_instance):
    
        fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
        fitness_sorted.reverse()
    
        parents = numpy.empty((num_parents, ga_instance.population.shape[1]))
    
        for parent_num in range(num_parents):
            parents[parent_num, :] = ga_instance.population[fitness_sorted[parent_num], :].copy()
    
        return parents, numpy.array(fitness_sorted[:num_parents])
    
    def crossover_func(self, parents, offspring_size, ga_instance):

        offspring = []
        idx = 0
        while len(offspring) != offspring_size[0]:
            parent1 = parents[idx % parents.shape[0], :].copy()
            parent2 = parents[(idx + 1) % parents.shape[0], :].copy()
    
            random_split_point = numpy.random.choice(range(offspring_size[0]))
    
            parent1[random_split_point:] = parent2[random_split_point:]
    
            offspring.append(parent1)
    
            idx += 1
    
        return numpy.array(offspring)
    
    def mutation_func(self, offspring, ga_instance):

        for chromosome_idx in range(offspring.shape[0]):
            random_gene_idx = numpy.random.choice(range(offspring.shape[1]))
    
            offspring[chromosome_idx, random_gene_idx] += numpy.random.random()
    
        return offspring

ga_instance = pygad.GA(num_generations=10,
                       sol_per_pop=5,
                       num_parents_mating=2,
                       num_genes=len(equation_inputs),
                       fitness_func=Test().fitness_func,
                       parent_selection_type=Test().parent_selection_func,
                       crossover_type=Test().crossover_func,
                       mutation_type=Test().mutation_func)

ga_instance.run()
ga_instance.plot_fitness()
```
