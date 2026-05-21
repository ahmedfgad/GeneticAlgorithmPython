# Adaptive Mutation

In the regular genetic algorithm, mutation uses a single fixed mutation rate for all solutions, regardless of their fitness values. So, no matter whether a solution has high or low quality, the same number of genes is mutated every time.

The pitfalls of using a constant mutation rate for all solutions are summarized in this paper [Libelli, S. Marsili, and P. Alba. "Adaptive mutation in genetic algorithms." *Soft computing* 4.2 (2000): 76-80](https://idp.springer.com/authorize/casa?redirect_uri=https://link.springer.com/content/pdf/10.1007/s005000000042.pdf&casa_token=IT4NfJUvslcAAAAA:VegHW6tm2fe3e0R9cRKjuGKkKWXJTQSfNMT6z0kGbMsAllyK1NrEY3cEWg8bj7AJWEQPaqWIJxmHNBHg) as follows:

> The weak point of "classical" GAs is the total randomness of mutation, which is applied equally to all chromosomes, irrespective of their fitness. Thus a very good chromosome is equally likely to be disrupted by mutation as a bad one.
>
> On the other hand, bad chromosomes are less likely to produce good ones through crossover, because of their lack of building blocks, until they remain unchanged. They would benefit the most from mutation and could be used to spread throughout the parameter space to increase the search thoroughness. So there are two conflicting needs in determining the best probability of mutation. 
>
> Usually, a reasonable compromise in the case of a constant mutation is to keep the probability low to avoid disruption of good chromosomes, but this would prevent a high mutation rate of low-fitness chromosomes. Thus a constant probability of mutation would probably miss both goals and result in a slow improvement of the population.

According to [Libelli, S. Marsili, and P. Alba.](https://idp.springer.com/authorize/casa?redirect_uri=https://link.springer.com/content/pdf/10.1007/s005000000042.pdf&casa_token=IT4NfJUvslcAAAAA:VegHW6tm2fe3e0R9cRKjuGKkKWXJTQSfNMT6z0kGbMsAllyK1NrEY3cEWg8bj7AJWEQPaqWIJxmHNBHg) work, the adaptive mutation solves the problems of constant mutation. 

Adaptive mutation works as follows:

1. Calculate the average fitness value of the population (`f_avg`).
2. For each chromosome, calculate its fitness value (`f`).
3. If `f<f_avg`, then this solution is regarded as a low-quality solution and thus the mutation rate should be kept high because this would increase the quality of this solution.
4. If `f>f_avg`, then this solution is regarded as a high-quality solution and thus the mutation rate should be kept low to avoid disrupting this high quality solution.

In PyGAD, if `f=f_avg`, then the solution is regarded as high quality.

The next figure summarizes the previous steps.

![Adaptive-Mutation](https://user-images.githubusercontent.com/16560492/103468973-e3c26600-4d2c-11eb-8af3-b3bb39b50540.jpg)

This strategy is applied in PyGAD. 

## Use Adaptive Mutation in PyGAD

In [PyGAD 2.10.0](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-10-0), adaptive mutation is supported. To use it, just follow the following 2 simple steps:

1. In the constructor of the `pygad.GA` class, set `mutation_type="adaptive"` to specify that the type of mutation is adaptive.
2. Specify the mutation rates for the low and high quality solutions using one of these 3 parameters according to your preference: `mutation_probability`, `mutation_num_genes`, and `mutation_percent_genes`. Please check the [documentation of each of these parameters](https://pygad.readthedocs.io/en/latest/pygad.html#init) for more information. 

When adaptive mutation is used, then the value assigned to any of the 3 parameters can be of any of these data types:

1. `list`
2. `tuple`
3. `numpy.ndarray`

Whatever the data type used, the length of the `list`, `tuple`, or the `numpy.ndarray` must be exactly 2. That is there are just 2 values:

1. The first value is the mutation rate for the low-quality solutions.
2. The second value is the mutation rate for the high-quality solutions.

PyGAD expects that the first value is higher than the second value and thus a warning is printed in case the first value is lower than the second one.

Here are some examples to feed the mutation rates:

```python
# mutation_probability
mutation_probability = [0.25, 0.1]
mutation_probability = (0.35, 0.17)
mutation_probability = numpy.array([0.15, 0.05])

# mutation_num_genes
mutation_num_genes = [4, 2]
mutation_num_genes = (3, 1)
mutation_num_genes = numpy.array([7, 2])

# mutation_percent_genes
mutation_percent_genes = [25, 12]
mutation_percent_genes = (15, 8)
mutation_percent_genes = numpy.array([21, 13])
```

Assume that the average fitness is 12 and the fitness values of 2 solutions are 15 and 7. If the mutation probabilities are specified as follows:

```python
mutation_probability = [0.25, 0.1]
```

Then the mutation probability of the first solution is 0.1 because its fitness is 15 which is higher than the average fitness 12. The mutation probability of the second solution is 0.25 because its fitness is 7 which is lower than the average fitness 12.

Here is an example that uses adaptive mutation.

```python
import pygad
import numpy

function_inputs = [4,-2,3.5,5,-11,-4.7] # Function inputs.
desired_output = 44 # Function output.

def fitness_func(ga_instance, solution, solution_idx):
    # The fitness function calculates the sum of products between each input and its corresponding weight.
    output = numpy.sum(solution*function_inputs)
    # The value 0.000001 is used to avoid the Inf value when the denominator numpy.abs(output - desired_output) is 0.0.
    fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
    return fitness

# Creating an instance of the GA class inside the ga module. Some parameters are initialized within the constructor.
ga_instance = pygad.GA(num_generations=200,
                       fitness_func=fitness_func,
                       num_parents_mating=10,
                       sol_per_pop=20,
                       num_genes=len(function_inputs),
                       mutation_type="adaptive",
                       mutation_num_genes=(3, 1))

# Running the GA to optimize the parameters of the function.
ga_instance.run()

ga_instance.plot_fitness(title="PyGAD with Adaptive Mutation", linewidth=5)
```
