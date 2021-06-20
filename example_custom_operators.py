import pygad
import numpy

"""
This script gives an example of using custom user-defined functions for the 3 operators:
    1) Parent selection.
    2) Crossover.
    3) Mutation.
For more information, check the User-Defined Crossover, Mutation, and Parent Selection Operators section in the documentation:
    https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#user-defined-crossover-mutation-and-parent-selection-operators
"""

equation_inputs = [4,-2,3.5]
desired_output = 44

def fitness_func(solution, solution_idx):
    output = numpy.sum(solution * equation_inputs)

    fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)

    return fitness

def parent_selection_func(fitness, num_parents, ga_instance):
    # Selects the best {num_parents} parents. Works as steady-state selection.

    fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
    fitness_sorted.reverse()

    parents = numpy.empty((num_parents, ga_instance.population.shape[1]))

    for parent_num in range(num_parents):
        parents[parent_num, :] = ga_instance.population[fitness_sorted[parent_num], :].copy()

    return parents, fitness_sorted[:num_parents]

def crossover_func(parents, offspring_size, ga_instance):
    # This is single-point crossover.
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

def mutation_func(offspring, ga_instance):
    # This is random mutation that mutates a single gene.
    for chromosome_idx in range(offspring.shape[0]):
        # Make some random changes in 1 or more genes.
        random_gene_idx = numpy.random.choice(range(offspring.shape[0]))

        offspring[chromosome_idx, random_gene_idx] += numpy.random.random()

    return offspring

ga_instance = pygad.GA(num_generations=10,
                       sol_per_pop=5,
                       num_parents_mating=2,
                       num_genes=len(equation_inputs),
                       fitness_func=fitness_func,
                       parent_selection_type=parent_selection_func,
                       crossover_type=crossover_func,
                       mutation_type=mutation_func)

ga_instance.run()
ga_instance.plot_fitness()