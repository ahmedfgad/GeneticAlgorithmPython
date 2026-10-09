"""Generate an initial population or start from supplied numeric values."""

import pygad


def fitness_func(ga_instance, solution, solution_idx):
    return sum(solution)


# Each column uses its own range and type. The float column is rounded
# to 2 decimal places while remaining inside [-2, 2).
random_population_ga = pygad.GA(num_generations=5,
                               num_parents_mating=2,
                               fitness_func=fitness_func,
                               sol_per_pop=4,
                               num_genes=3,
                               init_range_low=[0, -2, 20],
                               init_range_high=[5, 2, 30],
                               gene_type=[int, [float, 2], float],
                               mutation_num_genes=1,
                               random_seed=7)

print("Population from Per-Gene Ranges")
print(random_population_ga.initial_population)

# The third gene has no explicit space, so its values come from [20, 30).
# The last gene has the fixed value 5.
gene_space_ga = pygad.GA(num_generations=5,
                        num_parents_mating=2,
                        fitness_func=fitness_func,
                        sol_per_pop=4,
                        num_genes=4,
                        gene_space=[[0, 1, 2], {'low': 1, 'high': 2}, None, 5],
                        init_range_low=[0, 0, 20, 0],
                        init_range_high=[3, 3, 30, 10],
                        gene_type=[int, [float, 2], [float, 1], int],
                        mutation_num_genes=1,
                        random_seed=7)

print("Population from a Nested Gene Space")
print(gene_space_ga.initial_population)

# Neither sol_per_pop nor num_genes is needed. PyGAD infers both from
# the supplied population, converts the values, and keeps its own copy.
initial_population = ((1.236, 10), (2.341, 20))
supplied_population_ga = pygad.GA(num_generations=5,
                                num_parents_mating=2,
                                fitness_func=fitness_func,
                                initial_population=initial_population,
                                gene_type=([float, 2], int),
                                mutation_num_genes=1,
                                random_seed=7)

print("Population from Supplied Values")
print(supplied_population_ga.initial_population)
print("Inferred Population Shape", supplied_population_ga.pop_size)
print("Original Supplied Values", initial_population)
