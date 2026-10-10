"""Prevent duplicate genes when repair needs a chain of replacements."""

import pygad


def fitness_func(ga_instance, solution, solution_idx):
    return sum((index + 1) * value for index, value in enumerate(solution))


def on_generation(ga_instance):
    for solution in ga_instance.population:
        assert len(set(solution)) == len(solution)
        assert all(value in space for value, space in zip(solution, ga_instance.gene_space))


gene_space = [[0, 1], [1, 2], [2, 3], [0]]
initial_population = [[0, 1, 2, 0], [0, 1, 2, 0]]

ga_instance = pygad.GA(num_generations=5,
                       num_parents_mating=2,
                       fitness_func=fitness_func,
                       initial_population=initial_population,
                       gene_space=gene_space,
                       gene_type=int,
                       allow_duplicate_genes=False,
                       mutation_num_genes=1,
                       on_generation=on_generation,
                       random_seed=1)

# The last gene can only keep 0. Repair moves the first three genes to
# 1, 2, and 3, making room for 0 without leaving any gene's own space.
print("Initial Population")
print(ga_instance.initial_population)

ga_instance.run()

print("Final Population")
print(ga_instance.population)
