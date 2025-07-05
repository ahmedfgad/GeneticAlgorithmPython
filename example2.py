import pygad
import numpy
import random

def fitness_func(ga, solution, idx):
    return random.random()

ga_instance = pygad.GA(num_generations=1,
                       num_parents_mating=5,
                       sol_per_pop=10,
                       num_genes=10,
                       random_seed=2,
                       # mutation_type=None,
                       # crossover_type=None,
                       # random_mutation_min_val=1,
                       # random_mutation_max_val=100,
                       fitness_func=fitness_func,
                       gene_space=[5],
                       gene_type=float,
                       allow_duplicate_genes=False,
                       save_solutions=True)

print(ga_instance.initial_population)

ga_instance.run()

# print(ga_instance.gene_space_unpacked)
print(ga_instance.population)

"""
gene_space=[[0, 0],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 8],
            [8, 9],
            [9, 10]],
"""
