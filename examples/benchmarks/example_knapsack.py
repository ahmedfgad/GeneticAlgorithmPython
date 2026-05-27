"""
Run PyGAD on the 0/1 Knapsack benchmark from
`pygad.benchmarks.knapsack`.

A solution is a binary vector. The fitness is the total value when
the candidate is within the capacity, and a negative penalty scaled
by the overweight amount otherwise. The known optimal subset for
this small instance is items 0 and 1 (total value = 7).
"""

import pygad
from pygad.benchmarks.knapsack import Knapsack

problem = Knapsack(weights=[2, 3, 4, 5],
                   values=[3, 4, 5, 6],
                   capacity=5)

ga = pygad.GA(num_generations=100,
              num_parents_mating=10,
              fitness_func=problem,
              sol_per_pop=30,
              num_genes=problem.num_genes,
              gene_space=problem.gene_space,
              gene_type=problem.gene_type)
ga.run()

solution, solution_fitness, _ = ga.best_solution(ga.last_generation_fitness)
print(f"Best subset: {solution}")
print(f"Total value: {solution_fitness}")
