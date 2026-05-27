"""
Run PyGAD on the TSP benchmark from `pygad.benchmarks.tsp`.

A solution is a permutation of the city indices. The fitness is the
negative tour length so the GA maximizes toward the shortest tour.
For this 4-city square instance the optimal tour walks the
perimeter with length 4.
"""

import pygad
from pygad.benchmarks.tsp import TSP

problem = TSP(coordinates=[[0.0, 0.0],
                           [1.0, 0.0],
                           [1.0, 1.0],
                           [0.0, 1.0]])

ga = pygad.GA(num_generations=200,
              num_parents_mating=10,
              fitness_func=problem,
              sol_per_pop=30,
              num_genes=problem.num_genes,
              gene_space=problem.gene_space,
              gene_type=problem.gene_type,
              allow_duplicate_genes=problem.allow_duplicate_genes)
ga.run()

solution, solution_fitness, _ = ga.best_solution(ga.last_generation_fitness)
print(f"Best tour: {solution}")
print(f"Tour length: {-solution_fitness}")
