"""
Run PyGAD on the Sphere benchmark from `pygad.benchmarks.classic`.

The Sphere function has its global minimum at the origin where
f(x) = 0. Under PyGAD's max convention the best fitness is 0.
"""

import pygad
from pygad.benchmarks.classic import Sphere

problem = Sphere(num_genes=10)

ga = pygad.GA(num_generations=200,
              num_parents_mating=10,
              fitness_func=problem,
              sol_per_pop=30,
              num_genes=problem.num_genes,
              init_range_low=problem.bounds[0],
              init_range_high=problem.bounds[1],
              crossover_type='sbx',
              sbx_crossover_eta=30,
              mutation_type='polynomial',
              polynomial_mutation_eta=20)
ga.run()

solution, solution_fitness, _ = ga.best_solution(ga.last_generation_fitness)
print(f"Best fitness: {solution_fitness}")
print(f"Best solution: {solution}")
