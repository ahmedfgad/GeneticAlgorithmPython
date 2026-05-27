"""
Run PyGAD on the Rosenbrock benchmark from `pygad.benchmarks.classic`.

The global minimum is at x = (1, 1, ..., 1) where f(x) = 0. The
minimum lives at the bottom of a long, narrow, banana-shaped valley.
"""

import pygad
from pygad.benchmarks.classic import Rosenbrock

problem = Rosenbrock(num_genes=5)

ga = pygad.GA(num_generations=500,
              num_parents_mating=10,
              fitness_func=problem,
              sol_per_pop=40,
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
