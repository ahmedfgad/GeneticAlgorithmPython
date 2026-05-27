"""
Run PyGAD on the Schwefel benchmark from `pygad.benchmarks.classic`.

The global minimum sits at x = (420.9687, ..., 420.9687) far from
the next-best local minimum, so the function is hard for many
algorithms.
"""

import pygad
from pygad.benchmarks.classic import Schwefel

problem = Schwefel(num_genes=3)

ga = pygad.GA(num_generations=500,
              num_parents_mating=10,
              fitness_func=problem,
              sol_per_pop=50,
              num_genes=problem.num_genes,
              init_range_low=problem.bounds[0],
              init_range_high=problem.bounds[1],
              crossover_type='sbx',
              sbx_crossover_eta=20,
              mutation_type='polynomial',
              polynomial_mutation_eta=20)
ga.run()

solution, solution_fitness, _ = ga.best_solution(ga.last_generation_fitness)
print(f"Best fitness: {solution_fitness}")
print(f"Best solution: {solution}")
