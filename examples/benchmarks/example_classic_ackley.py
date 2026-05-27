"""
Run PyGAD on the Ackley benchmark from `pygad.benchmarks.classic`.

Ackley has a near-flat outer region with a deep, narrow basin at
the origin where f(x) = 0.
"""

import pygad
from pygad.benchmarks.classic import Ackley

problem = Ackley(num_genes=5)

ga = pygad.GA(num_generations=300,
              num_parents_mating=10,
              fitness_func=problem,
              sol_per_pop=40,
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
