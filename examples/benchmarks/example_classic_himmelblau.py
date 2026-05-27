"""
Run PyGAD on the 2D Himmelblau benchmark from
`pygad.benchmarks.classic`.

Himmelblau has four equal global minima at f(x, y) = 0:
    (3.0, 2.0),
    (-2.805, 3.131),
    (-3.779, -3.283),
    (3.584, -1.848).
"""

import pygad
from pygad.benchmarks.classic import Himmelblau

problem = Himmelblau()

ga = pygad.GA(num_generations=200,
              num_parents_mating=10,
              fitness_func=problem,
              sol_per_pop=30,
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
print(f"Best solution (x, y): {solution}")
