"""Count of new solutions per generation on a single-objective GA run."""

import pygad
from pygad.benchmarks.classic import Sphere

problem = Sphere(num_genes=5)

ga = pygad.GA(num_generations=50,
              num_parents_mating=10,
              fitness_func=problem,
              sol_per_pop=20,
              num_genes=problem.num_genes,
              init_range_low=problem.bounds[0],
              init_range_high=problem.bounds[1],
              save_solutions=True)
ga.run()

ga.plot_new_solution_rate()
