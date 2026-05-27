"""Mean pairwise distance per generation on the Sphere benchmark."""

import pygad
from pygad.benchmarks.classic import Sphere

problem = Sphere(num_genes=5)

ga = pygad.GA(num_generations=80,
              num_parents_mating=10,
              fitness_func=problem,
              sol_per_pop=20,
              num_genes=problem.num_genes,
              init_range_low=problem.bounds[0],
              init_range_high=problem.bounds[1],
              save_solutions=True)
ga.run()

ga.plot_population_diversity()
