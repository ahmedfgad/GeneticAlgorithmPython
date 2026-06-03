"""Hypervolume of the non-dominated set per generation (NSGA-II on ZDT1)."""

import pygad
from pygad.benchmarks.zdt import ZDT1

problem = ZDT1(num_genes=10)

ga = pygad.GA(num_generations=80,
              num_parents_mating=20,
              fitness_func=problem,
              sol_per_pop=30,
              num_genes=problem.num_genes,
              init_range_low=problem.bounds[0],
              init_range_high=problem.bounds[1],
              parent_selection_type='nsga2',
              save_solutions=True,
              crossover_type='sbx',
              sbx_crossover_eta=30,
              mutation_type='polynomial',
              polynomial_mutation_eta=20)
ga.run()

ga.plot_non_dominated_hypervolume()
