"""Run NSGA-II on ZDT6. Non-uniform front; solutions cluster at one end."""

import pygad
from pygad.benchmarks.zdt import ZDT6

problem = ZDT6(num_genes=10)

ga = pygad.GA(num_generations=300,
              num_parents_mating=20,
              fitness_func=problem,
              sol_per_pop=40,
              num_genes=problem.num_genes,
              init_range_low=problem.bounds[0],
              init_range_high=problem.bounds[1],
              parent_selection_type='nsga2',
              crossover_type='sbx',
              sbx_crossover_eta=20,
              mutation_type='polynomial',
              polynomial_mutation_eta=20)
ga.run()

ga.plot_pareto_front_curve()
