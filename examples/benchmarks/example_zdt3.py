"""
Run PyGAD with NSGA-II on the ZDT3 benchmark from
`pygad.benchmarks.zdt`.

ZDT3 has a Pareto front that is made of five disconnected convex
pieces, which makes it a good test for diversity preservation.
"""

import pygad
from pygad.benchmarks.zdt import ZDT3

problem = ZDT3(num_genes=10)

ga = pygad.GA(num_generations=200,
              num_parents_mating=20,
              fitness_func=problem,
              sol_per_pop=40,
              num_genes=problem.num_genes,
              init_range_low=problem.bounds[0],
              init_range_high=problem.bounds[1],
              parent_selection_type='nsga2',
              crossover_type='sbx',
              sbx_crossover_eta=30,
              mutation_type='polynomial',
              polynomial_mutation_eta=20)
ga.run()

ga.plot_pareto_front_curve()
