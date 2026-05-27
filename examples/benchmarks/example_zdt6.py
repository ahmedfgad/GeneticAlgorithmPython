"""
Run PyGAD with NSGA-II on the ZDT6 benchmark from
`pygad.benchmarks.zdt`.

ZDT6 has a non-uniform Pareto front. Solutions cluster toward one
end so a good algorithm has to preserve diversity over the whole
front.
"""

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
