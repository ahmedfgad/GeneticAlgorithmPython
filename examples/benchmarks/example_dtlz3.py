"""
Run PyGAD with NSGA-III on the DTLZ3 benchmark from
`pygad.benchmarks.dtlz`.

DTLZ3 has the same unit-sphere front as DTLZ2 but its g-function is
hard and multimodal, which makes convergence much harder.
"""

import pygad
from pygad.benchmarks.dtlz import DTLZ3

problem = DTLZ3(num_objectives=3, num_distance_vars=10)

ga = pygad.GA(num_generations=500,
              num_parents_mating=20,
              fitness_func=problem,
              sol_per_pop=40,
              num_genes=problem.num_genes,
              init_range_low=problem.bounds[0],
              init_range_high=problem.bounds[1],
              parent_selection_type='nsga3',
              nsga3_num_divisions=12,
              crossover_type='sbx',
              sbx_crossover_eta=20,
              mutation_type='polynomial',
              polynomial_mutation_eta=20)
ga.run()
