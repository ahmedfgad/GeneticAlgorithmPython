"""
Run PyGAD with NSGA-III on the DTLZ1 benchmark from
`pygad.benchmarks.dtlz`.

DTLZ1 has a linear hyperplane Pareto front where sum(f_i) = 0.5
for M objectives. Use NSGA-III because the front lives in 3D.
"""

import pygad
from pygad.benchmarks.dtlz import DTLZ1

problem = DTLZ1(num_objectives=3, num_distance_vars=5)

ga = pygad.GA(num_generations=300,
              num_parents_mating=20,
              fitness_func=problem,
              sol_per_pop=40,
              num_genes=problem.num_genes,
              init_range_low=problem.bounds[0],
              init_range_high=problem.bounds[1],
              parent_selection_type='nsga3',
              nsga3_num_divisions=12,
              crossover_type='sbx',
              sbx_crossover_eta=30,
              mutation_type='polynomial',
              polynomial_mutation_eta=20)
ga.run()
