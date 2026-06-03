"""Run NSGA-III on DTLZ4. Same shape as DTLZ2; alpha bias pushes solutions to one corner."""

import pygad
from pygad.benchmarks.dtlz import DTLZ4

problem = DTLZ4(num_objectives=3, num_distance_vars=10, alpha=100.0)

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
