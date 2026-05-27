"""Pairwise scatter matrix of the final Pareto front (DTLZ2, M=4)."""

import pygad
from pygad.benchmarks.dtlz import DTLZ2

problem = DTLZ2(num_objectives=4, num_distance_vars=5)

ga = pygad.GA(num_generations=100,
              num_parents_mating=20,
              fitness_func=problem,
              sol_per_pop=60,
              num_genes=problem.num_genes,
              init_range_low=problem.bounds[0],
              init_range_high=problem.bounds[1],
              parent_selection_type='nsga3',
              nsga3_num_divisions=5,
              crossover_type='sbx',
              sbx_crossover_eta=30,
              mutation_type='polynomial',
              polynomial_mutation_eta=20)
ga.run()

ga.plot_pareto_front_scatter_matrix()
