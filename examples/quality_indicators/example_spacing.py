"""Compute the spacing metric of the final NSGA-II population on ZDT1.

Spacing is the standard deviation of each solution's nearest
neighbour distance. Smaller means a more even spread.
"""

import pygad
from pygad.benchmarks.zdt import ZDT1
from pygad.utils.quality_indicators import spacing

problem = ZDT1(num_genes=10)

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

spacing_value = spacing(ga.last_generation_fitness)
print(f"Spacing: {spacing_value}")
