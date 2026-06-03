"""Compute the hypervolume of the final NSGA-II population on ZDT1.

The reference point must be worse than every solution on every axis,
which under PyGAD-max means strictly smaller than every fitness.
"""

import numpy

import pygad
from pygad.benchmarks.zdt import ZDT1
from pygad.utils.quality_indicators import hypervolume

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

final_fitness = numpy.asarray(ga.last_generation_fitness)
reference_point = final_fitness.min(axis=0) - 0.1

hv = hypervolume(final_fitness, reference_point)
print(f"Hypervolume: {hv}")
