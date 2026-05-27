"""
Compute the Generational Distance (GD) of the final NSGA-II
population on the ZDT2 benchmark.

GD is the mean Euclidean distance from each approximation point to
its nearest true-front reference point. A smaller value is better;
GD only measures convergence and not diversity.
"""

import pygad
from pygad.benchmarks.zdt import ZDT2
from pygad.utils.quality_indicators import generational_distance

problem = ZDT2(num_genes=10)

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

true_front = problem.pareto_front(num_points=100)
gd = generational_distance(ga.last_generation_fitness, true_front)
print(f"GD: {gd}")
