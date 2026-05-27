"""
Compute the Inverted Generational Distance (IGD) of the final
NSGA-II population on the ZDT1 benchmark.

IGD is the mean Euclidean distance from each true-front reference
point to its nearest approximation point. A smaller value is
better; it reports both convergence and diversity.
"""

import pygad
from pygad.benchmarks.zdt import ZDT1
from pygad.utils.quality_indicators import inverted_generational_distance

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

true_front = problem.pareto_front(num_points=100)
igd = inverted_generational_distance(ga.last_generation_fitness, true_front)
print(f"IGD: {igd}")
