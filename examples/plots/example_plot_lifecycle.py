"""Configured lifecycle for a single-objective GA on the Sphere benchmark."""

import pygad
from pygad.benchmarks.classic import Sphere

problem = Sphere(num_genes=5)


def report_generation(ga_instance):
    """Report progress using fitness already calculated by the GA."""
    solution, fitness, solution_idx = ga_instance.best_solution(
        pop_fitness=ga_instance.last_generation_fitness)
    print(f"Generation {ga_instance.generations_completed}: best fitness = {fitness}")


ga_instance = pygad.GA(num_generations=50,
                       num_parents_mating=10,
                       fitness_func=problem,
                       sol_per_pop=20,
                       num_genes=problem.num_genes,
                       gene_type=[float, 3],
                       init_range_low=problem.bounds[0],
                       init_range_high=problem.bounds[1],
                       parent_selection_type="tournament",
                       K_tournament=3,
                       crossover_type="uniform",
                       crossover_probability=0.8,
                       mutation_type="adaptive",
                       mutation_probability=[0.2, 0.05],
                       keep_elitism=2,
                       on_generation=report_generation,
                       stop_criteria="saturate_10",
                       random_seed=42)

# The configuration is enough to draw the chart. No fitness function
# or callback is called by plot_lifecycle(). SVG stays sharp when resized.
ga_instance.plot_lifecycle(title="PyGAD - Sphere Optimization",
                           save_dir="lifecycle.svg")

ga_instance.run()

# After run(), the chart also shows the known number of objectives
# and the fitness shape. show=False saves it without opening a window.
ga_instance.plot_lifecycle(title="PyGAD - Sphere Optimization",
                           save_dir="lifecycle_after_run.png",
                           show=False)

# A compact view keeps operator and callback names and the control flow.
ga_instance.plot_lifecycle(show_parameters=False)
