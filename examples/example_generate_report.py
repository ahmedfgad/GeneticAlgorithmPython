"""
Build a PDF report for a small multi-objective GA run.

Requires the optional ``report`` extra: pip install pygad[report].
The output file ``pygad_report.pdf`` is written next to this script.
"""

import numpy
import pygad

def fitness_func(ga_instance, solution, solution_idx):
    return [float(numpy.sum(solution)),
            -float(numpy.sum(numpy.asarray(solution) ** 2))]

ga_instance = pygad.GA(num_generations=30,
                       num_parents_mating=8,
                       fitness_func=fitness_func,
                       sol_per_pop=20,
                       num_genes=4,
                       parent_selection_type="nsga2",
                       save_solutions=True,
                       random_seed=42,
                       suppress_warnings=True,)
ga_instance.run()
output_path = ga_instance.generate_report(filename="pygad_report",
                                          title="PyGAD multi-objective demo",
                                          notes="A short two-objective example with 30 generations.",)
print(f"Report written to: {output_path}")
