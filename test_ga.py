import sys
sys.path.append('pygad')
import pygad_modified as ga
import numpy

def fitness_func(ga_instance, solution):
    return numpy.sum(solution)

print("Creating GA instance...")
try:
    ga_instance = ga.GA(
        num_generations=10,
        num_parents_mating=5,
        fitness_func=fitness_func,
        sol_per_pop=10,
        num_genes=5
    )
    print("GA instance created successfully!")
    print(f"Initial environment: {ga_instance.current_environment}")
except Exception as e:
    print(f"Error creating GA instance: {e}")
    import traceback
    traceback.print_exc()
