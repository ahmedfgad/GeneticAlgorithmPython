import sys
sys.path.append('pygad')
import pygad_modified as ga
import numpy

def fitness_func(ga_instance, solution):
    # Example multi-objective fitness function
    fitness_score = numpy.sum(solution)
    time_cost = numpy.random.randint(10, 50)  # Simulate time cost
    diversity_score = numpy.random.uniform(0.5, 1.0)  # Simulate diversity score
    return [fitness_score, time_cost, diversity_score]

def on_generation(ga_instance):
    print(f"Generation completed: {ga_instance.generations_completed}")
    print(f"Current environment: {ga_instance.current_environment}")

# Initialize the GA instance
ga_instance = ga.GA(
    num_generations=30,
    num_parents_mating=10,
    fitness_func=fitness_func,
    sol_per_pop=50,
    num_genes=5,
    parent_selection_type="sss",
    crossover_type="single_point",
    mutation_type="random",
    mutation_probability=0.05,
    crossover_probability=0.7,
    on_generation=on_generation
)

# Run the GA
ga_instance.run()

print("GA run completed!")
