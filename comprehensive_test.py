import sys
sys.path.append('pygad')
import pygad_modified as ga
import numpy

# Multi-objective fitness function
def fitness_func(ga_instance, solution):
    # Three objectives: maximize fitness, minimize time cost, maximize diversity
    fitness_score = numpy.sum(solution)
    time_cost = numpy.random.randint(10, 50)  # Simulate time cost
    diversity_score = numpy.random.uniform(0.5, 1.0)  # Simulate diversity score
    return [fitness_score, time_cost, diversity_score]

# Callback function to track environment changes
def on_generation(ga_instance):
    print(f"Generation {ga_instance.generations_completed}: Environment = {ga_instance.current_environment}")

print("=== Comprehensive Test: Environment State Machine & Multi-Objective Optimization ===\n")

# Create GA instance with environment state machine enabled
ga_instance = ga.GA(
    num_generations=30,  # Should cycle through 3 environments (30 generations / 10 = 3 cycles)
    num_parents_mating=5,
    fitness_func=fitness_func,
    sol_per_pop=20,
    num_genes=10,
    parent_selection_type="sss",
    crossover_type="single_point",
    mutation_type="random",
    mutation_probability=0.05,
    crossover_probability=0.7,
    on_generation=on_generation
)

print(f"Initial Environment: {ga_instance.current_environment}")
print("\nStarting Genetic Algorithm with Environment State Machine...\n")

# Run the GA with environment state machine
ga_instance.run()

print("\n=== GA Run Completed ===")
print(f"Total generations completed: {ga_instance.generations_completed}")
print(f"Final environment: {ga_instance.current_environment}")

# Show Pareto front data from the last generation
print("\n=== Pareto Front Data from Last Generation ===")
if hasattr(ga_instance, 'last_generation_fitness'):
    print(f"Last generation fitness data shape: {ga_instance.last_generation_fitness.shape}")
    print("Each solution has 3 objectives: [fitness_score, time_cost, diversity_score]")

print("\nTest completed successfully!")
