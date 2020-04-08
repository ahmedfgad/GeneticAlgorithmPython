import ga

"""
Given the following function:
    y = f(w1:w6) = w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + 6wx6
    where (x1,x2,x3,x4,x5,x6)=(4,-2,3.5,5,-11,-4.7) and y=44
What are the best values for the 6 weights (w1 to w6)? We are going to use the genetic algorithm to optimize this function.
"""

# Equation inputs.
function_inputs = [4,-2,3.5,5,-11,-4.7]
# Equation output.
function_output = 44

sol_per_pop = 8 # Number of solutions in the population.
num_parents_mating = 4 # Number of solutions to be selected as parents in the mating pool.
num_generations = 50 # Number of generations.

# Parameters of the mutation operation.
mutation_percent_genes=10 # Percentage of genes to mutate.

# Creating an instance of the GA class inside the ga module. Some parameters are initialized within the constructor.
ga_instance = ga.GA(num_generations=num_generations, 
          sol_per_pop=sol_per_pop, 
          num_parents_mating=num_parents_mating, 
          function_inputs=function_inputs,
          function_output=function_output,
          mutation_percent_genes=10)

# Training the GA to optimize the parameters of the function.
ga_instance.train()

# After the generations complete, some plots are showed that summarize the how the outputs/fitenss values evolve over generations.
ga_instance.plot_result()

# Returning the details of the best solution.
best_solution, best_solution_fitness = ga_instance.best_solution()
print("Parameters of the best solution :", best_solution)
print("Fitness value of the best solution :", best_solution_fitness)
