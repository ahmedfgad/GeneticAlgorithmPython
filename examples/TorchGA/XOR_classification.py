import torch
import pygad.torchga
import pygad

def fitness_func(ga_instanse, solution, sol_idx):
    global data_inputs, data_outputs, torch_ga, model, loss_function

    predictions = pygad.torchga.predict(model=model, 
                                        solution=solution, 
                                        data=data_inputs)

    solution_fitness = 1.0 / (loss_function(predictions, data_outputs).detach().numpy() + 0.00000001)

    return solution_fitness

def on_generation(ga_instance):
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {ga_instance.best_solution()[1]}")

# Create the PyTorch model.
input_layer  = torch.nn.Linear(2, 4)
relu_layer = torch.nn.ReLU()
dense_layer = torch.nn.Linear(4, 2)
output_layer = torch.nn.Softmax(1)

model = torch.nn.Sequential(input_layer,
                            relu_layer,
                            dense_layer,
                            output_layer)
# print(model)

# Create an instance of the pygad.torchga.TorchGA class to build the initial population.
torch_ga = pygad.torchga.TorchGA(model=model,
                                 num_solutions=10)

loss_function = torch.nn.BCELoss()

# XOR problem inputs
data_inputs = torch.tensor([[0.0, 0.0],
                            [0.0, 1.0],
                            [1.0, 0.0],
                            [1.0, 1.0]])

# XOR problem outputs
data_outputs = torch.tensor([[1.0, 0.0],
                             [0.0, 1.0],
                             [0.0, 1.0],
                             [1.0, 0.0]])

# Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
num_generations = 250 # Number of generations.
num_parents_mating = 5 # Number of solutions to be selected as parents in the mating pool.
initial_population = torch_ga.population_weights # Initial population of network weights.

# Create an instance of the pygad.GA class
ga_instance = pygad.GA(num_generations=num_generations, 
                       num_parents_mating=num_parents_mating, 
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       parallel_processing=3,
                       on_generation=on_generation)

# Start the genetic algorithm evolution.
ga_instance.run()

# After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Fitness value of the best solution = {solution_fitness}")
print(f"Index of the best solution : {solution_idx}")

predictions = pygad.torchga.predict(model=model, 
                                    solution=solution, 
                                    data=data_inputs)
print(f"Predictions : \n{predictions.detach().numpy()}")

# Calculate the binary crossentropy for the trained model.
print(f"Binary Crossentropy : {loss_function(predictions, data_outputs).detach().numpy()}")

# Calculate the classification accuracy of the trained model.
a = torch.max(predictions, axis=1)
b = torch.max(data_outputs, axis=1)
accuracy = torch.true_divide(torch.sum(a.indices == b.indices), len(data_outputs))
print(f"Accuracy : {accuracy.detach().numpy()}")
