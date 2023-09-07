import torch
import pygad.torchga
import pygad
import numpy

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

# Build the PyTorch model using the functional API.
input_layer = torch.nn.Linear(360, 50)
relu_layer = torch.nn.ReLU()
dense_layer = torch.nn.Linear(50, 4)
output_layer = torch.nn.Softmax(1)

model = torch.nn.Sequential(input_layer,
                            relu_layer,
                            dense_layer,
                            output_layer)

# Create an instance of the pygad.torchga.TorchGA class to build the initial population.
torch_ga = pygad.torchga.TorchGA(model=model,
                                 num_solutions=10)

loss_function = torch.nn.CrossEntropyLoss()

# Data inputs
data_inputs = torch.from_numpy(numpy.load("../data/dataset_features.npy")).float()

# Data outputs
data_outputs = torch.from_numpy(numpy.load("../data/outputs.npy")).long()
# The next 2 lines are equivelant to this Keras function to perform 1-hot encoding: tensorflow.keras.utils.to_categorical(data_outputs)
# temp_outs = numpy.zeros((data_outputs.shape[0], numpy.unique(data_outputs).size), dtype=numpy.uint8)
# temp_outs[numpy.arange(data_outputs.shape[0]), numpy.uint8(data_outputs)] = 1

# Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
num_generations = 200 # Number of generations.
num_parents_mating = 5 # Number of solutions to be selected as parents in the mating pool.
initial_population = torch_ga.population_weights # Initial population of network weights.

# Create an instance of the pygad.GA class
ga_instance = pygad.GA(num_generations=num_generations, 
                       num_parents_mating=num_parents_mating, 
                       initial_population=initial_population,
                       fitness_func=fitness_func,
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
# print("Predictions : \n", predictions)

# Calculate the crossentropy loss of the trained model.
print(f"Crossentropy : {loss_function(predictions, data_outputs).detach().numpy()}")

# Calculate the classification accuracy for the trained model.
accuracy = torch.true_divide(torch.sum(torch.max(predictions, axis=1).indices == data_outputs), len(data_outputs))
print(f"Accuracy : {accuracy.detach().numpy()}")
