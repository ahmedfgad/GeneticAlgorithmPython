import numpy
import pygad.cnn
import pygad.gacnn
import pygad

"""
Convolutional neural network implementation using NumPy
A tutorial that helps to get started (Building Convolutional Neural Network using NumPy from Scratch) available in these links: 
    https://www.linkedin.com/pulse/building-convolutional-neural-network-using-numpy-from-ahmed-gad
    https://towardsdatascience.com/building-convolutional-neural-network-using-numpy-from-scratch-b30aac50e50a
    https://www.kdnuggets.com/2018/04/building-convolutional-neural-network-numpy-scratch.html
It is also translated into Chinese: http://m.aliyun.com/yunqi/articles/585741
"""

def fitness_func(ga_instance, solution, sol_idx):
    global GACNN_instance, data_inputs, data_outputs

    predictions = GACNN_instance.population_networks[sol_idx].predict(data_inputs=data_inputs)
    correct_predictions = numpy.where(predictions == data_outputs)[0].size
    solution_fitness = (correct_predictions/data_outputs.size)*100

    return solution_fitness

def callback_generation(ga_instance):
    global GACNN_instance, last_fitness

    population_matrices = pygad.gacnn.population_as_matrices(population_networks=GACNN_instance.population_networks, 
                                                       population_vectors=ga_instance.population)

    GACNN_instance.update_population_trained_weights(population_trained_weights=population_matrices)

    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solutions_fitness))

data_inputs = numpy.load("../data/dataset_inputs.npy")
data_outputs = numpy.load("../data/dataset_outputs.npy")

sample_shape = data_inputs.shape[1:]
num_classes = 4

data_inputs = data_inputs
data_outputs = data_outputs

input_layer = pygad.cnn.Input2D(input_shape=sample_shape)
conv_layer1 = pygad.cnn.Conv2D(num_filters=2,
                               kernel_size=3,
                               previous_layer=input_layer,
                               activation_function="relu")
average_pooling_layer = pygad.cnn.AveragePooling2D(pool_size=5, 
                                                   previous_layer=conv_layer1,
                                                   stride=3)

flatten_layer = pygad.cnn.Flatten(previous_layer=average_pooling_layer)
dense_layer2 = pygad.cnn.Dense(num_neurons=num_classes, 
                               previous_layer=flatten_layer,
                               activation_function="softmax")

model = pygad.cnn.Model(last_layer=dense_layer2,
                        epochs=1,
                        learning_rate=0.01)

model.summary()


GACNN_instance = pygad.gacnn.GACNN(model=model,
                             num_solutions=4)

# GACNN_instance.update_population_trained_weights(population_trained_weights=population_matrices)

# population does not hold the numerical weights of the network instead it holds a list of references to each last layer of each network (i.e. solution) in the population. A solution or a network can be used interchangeably.
# If there is a population with 3 solutions (i.e. networks), then the population is a list with 3 elements. Each element is a reference to the last layer of each network. Using such a reference, all details of the network can be accessed.
population_vectors = pygad.gacnn.population_as_vectors(population_networks=GACNN_instance.population_networks)

# To prepare the initial population, there are 2 ways:
# 1) Prepare it yourself and pass it to the initial_population parameter. This way is useful when the user wants to start the genetic algorithm with a custom initial population.
# 2) Assign valid integer values to the sol_per_pop and num_genes parameters. If the initial_population parameter exists, then the sol_per_pop and num_genes parameters are useless.
initial_population = population_vectors.copy()

num_parents_mating = 2 # Number of solutions to be selected as parents in the mating pool.

num_generations = 10 # Number of generations.

mutation_percent_genes = 0.1 # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists.

parent_selection_type = "sss" # Type of parent selection.

crossover_type = "single_point" # Type of the crossover operator.

mutation_type = "random" # Type of the mutation operator.

keep_parents = -1 # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.

ga_instance = pygad.GA(num_generations=num_generations, 
                       num_parents_mating=num_parents_mating, 
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       mutation_percent_genes=mutation_percent_genes,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       keep_parents=keep_parents,
                       on_generation=callback_generation)

ga_instance.run()

# After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
ga_instance.plot_fitness()

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
print(f"Parameters of the best solution : {solution}")
print(f"Fitness value of the best solution = {solution_fitness}")
print(f"Index of the best solution : {solution_idx}")

if ga_instance.best_solution_generation != -1:
    print(f"Best fitness value reached after {ga_instance.best_solution_generation} generations.")

# Predicting the outputs of the data using the best solution.
predictions = GACNN_instance.population_networks[solution_idx].predict(data_inputs=data_inputs)
print("Predictions of the trained network : {predictions}")

# Calculating some statistics
num_wrong = numpy.where(predictions != data_outputs)[0]
num_correct = data_outputs.size - num_wrong.size
accuracy = 100 * (num_correct/data_outputs.size)
print(f"Number of correct classifications : {num_correct}.")
print(f"Number of wrong classifications : {num_wrong.size}.")
print(f"Classification accuracy : {accuracy}.")
