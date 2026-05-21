# `pygad.gann` Module

This section of the documentation discusses the **pygad.gann** module. 

The `pygad.gann` module trains neural networks (for either classification or regression) using the genetic algorithm. It makes use of the 2 modules `pygad` and `pygad.nn`.

## `pygad.gann.GANN` Class

The `pygad.gann` module has a class named `pygad.gann.GANN` for training neural networks using the genetic algorithm. The constructor, methods, function, and attributes within the class are discussed in this section. 

### `__init__()`

In order to train a neural network using the genetic algorithm, the first thing to do is to create an instance of the `pygad.gann.GANN` class. 

The `pygad.gann.GANN` class constructor accepts the following parameters:

- `num_solutions`: Number of neural networks (i.e. solutions) in the population. Based on the value passed to this parameter, a number of identical neural networks are created where their parameters are optimized using the genetic algorithm.
- `num_neurons_input`: Number of neurons in the input layer.
- `num_neurons_output`: Number of neurons in the output layer.
- `num_neurons_hidden_layers=[]`: A list holding the number of neurons in the hidden layer(s). If empty `[]`, then no hidden layers are used. For each `int` value it holds, then a hidden layer is created with a number of hidden neurons specified by the corresponding `int` value. For example, `num_neurons_hidden_layers=[10]` creates a single hidden layer with **10** neurons. `num_neurons_hidden_layers=[10, 5]` creates 2 hidden layers with 10 neurons for the first and 5 neurons for the second hidden layer.
- `output_activation="softmax"`: The name of the activation function of the output layer which defaults to `"softmax"`.
- `hidden_activations="relu"`: The name(s) of the activation function(s) of the hidden layer(s). It defaults to `"relu"`. If passed as a string, this means the specified activation function will be used across all the hidden layers. If passed as a list, then it must have the same length as the length of the `num_neurons_hidden_layers` list. An exception is raised if their lengths are different. When `hidden_activations` is a list, a one-to-one mapping between the `num_neurons_hidden_layers` and `hidden_activations` lists occurs.

In order to validate the parameters passed to the `pygad.gann.GANN` class constructor, the `pygad.gann.validate_network_parameters()` function is called.

### Instance Attributes

All the parameters in the `pygad.gann.GANN` class constructor are used as instance attributes. Besides such attributes, there are other attributes added to the instances from the `pygad.gann.GANN` class which are:

- `parameters_validated`: If `True`, then the parameters passed to the GANN class constructor are valid. Its initial value is `False`. 

- `population_networks`: A list holding references to all the solutions (i.e. neural networks) used in the population.

### Methods in the GANN Class

This section discusses the methods available for instances of the `pygad.gann.GANN` class.

#### `create_population()`

The `create_population()` method creates the initial population of the genetic algorithm as a list of neural networks (i.e. solutions). For each network to be created, the `pygad.gann.create_network()` function is called.

Each element in the list holds a reference to the last (i.e. output) layer for the network. The method does not accept any parameter and it accesses all the required details from the `pygad.gann.GANN` instance.

The method returns the list holding the references to the networks. This list is later assigned to the `population_networks` attribute of the instance.

#### `update_population_trained_weights()`

The `update_population_trained_weights()` method updates the `trained_weights` attribute of the layers of each network (check the [documentation of the pygad.nn.DenseLayer class](https://github.com/ahmedfgad/NumPyANN#nndenselayer-class) for more information) according to the weights passed in the `population_trained_weights` parameter.

Accepts the following parameters:

- `population_trained_weights`: A list holding the trained weights of all networks as matrices. Such matrices are to be assigned to the `trained_weights` attribute of all layers of all networks.

## Functions in the `pygad.gann` Module

This section discusses the functions in the `pygad.gann` module.

### `pygad.gann.validate_network_parameters()`
Validates the parameters passed to the constructor of the `pygad.gann.GANN` class. If at least one invalid parameter exists, an exception is raised and execution stops.

The function accepts the same parameters passed to the constructor of the `pygad.gann.GANN` class. Please check the documentation of such parameters in the section discussing the class constructor.

The reason why this function sets a default value to the `num_solutions` parameter is differentiating whether a population of networks or a single network is to be created. If `None`, then a single network will be created. If not `None`, then a population of networks is to be created.

If the value passed to the `hidden_activations` parameter is a string, not a list, then a list is created by replicating the passed name of the activation function a number of times equal to the number of hidden layers (i.e. the length of the `num_neurons_hidden_layers` parameter). 

Returns a list holding the name(s) of the activation function(s) of the hidden layer(s).

### `pygad.gann.create_network()`

Creates a neural network as a linked list between the input, hidden, and output layers, where the layer at index N (the last/output layer) references the layer at index N-1 (a hidden layer) using its `previous_layer` attribute. The input layer does not reference any layer because it is the first layer in the linked list.

In addition to the `parameters_validated` parameter, this function accepts the same parameters passed to the constructor of the `pygad.gann.GANN` class except for the `num_solutions` parameter because only a single network is created out of the `create_network()` function.

`parameters_validated`: If `False`, then the parameters are not validated and a call to the `validate_network_parameters()` function is made.

Returns the reference to the last layer in the network architecture which is the output layer. Based on such a reference, all network layers can be fetched.

### `pygad.gann.population_as_vectors()`    

Accepts the population as networks and returns a list holding all weights of the layers of each solution (i.e. network) in the population as a vector. 

For example, if the population has 6 solutions (i.e. networks), this function accepts references to such networks and returns a list with 6 vectors, one for each network (i.e. solution). Each vector holds the weights for all layers for a single network.

Accepts the following parameters:

- `population_networks`: A list holding references to the output (last) layers of the neural networks used in the population. 

Returns a list holding the weights vectors for all solutions (i.e. networks).

### `pygad.gann.population_as_matrices()`

Accepts the population as both networks and weights vectors and returns the weights of all layers of each solution (i.e. network) in the population as a matrix.

For example, if the population has 6 solutions (i.e. networks), this function returns a list with 6 matrices, one for each network holding its weights for all layers.

Accepts the following parameters:

- `population_networks`: A list holding references to the output (last) layers of the neural networks used in the population. 
- `population_vectors`: A list holding the weights of all networks as vectors. Such vectors are to be converted into matrices.

Returns a list holding the weights matrices for all solutions (i.e. networks).

## Steps to Build and Train Neural Networks using Genetic Algorithm

The steps to use this project for building and training a neural network using the genetic algorithm are as follows:

- Prepare the training data.
- Create an instance of the `pygad.gann.GANN` class.
- Fetch the population weights as vectors.
- Prepare the fitness function.
- Prepare the generation callback function.
- Create an instance of the `pygad.GA` class.
- Run the created instance of the `pygad.GA` class.
- Plot the fitness values.
- Get information about the best solution.
- Make predictions using the trained weights.
- Calculate some statistics.

The next sections cover each step.

### Prepare the Training Data

Before building and training neural networks, the training data (input and output) is to be prepared. The inputs and the outputs of the training data are NumPy arrays. 

Here is an example of preparing the training data for the XOR problem.

For the input array, each element must be a list representing the inputs (i.e. features) for the sample. If there are 200 samples and each sample has 50 features, then the shape of the inputs array is `(200, 50)`. The variable `num_inputs` holds the length of each sample which is 2 in this example.

```python
data_inputs = numpy.array([[1, 1],
                           [1, 0],
                           [0, 1],
                           [0, 0]])

data_outputs = numpy.array([0, 
                            1, 
                            1, 
                            0])

num_inputs = data_inputs.shape[1]
```

For the output array, each element must be a single number representing the class label of the sample. The class labels must start at `0`. So, if there are 200 samples, then the shape of the output array is `(200)`. If there are 5 classes in the data, then the values of all the 200 elements in the output array must range from 0 to 4 inclusive. Generally, the class labels start from `0` to `N-1` where `N` is the number of classes.

For the XOR example, there are 2 classes and thus their labels are 0 and 1. The `num_classes` variable is assigned to 2.

Note that the project only supports classification problems where each sample is assigned to only one class.

### Create an Instance of the `pygad.gann.GANN` Class

After preparing the input data, an instance of the `pygad.gann.GANN` class is created by passing the appropriate parameters.

Here is an example that creates a network for the XOR problem. The `num_solutions` parameter is set to 6 which means the genetic algorithm population will have 6 solutions (i.e. networks). All of these 6 neural networks will have the same architectures as specified by the other parameters. 

The output layer has 2 neurons because there are only 2 classes (0 and 1).

```python
import pygad.gann
import pygad.nn

num_solutions = 6
GANN_instance = pygad.gann.GANN(num_solutions=num_solutions,
                                num_neurons_input=num_inputs,
                                num_neurons_hidden_layers=[2],
                                num_neurons_output=2,
                                hidden_activations=["relu"],
                                output_activation="softmax")
```

The architecture of the created network has the following layers:

- An input layer with 2 neurons (i.e. inputs)
- A single hidden layer with 2 neurons.
- An output layer with 2 neurons (i.e. classes). 

The weights of the network are as follows:

- Between the input and the hidden layer, there is a weights matrix of size equal to `(number inputs x number of hidden neurons) = (2x2)`.
- Between the hidden and the output layer, there is a weights matrix of size equal to `(number of hidden neurons x number of outputs) = (2x2)`.

The activation function used for the output layer is `softmax`. The `relu` activation function is used for the hidden layer.

After creating the instance of the `pygad.gann.GANN` class, the next step is to fetch the weights of the population as a list of vectors.

### Fetch the Population Weights as Vectors

For the genetic algorithm, the parameters (i.e. genes) of each solution are represented as a single vector. 

For the task of training the network for the XOR problem, the weights of each network in the population are not represented as a vector but 2 matrices each of size 2x2.

To create a list holding the population weights as vectors, one for each network, the `pygad.gann.population_as_vectors()` function is used.

```python
population_vectors = pygad.gann.population_as_vectors(population_networks=GANN_instance.population_networks)
```

After preparing the population weights as a set of vectors, the next step is to prepare 2 functions:

1. Fitness function.
2. Callback function after each generation.

### Prepare the Fitness Function

The PyGAD library works by allowing the users to customize the genetic algorithm for their own problems. Because the problems differ in how the fitness values are calculated, then PyGAD allows the user to use a custom function as a maximization fitness function. This function must accept 2 positional parameters representing the following:

- The solution.
- The solution index in the population.

The fitness function must return a single number representing the fitness. The higher the fitness value, the better the solution.

Here is the implementation of the fitness function for training a neural network. It uses the `pygad.nn.predict()` function to predict the class labels based on the current solution's weights. The `pygad.nn.predict()` function uses the trained weights available in the `trained_weights` attribute of each layer of the network for making predictions.

Based on such predictions, the classification accuracy is calculated. This accuracy is used as the fitness value of the solution. Finally, the fitness value is returned.

```python
def fitness_func(ga_instance, solution, sol_idx):
    global GANN_instance, data_inputs, data_outputs

    predictions = pygad.nn.predict(last_layer=GANN_instance.population_networks[sol_idx],
                                   data_inputs=data_inputs)
    correct_predictions = numpy.where(predictions == data_outputs)[0].size
    solution_fitness = (correct_predictions/data_outputs.size)*100

    return solution_fitness
```

### Prepare the Generation Callback Function

After each generation of the genetic algorithm, the fitness function will be called to calculate the fitness value of each solution. Within the fitness function, the `pygad.nn.predict()` function is used for predicting the outputs based on the current solution's `trained_weights` attribute. Thus, it is required that such an attribute is updated by weights evolved by the genetic algorithm after each generation. 

PyGAD 2.0.0 and higher has a new parameter accepted by the `pygad.GA` class constructor named `on_generation`. It could be assigned to a function that is called after each generation. The function must accept a single parameter representing the instance of the `pygad.GA` class. 

This callback function can be used to update the `trained_weights` attribute of layers of each network in the population. 

Here is the implementation for a function that updates the `trained_weights` attribute of the layers of the population networks. 

It works by converting the current population from the vector form to the matrix form using the `pygad.gann.population_as_matrices()` function. It accepts the population as vectors and returns it as matrices.

The population matrices are then passed to the `update_population_trained_weights()` method in the `pygad.gann` module to update the `trained_weights` attribute of all layers for all solutions within the population.

```python
def callback_generation(ga_instance):
    global GANN_instance

    population_matrices = pygad.gann.population_as_matrices(population_networks=GANN_instance.population_networks, population_vectors=ga_instance.population)
    GANN_instance.update_population_trained_weights(population_trained_weights=population_matrices)

    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {ga_instance.best_solution()[1]}")
```

After preparing the fitness and callback functions, the next step is to create an instance of the `pygad.GA` class.

### Create an Instance of the `pygad.GA` Class

Once the parameters of the genetic algorithm are prepared, an instance of the `pygad.GA` class can be created. 

Here is an example.

```python
initial_population = population_vectors.copy()

num_parents_mating = 4 

num_generations = 500

mutation_percent_genes = 5

parent_selection_type = "sss"

crossover_type = "single_point"

mutation_type = "random"

keep_parents = 1

init_range_low = -2
init_range_high = 5

ga_instance = pygad.GA(num_generations=num_generations, 
                       num_parents_mating=num_parents_mating, 
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       mutation_percent_genes=mutation_percent_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       keep_parents=keep_parents,
                       on_generation=callback_generation)
```

The last step for training the neural networks using the genetic algorithm is calling the `run()` method.

### Run the Created Instance of the `pygad.GA` Class

By calling the `run()` method from the `pygad.GA` instance, the genetic algorithm will iterate through the number of generations specified in its `num_generations` parameter. 

```python
ga_instance.run()
```

### Plot the Fitness Values

After the `run()` method completes, the `plot_fitness()` method can be called to show how the fitness values evolve by generation. A fitness value (i.e. accuracy) of 100 is reached after around 180 generations. 

```python
ga_instance.plot_fitness()
```

![XOR_Fitness](https://user-images.githubusercontent.com/16560492/82078638-c11e0700-96e1-11ea-8aa9-c36761c5e9c7.png)

By running the code again, a different initial population is created, so a classification accuracy of 100 can be reached using fewer generations. On the other hand, a different initial population might cause 100% accuracy to be reached using more generations or not reached at all.

### Information about the Best Solution

The following information about the best solution in the last population is returned using the `best_solution()` method in the `pygad.GA` class. 

- Solution
- Fitness value of the solution
- Index of the solution within the population

Here is how such information is returned. The fitness value (i.e. accuracy) is 100.

```python
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Parameters of the best solution : {solution}")
print(f"Fitness value of the best solution = {solution_fitness}")
print(f"Index of the best solution : {solution_idx}")
```

```
Parameters of the best solution : [3.55081391 -3.21562011 -14.2617784 0.68044231 -1.41258145 -3.2979315 1.58136006 -7.83726169]
Fitness value of the best solution = 100.0
Index of the best solution : 0
```

Using the `best_solution_generation` attribute of the `pygad.GA` instance, you can get the generation number at which the best fitness was reached. In this run, the best fitness value is reached after 182 generations.

```python
if ga_instance.best_solution_generation != -1:
    print(f"Best fitness value reached after {ga_instance.best_solution_generation} generations.")
```

```
Best solution reached after 182 generations.
```

### Making Predictions using the Trained Weights

The `pygad.nn.predict()` function can be used to make predictions using the trained network. As printed, the network is able to predict the labels correctly. 

```python
predictions = pygad.nn.predict(last_layer=GANN_instance.population_networks[solution_idx], data_inputs=data_inputs)
print(f"Predictions of the trained network : {predictions}")
```

```
Predictions of the trained network : [0. 1. 1. 0.]
```

### Calculating Some Statistics

Based on the predictions the network made, some statistics can be calculated such as the number of correct and wrong predictions in addition to the classification accuracy.

```python
num_wrong = numpy.where(predictions != data_outputs)[0]
num_correct = data_outputs.size - num_wrong.size
accuracy = 100 * (num_correct/data_outputs.size)
print(f"Number of correct classifications : {num_correct}.")
print(f"Number of wrong classifications : {num_wrong.size}.")
print(f"Classification accuracy : {accuracy}.")
```

```
Number of correct classifications : 4.
Number of wrong classifications : 0.
Classification accuracy : 100.0.
```

## Examples

This section gives the complete code of some examples that build and train neural networks using the genetic algorithm. Each subsection builds a different network.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} XOR Classification
:link: gann_xor
:link-type: doc
:::

:::{grid-item-card} Image Classification
:link: gann_image_classification
:link-type: doc
:::

:::{grid-item-card} Regression Example 1
:link: gann_regression_1
:link-type: doc
:::

:::{grid-item-card} Regression Example 2 - Fish Weight Prediction
:link: gann_regression_2
:link-type: doc
:::

::::

:::{toctree}
:hidden:

gann_xor
gann_image_classification
gann_regression_1
gann_regression_2
:::
