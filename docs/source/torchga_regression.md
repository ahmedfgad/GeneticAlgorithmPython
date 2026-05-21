# Example 1: Regression Example

The next code builds a simple PyTorch model for regression. The next subsections discuss each part in the code. 

```python
import torch
import torchga
import pygad

def fitness_func(ga_instance, solution, sol_idx):
    global data_inputs, data_outputs, torch_ga, model, loss_function

    predictions = pygad.torchga.predict(model=model, 
                                        solution=solution, 
                                        data=data_inputs)

    abs_error = loss_function(predictions, data_outputs).detach().numpy() + 0.00000001

    solution_fitness = 1.0 / abs_error

    return solution_fitness

def on_generation(ga_instance):
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {ga_instance.best_solution()[1]}")

# Create the PyTorch model.
input_layer = torch.nn.Linear(3, 5)
relu_layer = torch.nn.ReLU()
output_layer = torch.nn.Linear(5, 1)

model = torch.nn.Sequential(input_layer,
                            relu_layer,
                            output_layer)
# print(model)

# Create an instance of the pygad.torchga.TorchGA class to build the initial population.
torch_ga = torchga.TorchGA(model=model,
                           num_solutions=10)

loss_function = torch.nn.L1Loss()

# Data inputs
data_inputs = torch.tensor([[0.02, 0.1, 0.15],
                            [0.7, 0.6, 0.8],
                            [1.5, 1.2, 1.7],
                            [3.2, 2.9, 3.1]])

# Data outputs
data_outputs = torch.tensor([[0.1],
                             [0.6],
                             [1.3],
                             [2.5]])

# Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/pygad.html#pygad-ga-class
num_generations = 250 # Number of generations.
num_parents_mating = 5 # Number of solutions to be selected as parents in the mating pool.
initial_population = torch_ga.population_weights # Initial population of network weights

ga_instance = pygad.GA(num_generations=num_generations, 
                       num_parents_mating=num_parents_mating, 
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       on_generation=on_generation)

ga_instance.run()

# After the generations complete, a plot is shown that summarizes how the fitness values evolve over the generations.
ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Fitness value of the best solution = {solution_fitness}")
print(f"Index of the best solution : {solution_idx}")

# Make predictions based on the best solution.
predictions = pygad.torchga.predict(model=model, 
                                    solution=solution, 
                                    data=data_inputs)
print("Predictions : \n", predictions.detach().numpy())

abs_error = loss_function(predictions, data_outputs)
print("Absolute Error : ", abs_error.detach().numpy())
```

## Create a PyTorch model

According to the steps mentioned previously, the first step is to create a PyTorch model. Here is the code that builds the model using the Functional API.

```python
import torch

input_layer = torch.nn.Linear(3, 5)
relu_layer = torch.nn.ReLU()
output_layer = torch.nn.Linear(5, 1)

model = torch.nn.Sequential(input_layer,
                            relu_layer,
                            output_layer)
```

## Create an Instance of the `pygad.torchga.TorchGA` Class

The second step is to create an instance of the `pygad.torchga.TorchGA` class. There are 10 solutions per population. Change this number according to your needs.

```python
import pygad.torchga

torch_ga = torchga.TorchGA(model=model,
                           num_solutions=10)
```

## Prepare the Training Data

The third step is to prepare the training data inputs and outputs. Here is an example where there are 4 samples. Each sample has 3 inputs and 1 output.

```python
import numpy

# Data inputs
data_inputs = numpy.array([[0.02, 0.1, 0.15],
                           [0.7, 0.6, 0.8],
                           [1.5, 1.2, 1.7],
                           [3.2, 2.9, 3.1]])

# Data outputs
data_outputs = numpy.array([[0.1],
                            [0.6],
                            [1.3],
                            [2.5]])
```

## Build the Fitness Function

The fourth step is to build the fitness function. This function must accept 2 parameters representing the solution and its index within the population.

The next fitness function calculates the mean absolute error (MAE) of the PyTorch model based on the parameters in the solution. The reciprocal of the MAE is used as the fitness value. Feel free to use any other loss function to calculate the fitness value.

```python
loss_function = torch.nn.L1Loss()

def fitness_func(ga_instance, solution, sol_idx):
    global data_inputs, data_outputs, torch_ga, model, loss_function

    predictions = pygad.torchga.predict(model=model, 
                                        solution=solution, 
                                        data=data_inputs)

    abs_error = loss_function(predictions, data_outputs).detach().numpy() + 0.00000001

    solution_fitness = 1.0 / abs_error

    return solution_fitness
```

## Create an Instance of the `pygad.GA` Class

The fifth step is to instantiate the `pygad.GA` class. Note how the `initial_population` parameter is assigned to the initial weights of the PyTorch models.

For more information, please check the [parameters this class accepts](https://pygad.readthedocs.io/en/latest/pygad.html#init).

```python
# Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/pygad.html#pygad-ga-class
num_generations = 250 # Number of generations.
num_parents_mating = 5 # Number of solutions to be selected as parents in the mating pool.
initial_population = torch_ga.population_weights # Initial population of network weights

ga_instance = pygad.GA(num_generations=num_generations, 
                       num_parents_mating=num_parents_mating, 
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       on_generation=on_generation)
```

## Run the Genetic Algorithm

The sixth and last step is to run the genetic algorithm by calling the `run()` method.

```python
ga_instance.run()
```

After PyGAD completes its execution, a figure shows how the fitness value changes by generation. Call the `plot_fitness()` method to show the figure.

```python
ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)
```

Here is the figure.

![PyTorch PyGAD XOR Regression 250 Generations](https://user-images.githubusercontent.com/16560492/103469779-22f5b480-4d37-11eb-80dc-95503065ebb1.png)

To get information about the best solution found by PyGAD, use the `best_solution()` method.

```python
# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Fitness value of the best solution = {solution_fitness}")
print(f"Index of the best solution : {solution_idx}")
```

```python
Fitness value of the best solution = 145.42425295191546
Index of the best solution : 0
```

The next code restores the trained model weights using the `model_weights_as_dict()` function. The restored weights are used to calculate the predicted values.

```python
predictions = pygad.torchga.predict(model=model, 
                                    solution=solution, 
                                    data=data_inputs)
print("Predictions : \n", predictions.detach().numpy())
```

```python
Predictions : 
[[0.08401088]
 [0.60939324]
 [1.3010881 ]
 [2.5010352 ]]
```

The next code measures the trained model error.

```python
abs_error = loss_function(predictions, data_outputs)
print("Absolute Error : ", abs_error.detach().numpy())
```

```
Absolute Error :  0.006876422
```
