# Example 2: XOR Binary Classification

The next code creates a PyTorch model to build the XOR binary classification problem. Let's highlight the changes compared to the previous example. 

```python
import torch
import torchga
import pygad

def fitness_func(ga_instance, solution, sol_idx):
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
torch_ga = torchga.TorchGA(model=model,
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

# Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/pygad.html#pygad-ga-class
num_generations = 250 # Number of generations.
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

# Calculate the binary crossentropy for the trained model.
print("Binary Crossentropy : ", loss_function(predictions, data_outputs).detach().numpy())

# Calculate the classification accuracy of the trained model.
a = torch.max(predictions, axis=1)
b = torch.max(data_outputs, axis=1)
accuracy = torch.sum(a.indices == b.indices) / len(data_outputs)
print("Accuracy : ", accuracy.detach().numpy())
```

Compared to the previous regression example, here are the changes:

* The PyTorch model is changed according to the nature of the problem. Now, it has 2 inputs and 2 outputs with an in-between hidden layer of 4 neurons.

```python
input_layer  = torch.nn.Linear(2, 4)
relu_layer = torch.nn.ReLU()
dense_layer = torch.nn.Linear(4, 2)
output_layer = torch.nn.Softmax(1)

model = torch.nn.Sequential(input_layer,
                            relu_layer,
                            dense_layer,
                            output_layer)
```

* The train data is changed. Note that the output of each sample is a 1D vector of 2 values, 1 for each class.

```python
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
```

* The fitness value is calculated based on the binary cross entropy.

```python
loss_function = torch.nn.BCELoss()
```

After the previous code completes, the next figure shows how the fitness value change by generation.

![PyTorch PyGAD XOR Classification 250 Generations](https://user-images.githubusercontent.com/16560492/103469818-c646c980-4d37-11eb-98c3-d9d591acd5e2.png)

Here is some information about the trained model. Its fitness value is `100000000.0`, loss is `0.0` and accuracy is 100%.

```python
Fitness value of the best solution = 100000000.0

Index of the best solution : 0

Predictions : 
[[1.0000000e+00 1.3627675e-10]
 [3.8521746e-09 1.0000000e+00]
 [4.2789325e-10 1.0000000e+00]
 [1.0000000e+00 3.3668417e-09]]

Binary Crossentropy :  0.0

Accuracy :  1.0
```
