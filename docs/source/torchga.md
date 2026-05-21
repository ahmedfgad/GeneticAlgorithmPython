# `pygad.torchga` Module

This section of the documentation discusses the **pygad.torchga** module. 

The `pygad.torchga` module has a helper class and 2 functions to train PyTorch models using the genetic algorithm (PyGAD). 

The contents of this module are:

1. `TorchGA`: A class for creating an initial population of all parameters in the PyTorch model.
2. `model_weights_as_vector()`: A function to reshape the PyTorch model weights to a single vector.
3. `model_weights_as_dict()`: A function to restore the PyTorch model weights from a vector.
4. `predict()`: A function to make predictions based on the PyTorch model and a solution.

More details are given in the next sections.

## Steps Summary

The steps used to train a PyTorch model using PyGAD are summarized as follows:

1. Create a PyTorch model.
2. Create an instance of the `pygad.torchga.TorchGA` class.
3. Prepare the training data.
4. Build the fitness function.
5. Create an instance of the `pygad.GA` class.
6. Run the genetic algorithm.

## Create PyTorch Model

Before discussing training a PyTorch model using PyGAD, the first thing to do is to create the PyTorch model. To get started, please check the [PyTorch library documentation](https://pytorch.org/docs/stable/index.html).

Here is an example of a PyTorch model.

```python
import torch

input_layer = torch.nn.Linear(3, 5)
relu_layer = torch.nn.ReLU()
output_layer = torch.nn.Linear(5, 1)

model = torch.nn.Sequential(input_layer,
                            relu_layer,
                            output_layer)
```

Feel free to add the layers of your choice.

## `pygad.torchga.TorchGA` Class

The `pygad.torchga` module has a class named `TorchGA` for creating an initial population for the genetic algorithm based on a PyTorch model. The constructor, methods, and attributes within the class are discussed in this section. 

### `__init__()`

The `pygad.torchga.TorchGA` class constructor accepts the following parameters:

- `model`: An instance of the PyTorch model.
- `num_solutions`: Number of solutions in the population. Each solution has different parameters of the model. 

### Instance Attributes

All parameters in the `pygad.torchga.TorchGA` class constructor are used as instance attributes in addition to adding a new attribute called `population_weights`. 

Here is a list of all instance attributes:

- `model`
- `num_solutions`
- `population_weights`: A nested list holding the weights of all solutions in the population.

### Methods in the `TorchGA` Class

This section discusses the methods available for instances of the `pygad.torchga.TorchGA` class.

#### `create_population()`

The `create_population()` method creates the initial population of the genetic algorithm as a list of solutions where each solution represents different model parameters. The list of networks is assigned to the `population_weights` attribute of the instance.

## Functions in the `pygad.torchga` Module

This section discusses the functions in the `pygad.torchga` module.

### `pygad.torchga.model_weights_as_vector()`    

The `model_weights_as_vector()` function accepts a single parameter named `model` representing the PyTorch model. It returns a vector holding all model weights. The reason for representing the model weights as a vector is that the genetic algorithm expects all parameters of any solution to be in a 1D vector form.

The function accepts the following parameters:

- `model`: The PyTorch model. 

It returns a 1D vector holding the model weights.

### `pygad.torchga.model_weights_as_dict()`

The `model_weights_as_dict()` function accepts the following parameters: 

1. `model`: The PyTorch model.
2. `weights_vector`: The model parameters as a vector.

It returns the restored model weights in the same form used by the `state_dict()` method. The returned dictionary is ready to be passed to the `load_state_dict()` method for setting the PyTorch model's parameters.

### `pygad.torchga.predict()`

The `predict()` function makes a prediction based on a solution. It accepts the following parameters:

1. `model`: The PyTorch model.
2. `solution`: The solution evolved.
3. `data`: The test data inputs.

It returns the predictions for the data samples.

## Examples

This section gives the complete code of some examples that build and train a PyTorch model using PyGAD. Each subsection builds a different network.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Example 1: Regression Example
:link: torchga_regression
:link-type: doc
:::

:::{grid-item-card} Example 2: XOR Binary Classification
:link: torchga_xor
:link-type: doc
:::

:::{grid-item-card} Example 3: Image Multi-Class Classification (Dense Layers)
:link: torchga_image_dense
:link-type: doc
:::

:::{grid-item-card} Example 4: Image Multi-Class Classification (Conv Layers)
:link: torchga_image_conv
:link-type: doc
:::

::::

:::{toctree}
:hidden:

torchga_regression
torchga_xor
torchga_image_dense
torchga_image_conv
:::
