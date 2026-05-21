# `pygad.kerasga` Module

This section of the documentation discusses the [**pygad.kerasga**](https://pygad.readthedocs.io/en/latest/kerasga.html) module. 

The `pygad.kerasga` module has a helper class and 2 functions to train Keras models using the genetic algorithm (PyGAD). The Keras model can be built using either the [Sequential Model](https://keras.io/guides/sequential_model) or the [Functional API](https://keras.io/guides/functional_api).

The contents of this module are:

1. `KerasGA`: A class for creating an initial population of all parameters in the Keras model.
2. `model_weights_as_vector()`: A function to reshape the Keras model weights to a single vector.
3. `model_weights_as_matrix()`: A function to restore the Keras model weights from a vector.
4. `predict()`: A function to make predictions based on the Keras model and a solution.

More details are given in the next sections.

## Steps Summary

The steps used to train a Keras model using PyGAD are summarized as follows:

1. Create a Keras model.
2. Create an instance of the `pygad.kerasga.KerasGA` class.
3. Prepare the training data.
4. Build the fitness function.
5. Create an instance of the `pygad.GA` class.
6. Run the genetic algorithm.

## Create Keras Model

Before discussing training a Keras model using PyGAD, the first thing to do is to create the Keras model. 

According to the [Keras library documentation](https://keras.io/api/models), there are 3 ways to build a Keras model:

1. [Sequential Model](https://keras.io/guides/sequential_model)

2. [Functional API](https://keras.io/guides/functional_api)

3. [Model Subclassing](https://keras.io/guides/model_subclassing)

PyGAD supports training the models created either using the Sequential Model or the Functional API.

Here is an example of a model created using the Sequential Model.

```python
import tensorflow.keras

input_layer  = tensorflow.keras.layers.Input(3)
dense_layer1 = tensorflow.keras.layers.Dense(5, activation="relu")
output_layer = tensorflow.keras.layers.Dense(1, activation="linear")

model = tensorflow.keras.Sequential()
model.add(input_layer)
model.add(dense_layer1)
model.add(output_layer)
```

This is the same model created using the Functional API.

```python
input_layer  = tensorflow.keras.layers.Input(3)
dense_layer1 = tensorflow.keras.layers.Dense(5, activation="relu")(input_layer)
output_layer = tensorflow.keras.layers.Dense(1, activation="linear")(dense_layer1)

model = tensorflow.keras.Model(inputs=input_layer, outputs=output_layer)
```

Feel free to add the layers of your choice.

## `pygad.kerasga.KerasGA` Class

The `pygad.kerasga` module has a class named `KerasGA` for creating an initial population for the genetic algorithm based on a Keras model. The constructor, methods, and attributes within the class are discussed in this section. 

### `__init__()`

The `pygad.kerasga.KerasGA` class constructor accepts the following parameters:

- `model`: An instance of the Keras model.
- `num_solutions`: Number of solutions in the population. Each solution has different parameters of the model. 

### Instance Attributes

All parameters in the `pygad.kerasga.KerasGA` class constructor are used as instance attributes in addition to adding a new attribute called `population_weights`. 

Here is a list of all instance attributes:

- `model`
- `num_solutions`
- `population_weights`: A nested list holding the weights of all solutions in the population.

### Methods in the `KerasGA` Class

This section discusses the methods available for instances of the `pygad.kerasga.KerasGA` class.

#### `create_population()`

The `create_population()` method creates the initial population of the genetic algorithm as a list of solutions where each solution represents different model parameters. The list of networks is assigned to the `population_weights` attribute of the instance.

## Functions in the `pygad.kerasga` Module

This section discusses the functions in the `pygad.kerasga` module.

### `pygad.kerasga.model_weights_as_vector()`    

The `model_weights_as_vector()` function accepts a single parameter named `model` representing the Keras model. It returns a vector holding all model weights. The reason for representing the model weights as a vector is that the genetic algorithm expects all parameters of any solution to be in a 1D vector form.

This function filters the layers based on the `trainable` attribute to see whether the layer weights are trained or not. For each layer, if its `trainable=False`, then its weights will not be evolved using the genetic algorithm. Otherwise, it will be represented in the chromosome and evolved.

The function accepts the following parameters:

- `model`: The Keras model. 

It returns a 1D vector holding the model weights. 

### `pygad.kerasga.model_weights_as_matrix()`

The `model_weights_as_matrix()` function accepts the following parameters: 

1. `model`: The Keras model.
2. `weights_vector`: The model parameters as a vector.

It returns the restored model weights after reshaping the vector.

### `pygad.kerasga.predict()`

The `predict()` function makes a prediction based on a solution. It accepts the following parameters:

1. `model`: The Keras model.
2. `solution`: The solution evolved.
3. `data`: The test data inputs.
4. `batch_size=None`: The batch size (i.e. number of samples per step or batch).
5. `verbose=None`: Verbosity mode.
6. `steps=None`: The total number of steps (batches of samples).

Check documentation of the [Keras Model.predict()](https://keras.io/api/models/model_training_apis) method for more information about the `batch_size`, `verbose`, and `steps` parameters. 

It returns the predictions of the data samples.

## Examples

This section gives the complete code of some examples that build and train a Keras model using PyGAD. Each subsection builds a different network.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Example 1: Regression Example
:link: kerasga_regression
:link-type: doc
:::

:::{grid-item-card} Example 2: XOR Binary Classification
:link: kerasga_xor
:link-type: doc
:::

:::{grid-item-card} Example 3: Image Multi-Class Classification (Dense Layers)
:link: kerasga_image_dense
:link-type: doc
:::

:::{grid-item-card} Example 4: Image Multi-Class Classification (Conv Layers)
:link: kerasga_image_conv
:link-type: doc
:::

:::{grid-item-card} Example 5: Image Classification using Data Generator
:link: kerasga_image_datagen
:link-type: doc
:::

::::

:::{toctree}
:hidden:

kerasga_regression
kerasga_xor
kerasga_image_dense
kerasga_image_conv
kerasga_image_datagen
:::
