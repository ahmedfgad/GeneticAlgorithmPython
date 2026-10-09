# `pygad.kerasga` Module

This section of the documentation discusses the [**pygad.kerasga**](https://pygad.readthedocs.io/en/latest/kerasga.html) module. 

The `pygad.kerasga` module has a helper class and 3 public functions to train Keras models using the genetic algorithm (PyGAD). The Keras model can be built using either the [Sequential Model](https://keras.io/guides/sequential_model) or the [Functional API](https://keras.io/guides/functional_api).

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

- `model`: The Keras model passed to the constructor.
- `num_solutions`: Number of chromosomes to create.
- `population_weights`: A list of one-dimensional NumPy arrays, one trainable-weight vector per solution. The first vector contains the model's current weights; each subsequent vector adds independent uniform noise in `[-1, 1]` to those weights. Non-trainable layers are excluded from the chromosome.

### Methods in the `KerasGA` Class

This section discusses the methods available for instances of the `pygad.kerasga.KerasGA` class.

#### `create_population()`

`create_population()` accepts no arguments and returns a new list of one-dimensional NumPy weight vectors. The constructor assigns this list to `population_weights`. Calling the method later returns a new population without updating that attribute; assign its result explicitly to replace the stored population. The model's weights are read without changing them.

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

(keras-predict)=
### `pygad.kerasga.predict()`

The `predict()` function makes a prediction based on a solution. It accepts the following parameters:

1. `model`: The Keras model.
2. `solution`: The solution evolved.
3. `data`: Input data accepted by the selected Keras execution path below.
4. `batch_size=None`: The batch size (i.e. number of samples per step or batch).
5. `verbose=0`: Verbosity mode.
6. `steps=None`: The total number of steps (batches of samples).

Check documentation of the [Keras Model.predict()](https://keras.io/api/models/model_training_apis) method for more information about the `batch_size`, `verbose`, and `steps` parameters. 

When `batch_size` and `steps` are both `None`, the helper calls `model(data, training=False)` directly and converts its output to a NumPy array. This path avoids `Model.predict()` overhead; `verbose` is not used. Supply inputs supported by the model's direct call, such as NumPy arrays or tensors.

If either `batch_size` or `steps` is specified, the helper calls `model.predict(x=data, batch_size=batch_size, verbose=verbose, steps=steps)`. In this path, the inputs and options follow Keras `Model.predict()` semantics, including dataset inputs. The helper returns the predictions produced by the selected path.

The model's original weights are restored after prediction, including when prediction raises an exception. Calls to `pygad.kerasga.predict()` sharing the same model are synchronized across threads so one solution cannot overwrite another solution's weights during evaluation. This makes shared-model predictions run one at a time; use separate models per worker when concurrent predictions are needed. Other code that directly calls `model.set_weights()` must manage its own synchronization.

The protected operation includes weight conversion, loading the solution, inference, and restoring original weights. Exceptions from weight conversion or inference propagate to the caller; when inference fails, the `finally` block still attempts to restore the original weights. This synchronization does not make every operation on a Keras model thread-safe.

### Internal Synchronization Helpers

These names are defined in `pygad.kerasga.kerasga`, rather than exported as public helpers by `pygad.kerasga`. They are documented for completeness and are not stable API.

- `_model_lock(model)`: Returns the `threading.RLock` associated with the passed model, creating it on first use. `predict()` holds this lock throughout its temporary weight changes and prediction.
- `_model_locks`: Module-level `weakref.WeakKeyDictionary` mapping models to locks. Weak keys allow a model's entry to be released when the model is no longer referenced. Locks are stored outside model state so this mechanism does not add a lock to serialized model attributes.
- `_model_locks_guard`: Module-level `threading.Lock` protecting lookup and creation of entries in `_model_locks`.

These objects are local to a Python process. They are module attributes, not additional `KerasGA` instance attributes, and callers should use `predict()` rather than access the registry directly.

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
