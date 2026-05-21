# `pygad.nn` Module

This section of the documentation discusses the **pygad.nn** module.

Using the **pygad.nn** module, artificial neural networks are created. The purpose of this module is to only implement the **forward pass** of a neural network without using a training algorithm. The **pygad.nn** module builds the network layers, implements the activation functions, trains the network, makes predictions, and more.

Later, the **pygad.gann** module is used to train the **pygad.nn** network using the genetic algorithm built in the **pygad** module.

Starting from [PyGAD 2.7.1](https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-7-1), the **pygad.nn** module supports both classification and regression problems. For more information, check the `problem_type` parameter in the `pygad.nn.train()` and `pygad.nn.predict()` functions.

## Supported Layers

Each layer supported by the **pygad.nn** module has a corresponding class. The layers and their classes are:

1. **Input**: Implemented using the `pygad.nn.InputLayer` class.

2. **Dense** (Fully Connected): Implemented using the `pygad.nn.DenseLayer` class.

In the future, more layers will be added. The next subsections discuss such layers.

### `pygad.nn.InputLayer` Class

The `pygad.nn.InputLayer` class creates the input layer for the neural network. For each network, there is only a single input layer. The network architecture must start with an input layer.

This class has no methods or class attributes. All it has is a constructor that accepts a parameter named `num_neurons` representing the number of neurons in the input layer. 

An instance attribute named `num_neurons` is created within the constructor to keep such a number. Here is an example of building an input layer with 20 neurons.

```python
input_layer = pygad.nn.InputLayer(num_neurons=20)
```

Here is how the single attribute `num_neurons` within the instance of the `pygad.nn.InputLayer` class can be accessed.

```python
num_input_neurons = input_layer.num_neurons

print("Number of input neurons =", num_input_neurons)
```

This is everything about the input layer.

### `pygad.nn.DenseLayer` Class

Using the `pygad.nn.DenseLayer` class, dense (fully-connected) layers can be created. To create a dense layer, just create a new instance of the class. The constructor accepts the following parameters:

- `num_neurons`: Number of neurons in the dense layer.
- `previous_layer`: A reference to the previous layer. Using the `previous_layer` attribute, a linked list is created that connects all network layers.
- `activation_function`: A string representing the activation function to be used in this layer. Defaults to `"sigmoid"`. Currently, the supported values for the activation functions are `"sigmoid"`,  `"relu"`, `"softmax"` (supported in PyGAD 2.3.0 and higher), and `"None"` (supported in PyGAD 2.7.0 and higher). When a layer has its activation function set to `"None"`, then it means no activation function is applied. For a **regression problem**, set the activation function of the output (last) layer to `"None"`. If all outputs in the regression problem are nonnegative, then it is possible to use the ReLU function in the output layer.

Within the constructor, the accepted parameters are used as instance attributes. Besides the parameters, some new instance attributes are created which are:

- `initial_weights`: The initial weights for the dense layer.
- `trained_weights`: The trained weights of the dense layer. This attribute is initialized by the value in the `initial_weights` attribute.

Here is an example for creating a dense layer with 12 neurons. Note that the `previous_layer` parameter is assigned to the input layer `input_layer`. 

```python
dense_layer = pygad.nn.DenseLayer(num_neurons=12,
                                  previous_layer=input_layer,
                                  activation_function="relu")
```

Here is how to access some attributes in the dense layer:

```python
num_dense_neurons = dense_layer.num_neurons
dense_initail_weights = dense_layer.initial_weights

print("Number of dense layer attributes =", num_dense_neurons)
print("Initial weights of the dense layer :", dense_initail_weights)
```

Because `dense_layer` holds a reference to the input layer, then the number of input neurons can be accessed.

```python
input_layer = dense_layer.previous_layer
num_input_neurons = input_layer.num_neurons

print("Number of input neurons =", num_input_neurons)
```

Here is another dense layer. This dense layer's `previous_layer` attribute points to the previously created dense layer.

```python
dense_layer2 = pygad.nn.DenseLayer(num_neurons=5,
                                   previous_layer=dense_layer,
                                   activation_function="relu")
```

Because `dense_layer2` holds a reference to `dense_layer` in its `previous_layer` attribute, then the number of neurons in `dense_layer` can be accessed.

```python
dense_layer = dense_layer2.previous_layer
dense_layer_neurons = dense_layer.num_neurons

print("Number of dense neurons =", num_input_neurons)
```

After getting the reference to `dense_layer`, we can use it to access the number of input neurons.

```python
dense_layer = dense_layer2.previous_layer
input_layer = dense_layer.previous_layer
num_input_neurons = input_layer.num_neurons

print("Number of input neurons =", num_input_neurons)
```

Assuming that `dense_layer2` is the last dense layer, then it is regarded as the output layer.

#### `previous_layer` Attribute

The `previous_layer` attribute in the `pygad.nn.DenseLayer` class creates a one way linked list between all the layers in the network architecture as described by the next figure. 

The last (output) layer indexed N points to layer **N-1**, layer **N-1** points to the layer **N-2**, the layer **N-2** points to the layer **N-3**, and so on until reaching the end of the linked list which is layer 1 (input layer).

![Layers Linked List](https://user-images.githubusercontent.com/16560492/81918975-816af880-95d7-11ea-83e3-34d14c3316db.jpg)

The one way linked list allows returning all properties of all layers in the network architecture by just passing the last layer in the network. The linked list moves from the output layer towards the input layer. 

Using the `previous_layer` attribute of layer **N**, the layer **N-1** can be accessed. Using the `previous_layer` attribute of layer **N-1**, layer **N-2** can be accessed. The process continues until reaching a layer that does not have a `previous_layer` attribute (which is the input layer).

The properties of the layers include the weights (initial or trained), activation functions, and more. Here is how a `while` loop is used to iterate through all the layers. The `while` loop stops only when the current layer does not have a `previous_layer` attribute. This layer is the input layer.

```python
layer = dense_layer2

while "previous_layer" in layer.__init__.__code__.co_varnames:
    print("Number of neurons =", layer.num_neurons)

    # Go to the previous layer.
    layer = layer.previous_layer
```

## Functions to Manipulate Neural Networks

There are a number of functions existing in the `pygad.nn` module that helps to manipulate the neural network.

### `pygad.nn.layers_weights()`

Creates and returns a list holding the weights matrices of all layers in the neural network.

Accepts the following parameters:

- `last_layer`: A reference to the last (output) layer in the network architecture.
- `initial`: When `True` (default), the function returns the **initial** weights of the layers using the layers' `initial_weights` attribute. When `False`, it returns the **trained** weights of the layers using the layers' `trained_weights` attribute. The initial weights are only needed before network training starts. The trained weights are needed to predict the network outputs.

The function uses a `while` loop to iterate through the layers using their `previous_layer` attribute. For each layer, either the initial weights or the trained weights are returned based on whether the `initial` parameter is `True` or `False`.

### `pygad.nn.layers_weights_as_vector()`

Creates and returns a list holding the weights **vectors** of all layers in the neural network. The weights array of each layer is reshaped to get a vector.

This function is similar to the `layers_weights()` function except that it returns the weights of each layer as a vector, not as an array. 

Accepts the following parameters:

- `last_layer`: A reference to the last (output) layer in the network architecture.
- `initial`: When `True` (default), the function returns the **initial** weights of the layers using the layers' `initial_weights` attribute. When `False`, it returns the **trained** weights of the layers using the layers' `trained_weights` attribute. The initial weights are only needed before network training starts. The trained weights are needed to predict the network outputs.

The function uses a `while` loop to iterate through the layers using their `previous_layer` attribute. For each layer, either the initial weights or the trained weights are returned based on whether the `initial` parameter is `True` or `False`.

### `pygad.nn.layers_weights_as_matrix()`

Converts the network weights from vectors to matrices.

Compared to the `layers_weights_as_vectors()` function that only accepts a reference to the last layer and returns the network weights as vectors, this function accepts a reference to the last layer in addition to a list holding the weights as vectors. Such vectors are converted into matrices.

Accepts the following parameters:

- `last_layer`: A reference to the last (output) layer in the network architecture.
- `vector_weights`: The network weights as vectors where the weights of each layer form a single vector.

The function uses a `while` loop to iterate through the layers using their `previous_layer` attribute. For each layer, the shape of its weights array is returned. This shape is used to reshape the weights vector of the layer into a matrix. 

### `pygad.nn.layers_activations()`

Creates and returns a list holding the names of the activation functions of all layers in the neural network.

Accepts the following parameter:

- `last_layer`: A reference to the last (output) layer in the network architecture.

The function uses a `while` loop to iterate through the layers using their `previous_layer` attribute. For each layer, the name of the activation function used is returned using the layer's `activation_function` attribute. 

### `pygad.nn.sigmoid()`

Applies the sigmoid function and returns its result.

Accepts the following parameters:

* `sop`: The input to which the sigmoid function is applied.

### `pygad.nn.relu()`

Applies the rectified linear unit (ReLU) function and returns its result.

Accepts the following parameters:

* `sop`: The input to which the relu function is applied.

### `pygad.nn.softmax()`

Applies the softmax function and returns its result.

Accepts the following parameters:

* `sop`: The input to which the softmax function is applied.

### `pygad.nn.train()`

Trains the neural network.

Accepts the following parameters:

- `num_epochs`: Number of epochs.
- `last_layer`: Reference to the last (output) layer in the network architecture.
- `data_inputs`: Data features.
- `data_outputs`: Data outputs.
- `problem_type`: The type of the problem which can be either `"classification"` or `"regression"`. Added in PyGAD 2.7.0 and higher.
- `learning_rate`: Learning rate.

For each epoch, all the data samples are fed to the network to return their predictions. After each epoch, the weights are updated using only the learning rate. No learning algorithm is used because the purpose of this project is to only build the forward pass of training a neural network.

### `pygad.nn.update_weights()`

Calculates and returns the updated weights. Even though no training algorithm is used in this project, the weights are updated using the learning rate. It is not the best way to update the weights, but making small changes is better than keeping them as they are.

Accepts the following parameters:

- `weights`: The current weights of the network.
- `network_error`: The network error.
- `learning_rate`: The learning rate.

### `pygad.nn.update_layers_trained_weights()`

After the network weights are trained, this function updates the `trained_weights` attribute of each layer by the weights calculated after passing all the epochs (such weights are passed in the `final_weights` parameter)

By just passing a reference to the last layer in the network (i.e. output layer) in addition to the final weights, this function updates the `trained_weights` attribute of all layers.

Accepts the following parameters:

- `last_layer`: A reference to the last (output) layer in the network architecture.
- `final_weights`: An array of weights of all layers in the network after passing through all the epochs.

The function uses a `while` loop to iterate through the layers using their `previous_layer` attribute. For each layer, its `trained_weights` attribute is assigned the weights of the layer from the `final_weights` parameter. 

### `pygad.nn.predict()`

Uses the trained weights for predicting the samples' outputs. It returns a list of the predicted outputs for all samples.

Accepts the following parameters:

* `last_layer`: A reference to the last (output) layer in the network architecture.
* `data_inputs`: Data features.
* `problem_type`: The type of the problem which can be either `"classification"` or `"regression"`. Added in PyGAD 2.7.0 and higher.

All the data samples are fed to the network to return their predictions. 

## Helper Functions

There are functions in the `pygad.nn` module that do not directly manipulate the neural networks.

### `pygad.nn.to_vector()`

Converts a NumPy array (of any dimensionality) passed to its `array` parameter into a 1D vector and returns the vector.

Accepts the following parameters:

* `array`: The NumPy array to be converted into a 1D vector.

### `pygad.nn.to_array()`

Converts a vector passed to its `vector` parameter into a NumPy array and returns the array.

Accepts the following parameters:

- `vector`: The 1D vector to be converted into an array.
- `shape`: The target shape of the array.

## Supported Activation Functions

The supported activation functions are:

1. Sigmoid: Implemented using the `pygad.nn.sigmoid()` function.
2. Rectified Linear Unit (ReLU): Implemented using the `pygad.nn.relu()` function.
3. Softmax: Implemented using the `pygad.nn.softmax()` function.

## Steps to Build a Neural Network

This section discusses how to use the `pygad.nn` module to build a neural network. The steps are summarized as follows:

- Reading the Data
- Building the Network Architecture
- Training the Network
- Making Predictions
- Calculating Some Statistics

### Reading the Data

Before building the network architecture, the first thing to do is to prepare the data that will be used for training the network. 

In this example, 4 classes of the **Fruits360** dataset are used for preparing the training data. The 4 classes are:

1. [**Apple Braeburn**](https://github.com/ahmedfgad/NumPyANN/tree/master/apple): This class's data is available at https://github.com/ahmedfgad/NumPyANN/tree/master/apple
2. [**Lemon Meyer**](https://github.com/ahmedfgad/NumPyANN/tree/master/lemon): This class's data is available at https://github.com/ahmedfgad/NumPyANN/tree/master/lemon
3. [**Mango**](https://github.com/ahmedfgad/NumPyANN/tree/master/mango): This class's data is available at https://github.com/ahmedfgad/NumPyANN/tree/master/mango
4. [**Raspberry**](https://github.com/ahmedfgad/NumPyANN/tree/master/raspberry): This class's data is available at https://github.com/ahmedfgad/NumPyANN/tree/master/raspberry

The features from such 4 classes are extracted according to the next code. This code reads the raw images of the 4 classes of the dataset, prepares the features and the outputs as NumPy arrays, and saves the arrays in 2 files.

This code extracts a feature vector from each image representing the color histogram of the HSV space's hue channel. 

```python
import numpy
import skimage.io, skimage.color, skimage.feature
import os

fruits = ["apple", "raspberry", "mango", "lemon"]
# Number of samples in the datset used = 492+490+490+490=1,962
# 360 is the length of the feature vector.
dataset_features = numpy.zeros(shape=(1962, 360))
outputs = numpy.zeros(shape=(1962))

idx = 0
class_label = 0
for fruit_dir in fruits:
    curr_dir = os.path.join(os.path.sep, fruit_dir)
    all_imgs = os.listdir(os.getcwd()+curr_dir)
    for img_file in all_imgs:
        if img_file.endswith(".jpg"): # Ensures reading only JPG files.
            fruit_data = skimage.io.imread(fname=os.path.sep.join([os.getcwd(), curr_dir, img_file]), as_gray=False)
            fruit_data_hsv = skimage.color.rgb2hsv(rgb=fruit_data)
            hist = numpy.histogram(a=fruit_data_hsv[:, :, 0], bins=360)
            dataset_features[idx, :] = hist[0]
            outputs[idx] = class_label
            idx = idx + 1
    class_label = class_label + 1

# Saving the extracted features and the outputs as NumPy files.
numpy.save("dataset_features.npy", dataset_features)
numpy.save("outputs.npy", outputs)
```

To save your time, the training data is already prepared and 2 files created by the next code are available for download at these links:

1. [dataset_features.npy](https://github.com/ahmedfgad/NumPyANN/blob/master/dataset_features.npy): The features https://github.com/ahmedfgad/NumPyANN/blob/master/dataset_features.npy
2. [outputs.npy](https://github.com/ahmedfgad/NumPyANN/blob/master/outputs.npy): The class labels https://github.com/ahmedfgad/NumPyANN/blob/master/outputs.npy

The [outputs.npy](https://github.com/ahmedfgad/NumPyANN/blob/master/outputs.npy) file gives the following labels for the 4 classes:

1. [**Apple Braeburn**](https://github.com/ahmedfgad/NumPyANN/tree/master/apple): Class label is **0**
2. [**Lemon Meyer**](https://github.com/ahmedfgad/NumPyANN/tree/master/lemon): Class label is **1**
3. [**Mango**](https://github.com/ahmedfgad/NumPyANN/tree/master/mango): Class label is **2**
4. [**Raspberry**](https://github.com/ahmedfgad/NumPyANN/tree/master/raspberry): Class label is **3**

The project has 4 folders holding the images for the 4 classes.

After the 2 files are created, then just read them to return the NumPy arrays according to the next 2 lines:

```python
data_inputs = numpy.load("dataset_features.npy")
data_outputs = numpy.load("outputs.npy")
```

After the data is prepared, the next step is to create the network architecture.

### Building the Network Architecture

The input layer is created by instantiating the `pygad.nn.InputLayer` class according to the next code. A network can only have a single input layer.

```python
import pygad.nn
num_inputs = data_inputs.shape[1]

input_layer = pygad.nn.InputLayer(num_inputs)
```

After the input layer is created, the next step is to create a number of dense layers according to the next code. Normally, the last dense layer is regarded as the output layer. Note that the output layer has a number of neurons equal to the number of classes in the dataset which is 4.

```python
hidden_layer = pygad.nn.DenseLayer(num_neurons=HL2_neurons, previous_layer=input_layer, activation_function="relu")
output_layer = pygad.nn.DenseLayer(num_neurons=4, previous_layer=hidden_layer2, activation_function="softmax")
```

After both the data and the network architecture are prepared, the next step is to train the network.

### Training the Network

Here is an example of using the `pygad.nn.train()` function.

```python
pygad.nn.train(num_epochs=10,
               last_layer=output_layer,
               data_inputs=data_inputs,
               data_outputs=data_outputs,
               learning_rate=0.01)
```

After training the network, the next step is to make predictions.

### Making Predictions

The `pygad.nn.predict()` function uses the trained network for making predictions. Here is an example.

```python
predictions = pygad.nn.predict(last_layer=output_layer, data_inputs=data_inputs)
```

It is not expected to have high accuracy in the predictions because no training algorithm is used. 

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

It is very important to note that it is not expected that the classification accuracy is high because no training algorithm is used. Please check the documentation of the `pygad.gann` module for training the network using the genetic algorithm.

## Examples

This section gives the complete code of some examples that build neural networks using `pygad.nn`. Each subsection builds a different network.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} XOR Classification
:link: nn_xor
:link-type: doc
:::

:::{grid-item-card} Image Classification
:link: nn_image_classification
:link-type: doc
:::

:::{grid-item-card} Regression Example 1
:link: nn_regression_1
:link-type: doc
:::

:::{grid-item-card} Regression Example 2 - Fish Weight Prediction
:link: nn_regression_2
:link-type: doc
:::

::::

:::{toctree}
:hidden:

nn_xor
nn_image_classification
nn_regression_1
nn_regression_2
:::
