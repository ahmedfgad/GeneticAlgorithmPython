.. _pygadnn-module:

``pygad.nn`` Module
===================

This section of the PyGAD's library documentation discusses the
**pygad.nn** module.

Using the **pygad.nn** module, artificial neural networks are created.
The purpose of this module is to only implement the **forward pass** of
a neural network without using a training algorithm. The **pygad.nn**
module builds the network layers, implements the activations functions,
trains the network, makes predictions, and more.

Later, the **pygad.gann** module is used to train the **pygad.nn**
network using the genetic algorithm built in the **pygad** module.

Starting from `PyGAD
2.7.1 <https://pygad.readthedocs.io/en/latest/Footer.html#pygad-2-7-1>`__,
the **pygad.nn** module supports both classification and regression
problems. For more information, check the ``problem_type`` parameter in
the ``pygad.nn.train()`` and ``pygad.nn.predict()`` functions.

Supported Layers
================

Each layer supported by the **pygad.nn** module has a corresponding
class. The layers and their classes are:

1. **Input**: Implemented using the ``pygad.nn.InputLayer`` class.

2. **Dense** (Fully Connected): Implemented using the
   ``pygad.nn.DenseLayer`` class.

In the future, more layers will be added. The next subsections discuss
such layers.

.. _pygadnninputlayer-class:

``pygad.nn.InputLayer`` Class
-----------------------------

The ``pygad.nn.InputLayer`` class creates the input layer for the neural
network. For each network, there is only a single input layer. The
network architecture must start with an input layer.

This class has no methods or class attributes. All it has is a
constructor that accepts a parameter named ``num_neurons`` representing
the number of neurons in the input layer.

An instance attribute named ``num_neurons`` is created within the
constructor to keep such a number. Here is an example of building an
input layer with 20 neurons.

.. code:: python

   input_layer = pygad.nn.InputLayer(num_neurons=20)

Here is how the single attribute ``num_neurons`` within the instance of
the ``pygad.nn.InputLayer`` class can be accessed.

.. code:: python

   num_input_neurons = input_layer.num_neurons

   print("Number of input neurons =", num_input_neurons)

This is everything about the input layer.

.. _pygadnndenselayer-class:

``pygad.nn.DenseLayer`` Class
-----------------------------

Using the ``pygad.nn.DenseLayer`` class, dense (fully-connected) layers
can be created. To create a dense layer, just create a new instance of
the class. The constructor accepts the following parameters:

-  ``num_neurons``: Number of neurons in the dense layer.

-  ``previous_layer``: A reference to the previous layer. Using the
   ``previous_layer`` attribute, a linked list is created that connects
   all network layers.

-  ``activation_function``: A string representing the activation
   function to be used in this layer. Defaults to ``"sigmoid"``.
   Currently, the supported values for the activation functions are
   ``"sigmoid"``, ``"relu"``, ``"softmax"`` (supported in PyGAD 2.3.0
   and higher), and ``"None"`` (supported in PyGAD 2.7.0 and higher).
   When a layer has its activation function set to ``"None"``, then it
   means no activation function is applied. For a **regression
   problem**, set the activation function of the output (last) layer to
   ``"None"``. If all outputs in the regression problem are nonnegative,
   then it is possible to use the ReLU function in the output layer.

Within the constructor, the accepted parameters are used as instance
attributes. Besides the parameters, some new instance attributes are
created which are:

-  ``initial_weights``: The initial weights for the dense layer.

-  ``trained_weights``: The trained weights of the dense layer. This
   attribute is initialized by the value in the ``initial_weights``
   attribute.

Here is an example for creating a dense layer with 12 neurons. Note that
the ``previous_layer`` parameter is assigned to the input layer
``input_layer``.

.. code:: python

   dense_layer = pygad.nn.DenseLayer(num_neurons=12,
                                     previous_layer=input_layer,
                                     activation_function="relu")

Here is how to access some attributes in the dense layer:

.. code:: python

   num_dense_neurons = dense_layer.num_neurons
   dense_initail_weights = dense_layer.initial_weights

   print("Number of dense layer attributes =", num_dense_neurons)
   print("Initial weights of the dense layer :", dense_initail_weights)

Because ``dense_layer`` holds a reference to the input layer, then the
number of input neurons can be accessed.

.. code:: python

   input_layer = dense_layer.previous_layer
   num_input_neurons = input_layer.num_neurons

   print("Number of input neurons =", num_input_neurons)

Here is another dense layer. This dense layer's ``previous_layer``
attribute points to the previously created dense layer.

.. code:: python

   dense_layer2 = pygad.nn.DenseLayer(num_neurons=5,
                                      previous_layer=dense_layer,
                                      activation_function="relu")

Because ``dense_layer2`` holds a reference to ``dense_layer`` in its
``previous_layer`` attribute, then the number of neurons in
``dense_layer`` can be accessed.

.. code:: python

   dense_layer = dense_layer2.previous_layer
   dense_layer_neurons = dense_layer.num_neurons

   print("Number of dense neurons =", num_input_neurons)

After getting the reference to ``dense_layer``, we can use it to access
the number of input neurons.

.. code:: python

   dense_layer = dense_layer2.previous_layer
   input_layer = dense_layer.previous_layer
   num_input_neurons = input_layer.num_neurons

   print("Number of input neurons =", num_input_neurons)

Assuming that ``dense_layer2`` is the last dense layer, then it is
regarded as the output layer.

.. _previouslayer-attribute:

``previous_layer`` Attribute
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``previous_layer`` attribute in the ``pygad.nn.DenseLayer`` class
creates a one way linked list between all the layers in the network
architecture as described by the next figure.

The last (output) layer indexed N points to layer **N-1**, layer **N-1**
points to the layer **N-2**, the layer **N-2** points to the layer
**N-3**, and so on until reaching the end of the linked list which is
layer 1 (input layer).

.. figure:: https://user-images.githubusercontent.com/16560492/81918975-816af880-95d7-11ea-83e3-34d14c3316db.jpg
   :alt: 

The one way linked list allows returning all properties of all layers in
the network architecture by just passing the last layer in the network.
The linked list moves from the output layer towards the input layer.

Using the ``previous_layer`` attribute of layer **N**, the layer **N-1**
can be accessed. Using the ``previous_layer`` attribute of layer
**N-1**, layer **N-2** can be accessed. The process continues until
reaching a layer that does not have a ``previous_layer`` attribute
(which is the input layer).

The properties of the layers include the weights (initial or trained),
activation functions, and more. Here is how a ``while`` loop is used to
iterate through all the layers. The ``while`` loop stops only when the
current layer does not have a ``previous_layer`` attribute. This layer
is the input layer.

.. code:: python

   layer = dense_layer2

   while "previous_layer" in layer.__init__.__code__.co_varnames:
       print("Number of neurons =", layer.num_neurons)

       # Go to the previous layer.
       layer = layer.previous_layer

Functions to Manipulate Neural Networks
=======================================

There are a number of functions existing in the ``pygad.nn`` module that
helps to manipulate the neural network.

.. _pygadnnlayersweights:

``pygad.nn.layers_weights()``
-----------------------------

Creates and returns a list holding the weights matrices of all layers in
the neural network.

Accepts the following parameters:

-  ``last_layer``: A reference to the last (output) layer in the network
   architecture.

-  ``initial``: When ``True`` (default), the function returns the
   **initial** weights of the layers using the layers'
   ``initial_weights`` attribute. When ``False``, it returns the
   **trained** weights of the layers using the layers'
   ``trained_weights`` attribute. The initial weights are only needed
   before network training starts. The trained weights are needed to
   predict the network outputs.

The function uses a ``while`` loop to iterate through the layers using
their ``previous_layer`` attribute. For each layer, either the initial
weights or the trained weights are returned based on where the
``initial`` parameter is ``True`` or ``False``.

.. _pygadnnlayersweightsasvector:

``pygad.nn.layers_weights_as_vector()``
---------------------------------------

Creates and returns a list holding the weights **vectors** of all layers
in the neural network. The weights array of each layer is reshaped to
get a vector.

This function is similar to the ``layers_weights()`` function except
that it returns the weights of each layer as a vector, not as an array.

Accepts the following parameters:

-  ``last_layer``: A reference to the last (output) layer in the network
   architecture.

-  ``initial``: When ``True`` (default), the function returns the
   **initial** weights of the layers using the layers'
   ``initial_weights`` attribute. When ``False``, it returns the
   **trained** weights of the layers using the layers'
   ``trained_weights`` attribute. The initial weights are only needed
   before network training starts. The trained weights are needed to
   predict the network outputs.

The function uses a ``while`` loop to iterate through the layers using
their ``previous_layer`` attribute. For each layer, either the initial
weights or the trained weights are returned based on where the
``initial`` parameter is ``True`` or ``False``.

.. _pygadnnlayersweightsasmatrix:

``pygad.nn.layers_weights_as_matrix()``
---------------------------------------

Converts the network weights from vectors to matrices.

Compared to the ``layers_weights_as_vectors()`` function that only
accepts a reference to the last layer and returns the network weights as
vectors, this function accepts a reference to the last layer in addition
to a list holding the weights as vectors. Such vectors are converted
into matrices.

Accepts the following parameters:

-  ``last_layer``: A reference to the last (output) layer in the network
   architecture.

-  ``vector_weights``: The network weights as vectors where the weights
   of each layer form a single vector.

The function uses a ``while`` loop to iterate through the layers using
their ``previous_layer`` attribute. For each layer, the shape of its
weights array is returned. This shape is used to reshape the weights
vector of the layer into a matrix.

.. _pygadnnlayersactivations:

``pygad.nn.layers_activations()``
---------------------------------

Creates and returns a list holding the names of the activation functions
of all layers in the neural network.

Accepts the following parameter:

-  ``last_layer``: A reference to the last (output) layer in the network
   architecture.

The function uses a ``while`` loop to iterate through the layers using
their ``previous_layer`` attribute. For each layer, the name of the
activation function used is returned using the layer's
``activation_function`` attribute.

.. _pygadnnsigmoid:

``pygad.nn.sigmoid()``
----------------------

Applies the sigmoid function and returns its result.

Accepts the following parameters:

-  ``sop``: The input to which the sigmoid function is applied.

.. _pygadnnrelu:

``pygad.nn.relu()``
-------------------

Applies the rectified linear unit (ReLU) function and returns its
result.

Accepts the following parameters:

-  ``sop``: The input to which the relu function is applied.

.. _pygadnnsoftmax:

``pygad.nn.softmax()``
----------------------

Applies the softmax function and returns its result.

Accepts the following parameters:

-  ``sop``: The input to which the softmax function is applied.

.. _pygadnntrain:

``pygad.nn.train()``
--------------------

Trains the neural network.

Accepts the following parameters:

-  ``num_epochs``: Number of epochs.

-  ``last_layer``: Reference to the last (output) layer in the network
   architecture.

-  ``data_inputs``: Data features.

-  ``data_outputs``: Data outputs.

-  ``problem_type``: The type of the problem which can be either
   ``"classification"`` or ``"regression"``. Added in PyGAD 2.7.0 and
   higher.

-  ``learning_rate``: Learning rate.

For each epoch, all the data samples are fed to the network to return
their predictions. After each epoch, the weights are updated using only
the learning rate. No learning algorithm is used because the purpose of
this project is to only build the forward pass of training a neural
network.

.. _pygadnnupdateweights:

``pygad.nn.update_weights()``
-----------------------------

Calculates and returns the updated weights. Even no training algorithm
is used in this project, the weights are updated using the learning
rate. It is not the best way to update the weights but it is better than
keeping it as it is by making some small changes to the weights.

Accepts the following parameters:

-  ``weights``: The current weights of the network.

-  ``network_error``: The network error.

-  ``learning_rate``: The learning rate.

.. _pygadnnupdatelayerstrainedweights:

``pygad.nn.update_layers_trained_weights()``
--------------------------------------------

After the network weights are trained, this function updates the
``trained_weights`` attribute of each layer by the weights calculated
after passing all the epochs (such weights are passed in the
``final_weights`` parameter)

By just passing a reference to the last layer in the network (i.e.
output layer) in addition to the final weights, this function updates
the ``trained_weights`` attribute of all layers.

Accepts the following parameters:

-  ``last_layer``: A reference to the last (output) layer in the network
   architecture.

-  ``final_weights``: An array of weights of all layers in the network
   after passing through all the epochs.

The function uses a ``while`` loop to iterate through the layers using
their ``previous_layer`` attribute. For each layer, its
``trained_weights`` attribute is assigned the weights of the layer from
the ``final_weights`` parameter.

.. _pygadnnpredict:

``pygad.nn.predict()``
----------------------

Uses the trained weights for predicting the samples' outputs. It returns
a list of the predicted outputs for all samples.

Accepts the following parameters:

-  ``last_layer``: A reference to the last (output) layer in the network
   architecture.

-  ``data_inputs``: Data features.

-  ``problem_type``: The type of the problem which can be either
   ``"classification"`` or ``"regression"``. Added in PyGAD 2.7.0 and
   higher.

All the data samples are fed to the network to return their predictions.

Helper Functions
================

There are functions in the ``pygad.nn`` module that does not directly
manipulate the neural networks.

.. _pygadnntovector:

``pygad.nn.to_vector()``
------------------------

Converts a passed NumPy array (of any dimensionality) to its ``array``
parameter into a 1D vector and returns the vector.

Accepts the following parameters:

-  ``array``: The NumPy array to be converted into a 1D vector.

.. _pygadnntoarray:

``pygad.nn.to_array()``
-----------------------

Converts a passed vector to its ``vector`` parameter into a NumPy array
and returns the array.

Accepts the following parameters:

-  ``vector``: The 1D vector to be converted into an array.

-  ``shape``: The target shape of the array.

Supported Activation Functions
==============================

The supported activation functions are:

1. Sigmoid: Implemented using the ``pygad.nn.sigmoid()`` function.

2. Rectified Linear Unit (ReLU): Implemented using the
   ``pygad.nn.relu()`` function.

3. Softmax: Implemented using the ``pygad.nn.softmax()`` function.

Steps to Build a Neural Network
===============================

This section discusses how to use the ``pygad.nn`` module for building a
neural network. The summary of the steps are as follows:

-  Reading the Data

-  Building the Network Architecture

-  Training the Network

-  Making Predictions

-  Calculating Some Statistics

Reading the Data
----------------

Before building the network architecture, the first thing to do is to
prepare the data that will be used for training the network.

In this example, 4 classes of the **Fruits360** dataset are used for
preparing the training data. The 4 classes are:

1. `Apple
   Braeburn <https://github.com/ahmedfgad/NumPyANN/tree/master/apple>`__:
   This class's data is available at
   https://github.com/ahmedfgad/NumPyANN/tree/master/apple

2. `Lemon
   Meyer <https://github.com/ahmedfgad/NumPyANN/tree/master/lemon>`__:
   This class's data is available at
   https://github.com/ahmedfgad/NumPyANN/tree/master/lemon

3. `Mango <https://github.com/ahmedfgad/NumPyANN/tree/master/mango>`__:
   This class's data is available at
   https://github.com/ahmedfgad/NumPyANN/tree/master/mango

4. `Raspberry <https://github.com/ahmedfgad/NumPyANN/tree/master/raspberry>`__:
   This class's data is available at
   https://github.com/ahmedfgad/NumPyANN/tree/master/raspberry

The features from such 4 classes are extracted according to the next
code. This code reads the raw images of the 4 classes of the dataset,
prepares the features and the outputs as NumPy arrays, and saves the
arrays in 2 files.

This code extracts a feature vector from each image representing the
color histogram of the HSV space's hue channel.

.. code:: python

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

To save your time, the training data is already prepared and 2 files
created by the next code are available for download at these links:

1. `dataset_features.npy <https://github.com/ahmedfgad/NumPyANN/blob/master/dataset_features.npy>`__:
   The features
   https://github.com/ahmedfgad/NumPyANN/blob/master/dataset_features.npy

2. `outputs.npy <https://github.com/ahmedfgad/NumPyANN/blob/master/outputs.npy>`__:
   The class labels
   https://github.com/ahmedfgad/NumPyANN/blob/master/outputs.npy

The
`outputs.npy <https://github.com/ahmedfgad/NumPyANN/blob/master/outputs.npy>`__
file gives the following labels for the 4 classes:

1. `Apple
   Braeburn <https://github.com/ahmedfgad/NumPyANN/tree/master/apple>`__:
   Class label is **0**

2. `Lemon
   Meyer <https://github.com/ahmedfgad/NumPyANN/tree/master/lemon>`__:
   Class label is **1**

3. `Mango <https://github.com/ahmedfgad/NumPyANN/tree/master/mango>`__:
   Class label is **2**

4. `Raspberry <https://github.com/ahmedfgad/NumPyANN/tree/master/raspberry>`__:
   Class label is **3**

The project has 4 folders holding the images for the 4 classes.

After the 2 files are created, then just read them to return the NumPy
arrays according to the next 2 lines:

.. code:: python

   data_inputs = numpy.load("dataset_features.npy")
   data_outputs = numpy.load("outputs.npy")

After the data is prepared, next is to create the network architecture.

Building the Network Architecture
---------------------------------

The input layer is created by instantiating the ``pygad.nn.InputLayer``
class according to the next code. A network can only have a single input
layer.

.. code:: python

   import pygad.nn
   num_inputs = data_inputs.shape[1]

   input_layer = pygad.nn.InputLayer(num_inputs)

After the input layer is created, next is to create a number of dense
layers according to the next code. Normally, the last dense layer is
regarded as the output layer. Note that the output layer has a number of
neurons equal to the number of classes in the dataset which is 4.

.. code:: python

   hidden_layer = pygad.nn.DenseLayer(num_neurons=HL2_neurons, previous_layer=input_layer, activation_function="relu")
   output_layer = pygad.nn.DenseLayer(num_neurons=4, previous_layer=hidden_layer2, activation_function="softmax")

After both the data and the network architecture are prepared, the next
step is to train the network.

Training the Network
--------------------

Here is an example of using the ``pygad.nn.train()`` function.

.. code:: python

   pygad.nn.train(num_epochs=10,
                  last_layer=output_layer,
                  data_inputs=data_inputs,
                  data_outputs=data_outputs,
                  learning_rate=0.01)

After training the network, the next step is to make predictions.

Making Predictions
------------------

The ``pygad.nn.predict()`` function uses the trained network for making
predictions. Here is an example.

.. code:: python

   predictions = pygad.nn.predict(last_layer=output_layer, data_inputs=data_inputs)

It is not expected to have high accuracy in the predictions because no
training algorithm is used.

Calculating Some Statistics
---------------------------

Based on the predictions the network made, some statistics can be
calculated such as the number of correct and wrong predictions in
addition to the classification accuracy.

.. code:: python

   num_wrong = numpy.where(predictions != data_outputs)[0]
   num_correct = data_outputs.size - num_wrong.size
   accuracy = 100 * (num_correct/data_outputs.size)
   print("Number of correct classifications : {num_correct}.".format(num_correct=num_correct))
   print("Number of wrong classifications : {num_wrong}.".format(num_wrong=num_wrong.size))
   print("Classification accuracy : {accuracy}.".format(accuracy=accuracy))

It is very important to note that it is not expected that the
classification accuracy is high because no training algorithm is used.
Please check the documentation of the ``pygad.gann`` module for training
the network using the genetic algorithm.

Examples
========

This section gives the complete code of some examples that build neural
networks using ``pygad.nn``. Each subsection builds a different network.

XOR Classification
------------------

This is an example of building a network with 1 hidden layer with 2
neurons for building a network that simulates the XOR logic gate.
Because the XOR problem has 2 classes (0 and 1), then the output layer
has 2 neurons, one for each class.

.. code:: python

   import numpy
   import pygad.nn

   # Preparing the NumPy array of the inputs.
   data_inputs = numpy.array([[1, 1],
                              [1, 0],
                              [0, 1],
                              [0, 0]])

   # Preparing the NumPy array of the outputs.
   data_outputs = numpy.array([0, 
                               1, 
                               1, 
                               0])

   # The number of inputs (i.e. feature vector length) per sample
   num_inputs = data_inputs.shape[1]
   # Number of outputs per sample
   num_outputs = 2

   HL1_neurons = 2

   # Building the network architecture.
   input_layer = pygad.nn.InputLayer(num_inputs)
   hidden_layer1 = pygad.nn.DenseLayer(num_neurons=HL1_neurons, previous_layer=input_layer, activation_function="relu")
   output_layer = pygad.nn.DenseLayer(num_neurons=num_outputs, previous_layer=hidden_layer1, activation_function="softmax")

   # Training the network.
   pygad.nn.train(num_epochs=10,
                  last_layer=output_layer,
                  data_inputs=data_inputs,
                  data_outputs=data_outputs,
                  learning_rate=0.01)

   # Using the trained network for predictions.
   predictions = pygad.nn.predict(last_layer=output_layer, data_inputs=data_inputs)

   # Calculating some statistics
   num_wrong = numpy.where(predictions != data_outputs)[0]
   num_correct = data_outputs.size - num_wrong.size
   accuracy = 100 * (num_correct/data_outputs.size)
   print("Number of correct classifications : {num_correct}.".format(num_correct=num_correct))
   print("Number of wrong classifications : {num_wrong}.".format(num_wrong=num_wrong.size))
   print("Classification accuracy : {accuracy}.".format(accuracy=accuracy))

Image Classification
--------------------

This example is discussed in the **Steps to Build a Neural Network**
section and its complete code is listed below.

Remember to either download or create the
`dataset_features.npy <https://github.com/ahmedfgad/NumPyANN/blob/master/dataset_features.npy>`__
and
`outputs.npy <https://github.com/ahmedfgad/NumPyANN/blob/master/outputs.npy>`__
files before running this code.

.. code:: python

   import numpy
   import pygad.nn

   # Reading the data features. Check the 'extract_features.py' script for extracting the features & preparing the outputs of the dataset.
   data_inputs = numpy.load("dataset_features.npy") # Download from https://github.com/ahmedfgad/NumPyANN/blob/master/dataset_features.npy

   # Optional step for filtering the features using the standard deviation.
   features_STDs = numpy.std(a=data_inputs, axis=0)
   data_inputs = data_inputs[:, features_STDs > 50]

   # Reading the data outputs. Check the 'extract_features.py' script for extracting the features & preparing the outputs of the dataset.
   data_outputs = numpy.load("outputs.npy") # Download from https://github.com/ahmedfgad/NumPyANN/blob/master/outputs.npy

   # The number of inputs (i.e. feature vector length) per sample
   num_inputs = data_inputs.shape[1]
   # Number of outputs per sample
   num_outputs = 4

   HL1_neurons = 150
   HL2_neurons = 60

   # Building the network architecture.
   input_layer = pygad.nn.InputLayer(num_inputs)
   hidden_layer1 = pygad.nn.DenseLayer(num_neurons=HL1_neurons, previous_layer=input_layer, activation_function="relu")
   hidden_layer2 = pygad.nn.DenseLayer(num_neurons=HL2_neurons, previous_layer=hidden_layer1, activation_function="relu")
   output_layer = pygad.nn.DenseLayer(num_neurons=num_outputs, previous_layer=hidden_layer2, activation_function="softmax")

   # Training the network.
   pygad.nn.train(num_epochs=10,
                  last_layer=output_layer,
                  data_inputs=data_inputs,
                  data_outputs=data_outputs,
                  learning_rate=0.01)

   # Using the trained network for predictions.
   predictions = pygad.nn.predict(last_layer=output_layer, data_inputs=data_inputs)

   # Calculating some statistics
   num_wrong = numpy.where(predictions != data_outputs)[0]
   num_correct = data_outputs.size - num_wrong.size
   accuracy = 100 * (num_correct/data_outputs.size)
   print("Number of correct classifications : {num_correct}.".format(num_correct=num_correct))
   print("Number of wrong classifications : {num_wrong}.".format(num_wrong=num_wrong.size))
   print("Classification accuracy : {accuracy}.".format(accuracy=accuracy))

Regression Example 1
--------------------

The next code listing builds a neural network for regression. Here is
what to do to make the code works for regression:

1. Set the ``problem_type`` parameter in the ``pygad.nn.train()`` and
   ``pygad.nn.predict()`` functions to the string ``"regression"``.

.. code:: python

   pygad.nn.train(...,
                  problem_type="regression")

   predictions = pygad.nn.predict(..., 
                                  problem_type="regression")

1. Set the activation function for the output layer to the string
   ``"None"``.

.. code:: python

   output_layer = pygad.nn.DenseLayer(num_neurons=num_outputs, previous_layer=hidden_layer1, activation_function="None")

1. Calculate the prediction error according to your preferred error
   function. Here is how the mean absolute error is calculated.

.. code:: python

   abs_error = numpy.mean(numpy.abs(predictions - data_outputs))
   print("Absolute error : {abs_error}.".format(abs_error=abs_error))

Here is the complete code. Yet, there is no algorithm used to train the
network and thus the network is expected to give bad results. Later, the
``pygad.gann`` module is used to train either a regression or
classification networks.

.. code:: python

   import numpy
   import pygad.nn

   # Preparing the NumPy array of the inputs.
   data_inputs = numpy.array([[2, 5, -3, 0.1],
                              [8, 15, 20, 13]])

   # Preparing the NumPy array of the outputs.
   data_outputs = numpy.array([0.1, 
                               1.5])

   # The number of inputs (i.e. feature vector length) per sample
   num_inputs = data_inputs.shape[1]
   # Number of outputs per sample
   num_outputs = 1

   HL1_neurons = 2

   # Building the network architecture.
   input_layer = pygad.nn.InputLayer(num_inputs)
   hidden_layer1 = pygad.nn.DenseLayer(num_neurons=HL1_neurons, previous_layer=input_layer, activation_function="relu")
   output_layer = pygad.nn.DenseLayer(num_neurons=num_outputs, previous_layer=hidden_layer1, activation_function="None")

   # Training the network.
   pygad.nn.train(num_epochs=100,
                  last_layer=output_layer,
                  data_inputs=data_inputs,
                  data_outputs=data_outputs,
                  learning_rate=0.01,
                  problem_type="regression")

   # Using the trained network for predictions.
   predictions = pygad.nn.predict(last_layer=output_layer, 
                            data_inputs=data_inputs, 
                            problem_type="regression")

   # Calculating some statistics
   abs_error = numpy.mean(numpy.abs(predictions - data_outputs))
   print("Absolute error : {abs_error}.".format(abs_error=abs_error))

Regression Example 2 - Fish Weight Prediction
---------------------------------------------

This example uses the Fish Market Dataset available at Kaggle
(https://www.kaggle.com/aungpyaeap/fish-market). Simply download the CSV
dataset from `this
link <https://www.kaggle.com/aungpyaeap/fish-market/download>`__
(https://www.kaggle.com/aungpyaeap/fish-market/download). The dataset is
also available at the `GitHub project of the ``pygad.nn``
module <https://github.com/ahmedfgad/NumPyANN>`__:
https://github.com/ahmedfgad/NumPyANN

Using the Pandas library, the dataset is read using the ``read_csv()``
function.

.. code:: python

   data = numpy.array(pandas.read_csv("Fish.csv"))

The last 5 columns in the dataset are used as inputs and the **Weight**
column is used as output.

.. code:: python

   # Preparing the NumPy array of the inputs.
   data_inputs = numpy.asarray(data[:, 2:], dtype=numpy.float32)

   # Preparing the NumPy array of the outputs.
   data_outputs = numpy.asarray(data[:, 1], dtype=numpy.float32) # Fish Weight

Note how the activation function at the last layer is set to ``"None"``.
Moreover, the ``problem_type`` parameter in the ``pygad.nn.train()`` and
``pygad.nn.predict()`` functions is set to ``"regression"``.

After the ``pygad.nn.train()`` function completes, the mean absolute
error is calculated.

.. code:: python

   abs_error = numpy.mean(numpy.abs(predictions - data_outputs))
   print("Absolute error : {abs_error}.".format(abs_error=abs_error))

Here is the complete code.

.. code:: python

   import numpy
   import pygad.nn
   import pandas

   data = numpy.array(pandas.read_csv("Fish.csv"))

   # Preparing the NumPy array of the inputs.
   data_inputs = numpy.asarray(data[:, 2:], dtype=numpy.float32)

   # Preparing the NumPy array of the outputs.
   data_outputs = numpy.asarray(data[:, 1], dtype=numpy.float32) # Fish Weight

   # The number of inputs (i.e. feature vector length) per sample
   num_inputs = data_inputs.shape[1]
   # Number of outputs per sample
   num_outputs = 1

   HL1_neurons = 2

   # Building the network architecture.
   input_layer = pygad.nn.InputLayer(num_inputs)
   hidden_layer1 = pygad.nn.DenseLayer(num_neurons=HL1_neurons, previous_layer=input_layer, activation_function="relu")
   output_layer = pygad.nn.DenseLayer(num_neurons=num_outputs, previous_layer=hidden_layer1, activation_function="None")

   # Training the network.
   pygad.nn.train(num_epochs=100,
                  last_layer=output_layer,
                  data_inputs=data_inputs,
                  data_outputs=data_outputs,
                  learning_rate=0.01,
                  problem_type="regression")

   # Using the trained network for predictions.
   predictions = pygad.nn.predict(last_layer=output_layer, 
                            data_inputs=data_inputs, 
                            problem_type="regression")

   # Calculating some statistics
   abs_error = numpy.mean(numpy.abs(predictions - data_outputs))
   print("Absolute error : {abs_error}.".format(abs_error=abs_error))
