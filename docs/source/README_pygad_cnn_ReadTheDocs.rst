.. _pygadcnn-module:

``pygad.cnn`` Module
====================

This section of the PyGAD's library documentation discusses the
**pygad.cnn** module.

Using the **pygad.cnn** module, convolutional neural networks (CNNs) are
created. The purpose of this module is to only implement the **forward
pass** of a convolutional neural network without using a training
algorithm. The **pygad.cnn** module builds the network layers,
implements the activations functions, trains the network, makes
predictions, and more.

Later, the **pygad.gacnn** module is used to train the **pygad.cnn**
network using the genetic algorithm built in the **pygad** module.

Supported Layers
================

Each layer supported by the **pygad.cnn** module has a corresponding
class. The layers and their classes are:

1. **Input**: Implemented using the ``pygad.cnn.Input2D`` class.

2. **Convolution**: Implemented using the ``pygad.cnn.Conv2D`` class.

3. **Max Pooling**: Implemented using the ``pygad.cnn.MaxPooling2D``
   class.

4. **Average Pooling**: Implemented using the
   ``pygad.cnn.AveragePooling2D`` class.

5. **Flatten**: Implemented using the ``pygad.cnn.Flatten`` class.

6. **ReLU**: Implemented using the ``pygad.cnn.ReLU`` class.

7. **Sigmoid**: Implemented using the ``pygad.cnn.Sigmoid`` class.

8. **Dense** (Fully Connected): Implemented using the
   ``pygad.cnn.Dense`` class.

In the future, more layers will be added.

Except for the input layer, all of listed layers has 4 instance
attributes that do the same function which are:

1. ``previous_layer``: A reference to the previous layer in the CNN
   architecture.

2. ``layer_input_size``: The size of the input to the layer.

3. ``layer_output_size``: The size of the output from the layer.

4. ``layer_output``: The latest output generated from the layer. It
   default to ``None``.

In addition to such attributes, the layers may have some additional
attributes. The next subsections discuss such layers.

.. _pygadcnninput2d-class:

``pygad.cnn.Input2D`` Class
---------------------------

The ``pygad.cnn.Input2D`` class creates the input layer for the
convolutional neural network. For each network, there is only a single
input layer. The network architecture must start with an input layer.

This class has no methods or class attributes. All it has is a
constructor that accepts a parameter named ``input_shape`` representing
the shape of the input.

The instances from the ``Input2D`` class has the following attributes:

1. ``input_shape``: The shape of the input to the pygad.cnn.

2. ``layer_output_size``

Here is an example of building an input layer with shape
``(50, 50, 3)``.

.. code:: python

   input_layer = pygad.cnn.Input2D(input_shape=(50, 50, 3))

Here is how to access the attributes within the instance of the
``pygad.cnn.Input2D`` class.

.. code:: python

   input_shape = input_layer.input_shape
   layer_output_size = input_layer.layer_output_size

   print("Input2D Input shape =", input_shape)
   print("Input2D Output shape =", layer_output_size)

This is everything about the input layer.

.. _pygadcnnconv2d-class:

``pygad.cnn.Conv2D`` Class
--------------------------

Using the ``pygad.cnn.Conv2D`` class, convolution (conv) layers can be
created. To create a convolution layer, just create a new instance of
the class. The constructor accepts the following parameters:

-  ``num_filters``: Number of filters.

-  ``kernel_size``: Filter kernel size.

-  ``previous_layer``: A reference to the previous layer. Using the
   ``previous_layer`` attribute, a linked list is created that connects
   all network layers. For more information about this attribute, please
   check the **previous_layer** attribute section of the ``pygad.nn``
   module documentation.

-  ``activation_function=None``: A string representing the activation
   function to be used in this layer. Defaults to ``None`` which means
   no activation function is applied while applying the convolution
   layer. An activation layer can be added separately in this case. The
   supported activation functions in the conv layer are ``relu`` and
   ``sigmoid``.

Within the constructor, the accepted parameters are used as instance
attributes. Besides the parameters, some new instance attributes are
created which are:

-  ``filter_bank_size``: Size of the filter bank in this layer.

-  ``initial_weights``: The initial weights for the conv layer.

-  ``trained_weights``: The trained weights of the conv layer. This
   attribute is initialized by the value in the ``initial_weights``
   attribute.

-  ``layer_input_size``

-  ``layer_output_size``

-  ``layer_output``

Here is an example for creating a conv layer with 2 filters and a kernel
size of 3. Note that the ``previous_layer`` parameter is assigned to the
input layer ``input_layer``.

.. code:: python

   conv_layer = pygad.cnn.Conv2D(num_filters=2,
                           kernel_size=3,
                           previous_layer=input_layer,
                           activation_function=None)

Here is how to access some attributes in the dense layer:

.. code:: python

   filter_bank_size = conv_layer.filter_bank_size
   conv_initail_weights = conv_layer.initial_weights

   print("Filter bank size attributes =", filter_bank_size)
   print("Initial weights of the conv layer :", conv_initail_weights)

Because ``conv_layer`` holds a reference to the input layer, then the
number of input neurons can be accessed.

.. code:: python

   input_layer = conv_layer.previous_layer
   input_shape = input_layer.num_neurons

   print("Input shape =", input_shape)

Here is another conv layer where its ``previous_layer`` attribute points
to the previously created conv layer and it uses the ``ReLU`` activation
function.

.. code:: python

   conv_layer2 = pygad.cnn.Conv2D(num_filters=2,
                            kernel_size=3,
                            previous_layer=conv_layer,
                            activation_function="relu")

Because ``conv_layer2`` holds a reference to ``conv_layer`` in its
``previous_layer`` attribute, then the attributes in ``conv_layer`` can
be accessed.

.. code:: python

   conv_layer = conv_layer2.previous_layer
   filter_bank_size = conv_layer.filter_bank_size

   print("Filter bank size attributes =", filter_bank_size)

After getting the reference to ``conv_layer``, we can use it to access
the number of input neurons.

.. code:: python

   conv_layer = conv_layer2.previous_layer
   input_layer = conv_layer.previous_layer
   input_shape = input_layer.num_neurons

   print("Input shape =", input_shape)

.. _pygadcnnmaxpooling2d-class:

``pygad.cnn.MaxPooling2D`` Class
--------------------------------

The ``pygad.cnn.MaxPooling2D`` class builds a max pooling layer for the
CNN architecture. The constructor of this class accepts the following
parameter:

-  ``pool_size``: Size of the window.

-  ``previous_layer``: A reference to the previous layer in the CNN
   architecture.

-  ``stride=2``: A stride that default to 2.

Within the constructor, the accepted parameters are used as instance
attributes. Besides the parameters, some new instance attributes are
created which are:

-  ``layer_input_size``

-  ``layer_output_size``

-  ``layer_output``

.. _pygadcnnaveragepooling2d-class:

``pygad.cnn.AveragePooling2D`` Class
------------------------------------

The ``pygad.cnn.AveragePooling2D`` class is similar to the
``pygad.cnn.MaxPooling2D`` class except that it applies the max pooling
operation rather than average pooling.

.. _pygadcnnflatten-class:

``pygad.cnn.Flatten`` Class
---------------------------

The ``pygad.cnn.Flatten`` class implements the flatten layer which
converts the output of the previous layer into a 1D vector. The
constructor accepts only the ``previous_layer`` parameter.

The following instance attributes exist:

-  ``previous_layer``

-  ``layer_input_size``

-  ``layer_output_size``

-  ``layer_output``

.. _pygadcnnrelu-class:

``pygad.cnn.ReLU`` Class
------------------------

The ``pygad.cnn.ReLU`` class implements the ReLU layer which applies the
ReLU activation function to the output of the previous layer.

The constructor accepts only the ``previous_layer`` parameter.

The following instance attributes exist:

-  ``previous_layer``

-  ``layer_input_size``

-  ``layer_output_size``

-  ``layer_output``

.. _pygadcnnsigmoid-class:

``pygad.cnn.Sigmoid`` Class
---------------------------

The ``pygad.cnn.Sigmoid`` class is similar to the ``pygad.cnn.ReLU``
class except that it applies the sigmoid function rather than the ReLU
function.

.. _pygadcnndense-class:

``pygad.cnn.Dense`` Class
-------------------------

The ``pygad.cnn.Dense`` class implement the dense layer. Its constructor
accepts the following parameters:

-  ``num_neurons``: Number of neurons in the dense layer.

-  ``previous_layer``: A reference to the previous layer.

-  ``activation_function``: A string representing the activation
   function to be used in this layer. Defaults to ``"sigmoid"``.
   Currently, the supported activation functions in the dense layer are
   ``"sigmoid"``, ``"relu"``, and ``softmax``.

Within the constructor, the accepted parameters are used as instance
attributes. Besides the parameters, some new instance attributes are
created which are:

-  ``initial_weights``: The initial weights for the dense layer.

-  ``trained_weights``: The trained weights of the dense layer. This
   attribute is initialized by the value in the ``initial_weights``
   attribute.

-  ``layer_input_size``

-  ``layer_output_size``

-  ``layer_output``

.. _pygadcnnmodel-class:

``pygad.cnn.Model`` Class
=========================

An instance of the ``pygad.cnn.Model`` class represents a CNN model. The
constructor of this class accepts the following parameters:

-  ``last_layer``: A reference to the last layer in the CNN architecture
   (i.e. dense layer).

-  ``epochs=10``: Number of epochs.

-  ``learning_rate=0.01``: Learning rate.

Within the constructor, the accepted parameters are used as instance
attributes. Besides the parameters, a new instance attribute named
``network_layers`` is created which holds a list with references to the
CNN layers. Such a list is returned using the ``get_layers()`` method in
the ``pygad.cnn.Model`` class.

There are a number of methods in the ``pygad.cnn.Model`` class which
serves in training, testing, and retrieving information about the model.
These methods are discussed in the next subsections.

.. _getlayers:

``get_layers()``
----------------

Creates a list of all layers in the CNN model. It accepts no parameters.

``train()``
-----------

Trains the CNN model.

Accepts the following parameters:

-  ``train_inputs``: Training data inputs.

-  ``train_outputs``: Training data outputs.

This method trains the CNN model according to the number of epochs
specified in the constructor of the ``pygad.cnn.Model`` class.

It is important to note that no learning algorithm is used for training
the pygad.cnn. Just the learning rate is used for making some changes
which is better than leaving the weights unchanged.

.. _feedsample:

``feed_sample()``
-----------------

Feeds a sample in the CNN layers and returns results of the last layer
in the pygad.cnn.

.. _updateweights:

``update_weights()``
--------------------

Updates the CNN weights using the learning rate. It is important to note
that no learning algorithm is used for training the pygad.cnn. Just the
learning rate is used for making some changes which is better than
leaving the weights unchanged.

``predict()``
-------------

Uses the trained CNN for making predictions.

Accepts the following parameter:

-  ``data_inputs``: The inputs to predict their label.

It returns a list holding the samples predictions.

``summary()``
-------------

Prints a summary of the CNN architecture.

Supported Activation Functions
==============================

The supported activation functions in the convolution layer are:

1. Sigmoid: Implemented using the ``pygad.cnn.sigmoid()`` function.

2. Rectified Linear Unit (ReLU): Implemented using the
   ``pygad.cnn.relu()`` function.

The dense layer supports these functions besides the ``softmax``
function implemented in the ``pygad.cnn.softmax()`` function.

Steps to Build a Neural Network
===============================

This section discusses how to use the ``pygad.cnn`` module for building
a neural network. The summary of the steps are as follows:

-  Reading the Data

-  Building the CNN Architecture

-  Building Model

-  Model Summary

-  Training the CNN

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

Just 20 samples from each of the 4 classes are saved into a NumPy array
available in the
`dataset_inputs.npy <https://github.com/ahmedfgad/NumPyCNN/blob/master/dataset_inputs.npy>`__
file:
https://github.com/ahmedfgad/NumPyCNN/blob/master/dataset_inputs.npy

The shape of this array is ``(80, 100, 100, 3)`` where the shape of the
single image is ``(100, 100, 3)``.

The
`dataset_outputs.npy <https://github.com/ahmedfgad/NumPyCNN/blob/master/dataset_outputs.npy>`__
file
(https://github.com/ahmedfgad/NumPyCNN/blob/master/dataset_outputs.npy)
has the class labels for the 4 classes:

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

Simply, download and reach the 2 files to return the NumPy arrays
according to the next 2 lines:

.. code:: python

   train_inputs = numpy.load("dataset_inputs.npy")
   train_outputs = numpy.load("dataset_outputs.npy")

After the data is prepared, next is to create the network architecture.

Building the Network Architecture
---------------------------------

The input layer is created by instantiating the ``pygad.cnn.Input2D``
class according to the next code. A network can only have a single input
layer.

.. code:: python

   import pygad.cnn
   sample_shape = train_inputs.shape[1:]

   input_layer = pygad.cnn.Input2D(input_shape=sample_shape)

After the input layer is created, next is to create a number of layers
layers according to the next code. Normally, the last dense layer is
regarded as the output layer. Note that the output layer has a number of
neurons equal to the number of classes in the dataset which is 4.

.. code:: python

   conv_layer1 = pygad.cnn.Conv2D(num_filters=2,
                                  kernel_size=3,
                                  previous_layer=input_layer,
                                  activation_function=None)
   relu_layer1 = pygad.cnn.Sigmoid(previous_layer=conv_layer1)
   average_pooling_layer = pygad.cnn.AveragePooling2D(pool_size=2, 
                                                      previous_layer=relu_layer1,
                                                      stride=2)

   conv_layer2 = pygad.cnn.Conv2D(num_filters=3,
                                  kernel_size=3,
                                  previous_layer=average_pooling_layer,
                                  activation_function=None)
   relu_layer2 = pygad.cnn.ReLU(previous_layer=conv_layer2)
   max_pooling_layer = pygad.cnn.MaxPooling2D(pool_size=2, 
                                              previous_layer=relu_layer2,
                                              stride=2)

   conv_layer3 = pygad.cnn.Conv2D(num_filters=1,
                                  kernel_size=3,
                                  previous_layer=max_pooling_layer,
                                  activation_function=None)
   relu_layer3 = pygad.cnn.ReLU(previous_layer=conv_layer3)
   pooling_layer = pygad.cnn.AveragePooling2D(pool_size=2, 
                                              previous_layer=relu_layer3,
                                              stride=2)

   flatten_layer = pygad.cnn.Flatten(previous_layer=pooling_layer)
   dense_layer1 = pygad.cnn.Dense(num_neurons=100, 
                                  previous_layer=flatten_layer,
                                  activation_function="relu")
   dense_layer2 = pygad.cnn.Dense(num_neurons=4, 
                                  previous_layer=dense_layer1,
                                  activation_function="softmax")

After the network architecture is prepared, the next step is to create a
CNN model.

Building Model
--------------

The CNN model is created as an instance of the ``pygad.cnn.Model``
class. Here is an example.

.. code:: python

   model = pygad.cnn.Model(last_layer=dense_layer2,
                           epochs=5,
                           learning_rate=0.01)

After the model is created, a summary of the model architecture can be
printed.

Model Summary
-------------

The ``summary()`` method in the ``pygad.cnn.Model`` class prints a
summary of the CNN model.

.. code:: python

   model.summary()

.. code:: python

   ----------Network Architecture----------
   <class 'pygad.cnn.Conv2D'>
   <class 'pygad.cnn.Sigmoid'>
   <class 'pygad.cnn.AveragePooling2D'>
   <class 'pygad.cnn.Conv2D'>
   <class 'pygad.cnn.ReLU'>
   <class 'pygad.cnn.MaxPooling2D'>
   <class 'pygad.cnn.Conv2D'>
   <class 'pygad.cnn.ReLU'>
   <class 'pygad.cnn.AveragePooling2D'>
   <class 'pygad.cnn.Flatten'>
   <class 'pygad.cnn.Dense'>
   <class 'pygad.cnn.Dense'>
   ----------------------------------------

Training the Network
--------------------

After the model and the data are prepared, then the model can be trained
using the the ``pygad.cnn.train()`` method.

.. code:: python

   model.train(train_inputs=train_inputs, 
               train_outputs=train_outputs)

After training the network, the next step is to make predictions.

Making Predictions
------------------

The ``pygad.cnn.predict()`` method uses the trained network for making
predictions. Here is an example.

.. code:: python

   predictions = model.predict(data_inputs=train_inputs)

It is not expected to have high accuracy in the predictions because no
training algorithm is used.

Calculating Some Statistics
---------------------------

Based on the predictions the network made, some statistics can be
calculated such as the number of correct and wrong predictions in
addition to the classification accuracy.

.. code:: python

   num_wrong = numpy.where(predictions != train_outputs)[0]
   num_correct = train_outputs.size - num_wrong.size
   accuracy = 100 * (num_correct/train_outputs.size)
   print("Number of correct classifications : {num_correct}.".format(num_correct=num_correct))
   print("Number of wrong classifications : {num_wrong}.".format(num_wrong=num_wrong.size))
   print("Classification accuracy : {accuracy}.".format(accuracy=accuracy))

It is very important to note that it is not expected that the
classification accuracy is high because no training algorithm is used.
Please check the documentation of the ``pygad.gacnn`` module for
training the CNN using the genetic algorithm.

Examples
========

This section gives the complete code of some examples that build neural
networks using ``pygad.cnn``. Each subsection builds a different
network.

Image Classification
--------------------

This example is discussed in the **Steps to Build a Convolutional Neural
Network** section and its complete code is listed below.

Remember to either download or create the
`dataset_features.npy <https://github.com/ahmedfgad/NumPyCNN/blob/master/dataset_features.npy>`__
and
`dataset_outputs.npy <https://github.com/ahmedfgad/NumPyCNN/blob/master/dataset_outputs.npy>`__
files before running this code.

.. code:: python

   import numpy
   import pygad.cnn

   """
   Convolutional neural network implementation using NumPy
   A tutorial that helps to get started (Building Convolutional Neural Network using NumPy from Scratch) available in these links: 
       https://www.linkedin.com/pulse/building-convolutional-neural-network-using-numpy-from-ahmed-gad
       https://towardsdatascience.com/building-convolutional-neural-network-using-numpy-from-scratch-b30aac50e50a
       https://www.kdnuggets.com/2018/04/building-convolutional-neural-network-numpy-scratch.html
   It is also translated into Chinese: http://m.aliyun.com/yunqi/articles/585741
   """

   train_inputs = numpy.load("dataset_inputs.npy")
   train_outputs = numpy.load("dataset_outputs.npy")

   sample_shape = train_inputs.shape[1:]
   num_classes = 4

   input_layer = pygad.cnn.Input2D(input_shape=sample_shape)
   conv_layer1 = pygad.cnn.Conv2D(num_filters=2,
                                  kernel_size=3,
                                  previous_layer=input_layer,
                                  activation_function=None)
   relu_layer1 = pygad.cnn.Sigmoid(previous_layer=conv_layer1)
   average_pooling_layer = pygad.cnn.AveragePooling2D(pool_size=2, 
                                                      previous_layer=relu_layer1,
                                                      stride=2)

   conv_layer2 = pygad.cnn.Conv2D(num_filters=3,
                                  kernel_size=3,
                                  previous_layer=average_pooling_layer,
                                  activation_function=None)
   relu_layer2 = pygad.cnn.ReLU(previous_layer=conv_layer2)
   max_pooling_layer = pygad.cnn.MaxPooling2D(pool_size=2, 
                                              previous_layer=relu_layer2,
                                              stride=2)

   conv_layer3 = pygad.cnn.Conv2D(num_filters=1,
                                  kernel_size=3,
                                  previous_layer=max_pooling_layer,
                                  activation_function=None)
   relu_layer3 = pygad.cnn.ReLU(previous_layer=conv_layer3)
   pooling_layer = pygad.cnn.AveragePooling2D(pool_size=2, 
                                              previous_layer=relu_layer3,
                                              stride=2)

   flatten_layer = pygad.cnn.Flatten(previous_layer=pooling_layer)
   dense_layer1 = pygad.cnn.Dense(num_neurons=100, 
                                  previous_layer=flatten_layer,
                                  activation_function="relu")
   dense_layer2 = pygad.cnn.Dense(num_neurons=num_classes, 
                                  previous_layer=dense_layer1,
                                  activation_function="softmax")

   model = pygad.cnn.Model(last_layer=dense_layer2,
                           epochs=1,
                           learning_rate=0.01)

   model.summary()

   model.train(train_inputs=train_inputs, 
               train_outputs=train_outputs)

   predictions = model.predict(data_inputs=train_inputs)
   print(predictions)

   num_wrong = numpy.where(predictions != train_outputs)[0]
   num_correct = train_outputs.size - num_wrong.size
   accuracy = 100 * (num_correct/train_outputs.size)
   print("Number of correct classifications : {num_correct}.".format(num_correct=num_correct))
   print("Number of wrong classifications : {num_wrong}.".format(num_wrong=num_wrong.size))
   print("Classification accuracy : {accuracy}.".format(accuracy=accuracy))
