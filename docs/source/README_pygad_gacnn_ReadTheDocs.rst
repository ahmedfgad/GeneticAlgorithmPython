.. _pygadgacnn-module:

``pygad.gacnn`` Module
======================

This section of the PyGAD's library documentation discusses the
**pygad.gacnn** module.

The ``pygad.gacnn`` module trains convolutional neural networks using
the genetic algorithm. It makes use of the 2 modules ``pygad`` and
``pygad.cnn``.

.. _pygadgacnngacnn-class:

``pygad.gacnn.GACNN`` Class
===========================

The ``pygad.gacnn`` module has a class named ``pygad.gacnn.GACNN`` for
training convolutional neural networks (CNNs) using the genetic
algorithm. The constructor, methods, function, and attributes within the
class are discussed in this section.

.. _init:

``__init__()``
--------------

In order to train a CNN using the genetic algorithm, the first thing to
do is to create an instance of the ``pygad.gacnn.GACNN`` class.

The ``pygad.gacnn.GACNN`` class constructor accepts the following
parameters:

-  ``model``: model: An instance of the pygad.cnn.Model class
   representing the architecture of all solutions in the population.

-  ``num_solutions``: Number of CNNs (i.e. solutions) in the population.
   Based on the value passed to this parameter, a number of identical
   CNNs are created where their parameters are optimized using the
   genetic algorithm.

Instance Attributes
-------------------

All the parameters in the ``pygad.gacnn.GACNN`` class constructor are
used as instance attributes. Besides such attributes, there is an extra
attribute added to the instances from the ``pygad.gacnn.GACNN`` class
which is:

-  ``population_networks``: A list holding references to all the
   solutions (i.e. CNNs) used in the population.

Methods in the GACNN Class
--------------------------

This section discusses the methods available for instances of the
``pygad.gacnn.GACNN`` class.

.. _createpopulation:

``create_population()``
~~~~~~~~~~~~~~~~~~~~~~~

The ``create_population()`` method creates the initial population of the
genetic algorithm as a list of CNNs (i.e. solutions). All the networks
are copied from the CNN model passed to constructor of the GACNN class.

The list of networks is assigned to the ``population_networks``
attribute of the instance.

.. _updatepopulationtrainedweights:

``update_population_trained_weights()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``update_population_trained_weights()`` method updates the
``trained_weights`` attribute of the layers of each network (check the
documentation of the ``pygad.cnn`` module) for more information)
according to the weights passed in the ``population_trained_weights``
parameter.

Accepts the following parameters:

-  ``population_trained_weights``: A list holding the trained weights of
   all networks as matrices. Such matrices are to be assigned to the
   ``trained_weights`` attribute of all layers of all networks.

.. _functions-in-the-pygadgacnn-module:

Functions in the ``pygad.gacnn`` Module
=======================================

This section discusses the functions in the ``pygad.gacnn`` module.

.. _pygadgacnnpopulationasvectors:

``pygad.gacnn.population_as_vectors()`` 
---------------------------------------

Accepts the population as a list of references to the
``pygad.cnn.Model`` class and returns a list holding all weights of the
layers of each solution (i.e. network) in the population as a vector.

For example, if the population has 6 solutions (i.e. networks), this
function accepts references to such networks and returns a list with 6
vectors, one for each network (i.e. solution). Each vector holds the
weights for all layers for a single network.

Accepts the following parameters:

-  ``population_networks``: A list holding references to the
   ``pygad.cnn.Model`` class of the networks used in the population.

Returns a list holding the weights vectors for all solutions (i.e.
networks).

.. _pygadgacnnpopulationasmatrices:

``pygad.gacnn.population_as_matrices()``
----------------------------------------

Accepts the population as both networks and weights vectors and returns
the weights of all layers of each solution (i.e. network) in the
population as a matrix.

For example, if the population has 6 solutions (i.e. networks), this
function returns a list with 6 matrices, one for each network holding
its weights for all layers.

Accepts the following parameters:

-  ``population_networks``: A list holding references to the
   ``pygad.cnn.Model`` class of the networks used in the population.

-  ``population_vectors``: A list holding the weights of all networks as
   vectors. Such vectors are to be converted into matrices.

Returns a list holding the weights matrices for all solutions (i.e.
networks).

Steps to Build and Train CNN using Genetic Algorithm
====================================================

The steps to use this project for building and training a neural network
using the genetic algorithm are as follows:

-  Prepare the training data.

-  Create an instance of the ``pygad.gacnn.GACNN`` class.

-  Fetch the population weights as vectors.

-  Prepare the fitness function.

-  Prepare the generation callback function.

-  Create an instance of the ``pygad.GA`` class.

-  Run the created instance of the ``pygad.GA`` class.

-  Plot the Fitness Values

-  Information about the best solution.

-  Making predictions using the trained weights.

-  Calculating some statistics.

Let's start covering all of these steps.

Prepare the Training Data
-------------------------

Before building and training neural networks, the training data (input
and output) is to be prepared. The inputs and the outputs of the
training data are NumPy arrays.

The data used in this example is available as 2 files:

1. `dataset_inputs.npy <https://github.com/ahmedfgad/NumPyCNN/blob/master/dataset_inputs.npy>`__:
   Data inputs.
   https://github.com/ahmedfgad/NumPyCNN/blob/master/dataset_inputs.npy

2. `dataset_outputs.npy <https://github.com/ahmedfgad/NumPyCNN/blob/master/dataset_outputs.npy>`__:
   Class labels.
   https://github.com/ahmedfgad/NumPyCNN/blob/master/dataset_outputs.npy

The data consists of 4 classes of images. The image shape is
``(100, 100, 3)`` and there are 20 images per class. For more
information about the dataset, check the **Reading the Data** section of
the ``pygad.cnn`` module.

Simply download these 2 files and read them according to the next code.

.. code:: python

   import numpy

   train_inputs = numpy.load("dataset_inputs.npy")
   train_outputs = numpy.load("dataset_outputs.npy")

For the output array, each element must be a single number representing
the class label of the sample. The class labels must start at ``0``. So,
if there are 80 samples, then the shape of the output array is ``(80)``.
If there are 5 classes in the data, then the values of all the 200
elements in the output array must range from 0 to 4 inclusive.
Generally, the class labels start from ``0`` to ``N-1`` where ``N`` is
the number of classes.

Note that the project only supports that each sample is assigned to only
one class.

Building the Network Architecture
---------------------------------

Here is an example for a CNN architecture.

.. code:: python

   import pygad.cnn

   input_layer = pygad.cnn.Input2D(input_shape=(80, 80, 3))
   conv_layer = pygad.cnn.Conv2D(num_filters=2,
                                 kernel_size=3,
                                 previous_layer=input_layer,
                                 activation_function="relu")
   average_pooling_layer = pygad.cnn.AveragePooling2D(pool_size=5, 
                                                      previous_layer=conv_layer,
                                                      stride=3)

   flatten_layer = pygad.cnn.Flatten(previous_layer=average_pooling_layer)
   dense_layer = pygad.cnn.Dense(num_neurons=4, 
                                 previous_layer=flatten_layer,
                                 activation_function="softmax")

After the network architecture is prepared, the next step is to create a
CNN model.

Building Model
--------------

The CNN model is created as an instance of the ``pygad.cnn.Model``
class. Here is an example.

.. code:: python

   model = pygad.cnn.Model(last_layer=dense_layer,
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
   <class 'cnn.Conv2D'>
   <class 'cnn.AveragePooling2D'>
   <class 'cnn.Flatten'>
   <class 'cnn.Dense'>
   ----------------------------------------

The next step is to create an instance of the ``pygad.gacnn.GACNN``
class.

.. _create-an-instance-of-the-pygadgacnngacnn-class:

Create an Instance of the ``pygad.gacnn.GACNN`` Class
-----------------------------------------------------

After preparing the input data and building the CNN model, an instance
of the ``pygad.gacnn.GACNN`` class is created by passing the appropriate
parameters.

Here is an example where the ``num_solutions`` parameter is set to 4
which means the genetic algorithm population will have 6 solutions (i.e.
networks). All of these 6 CNNs will have the same architectures as
specified by the ``model`` parameter.

.. code:: python

   import pygad.gacnn

   GACNN_instance = pygad.gacnn.GACNN(model=model,
                                      num_solutions=4)

After creating the instance of the ``pygad.gacnn.GACNN`` class, next is
to fetch the weights of the population as a list of vectors.

Fetch the Population Weights as Vectors
---------------------------------------

For the genetic algorithm, the parameters (i.e. genes) of each solution
are represented as a single vector.

For this task, the weights of each CNN must be available as a single
vector. In other words, the weights of all layers of a CNN must be
grouped into a vector.

To create a list holding the population weights as vectors, one for each
network, the ``pygad.gacnn.population_as_vectors()`` function is used.

.. code:: python

   population_vectors = gacnn.population_as_vectors(population_networks=GACNN_instance.population_networks)

Such population of vectors is used as the initial population.

.. code:: python

   initial_population = population_vectors.copy()

After preparing the population weights as a set of vectors, next is to
prepare 2 functions which are:

1. Fitness function.

2. Callback function after each generation.

Prepare the Fitness Function
----------------------------

The PyGAD library works by allowing the users to customize the genetic
algorithm for their own problems. Because the problems differ in how the
fitness values are calculated, then PyGAD allows the user to use a
custom function as a maximization fitness function. This function must
accept 2 positional parameters representing the following:

-  The solution.

-  The solution index in the population.

The fitness function must return a single number representing the
fitness. The higher the fitness value, the better the solution.

Here is the implementation of the fitness function for training a CNN.

It uses the ``pygad.cnn.predict()`` function to predict the class labels
based on the current solution's weights. The ``pygad.cnn.predict()``
function uses the trained weights available in the ``trained_weights``
attribute of each layer of the network for making predictions.

Based on such predictions, the classification accuracy is calculated.
This accuracy is used as the fitness value of the solution. Finally, the
fitness value is returned.

.. code:: python

   def fitness_func(solution, sol_idx):
       global GACNN_instance, data_inputs, data_outputs

       predictions = GACNN_instance.population_networks[sol_idx].predict(data_inputs=data_inputs)
       correct_predictions = numpy.where(predictions == data_outputs)[0].size
       solution_fitness = (correct_predictions/data_outputs.size)*100

       return solution_fitness

Prepare the Generation Callback Function
----------------------------------------

After each generation of the genetic algorithm, the fitness function
will be called to calculate the fitness value of each solution. Within
the fitness function, the ``pygad.cnn.predict()`` function is used for
predicting the outputs based on the current solution's
``trained_weights`` attribute. Thus, it is required that such an
attribute is updated by weights evolved by the genetic algorithm after
each generation.

PyGAD has a parameter accepted by the ``pygad.GA`` class constructor
named ``on_generation``. It could be assigned to a function that is
called after each generation. The function must accept a single
parameter representing the instance of the ``pygad.GA`` class.

This callback function can be used to update the ``trained_weights``
attribute of layers of each network in the population.

Here is the implementation for a function that updates the
``trained_weights`` attribute of the layers of the population networks.

It works by converting the current population from the vector form to
the matric form using the ``pygad.gacnn.population_as_matrices()``
function. It accepts the population as vectors and returns it as
matrices.

The population matrices are then passed to the
``update_population_trained_weights()`` method in the ``pygad.gacnn``
module to update the ``trained_weights`` attribute of all layers for all
solutions within the population.

.. code:: python

   def callback_generation(ga_instance):
       global GACNN_instance, last_fitness

       population_matrices = gacnn.population_as_matrices(population_networks=GACNN_instance.population_networks, population_vectors=ga_instance.population)
       GACNN_instance.update_population_trained_weights(population_trained_weights=population_matrices)

       print("Generation = {generation}".format(generation=ga_instance.generations_completed))

After preparing the fitness and callback function, next is to create an
instance of the ``pygad.GA`` class.

.. _create-an-instance-of-the-pygadga-class:

Create an Instance of the ``pygad.GA`` Class
--------------------------------------------

Once the parameters of the genetic algorithm are prepared, an instance
of the ``pygad.GA`` class can be created. Here is an example where the
number of generations is 10.

.. code:: python

   import pygad

   num_parents_mating = 4 

   num_generations = 10

   mutation_percent_genes = 5

   ga_instance = pygad.GA(num_generations=num_generations, 
                          num_parents_mating=num_parents_mating, 
                          initial_population=initial_population,
                          fitness_func=fitness_func,
                          mutation_percent_genes=mutation_percent_genes,
                          on_generation=callback_generation)

The last step for training the neural networks using the genetic
algorithm is calling the ``run()`` method.

.. _run-the-created-instance-of-the-pygadga-class:

Run the Created Instance of the ``pygad.GA`` Class
--------------------------------------------------

By calling the ``run()`` method from the ``pygad.GA`` instance, the
genetic algorithm will iterate through the number of generations
specified in its ``num_generations`` parameter.

.. code:: python

   ga_instance.run()

Plot the Fitness Values
-----------------------

After the ``run()`` method completes, the ``plot_fitness()`` method can
be called to show how the fitness values evolve by generation.

.. code:: python

   ga_instance.plot_fitness()

.. figure:: https://user-images.githubusercontent.com/16560492/83429675-ab744580-a434-11ea-8f21-9d3804b50d15.png
   :alt: 

Information about the Best Solution
-----------------------------------

The following information about the best solution in the last population
is returned using the ``best_solution()`` method in the ``pygad.GA``
class.

-  Solution

-  Fitness value of the solution

-  Index of the solution within the population

Here is how such information is returned.

.. code:: python

   solution, solution_fitness, solution_idx = ga_instance.best_solution()
   print("Parameters of the best solution : {solution}".format(solution=solution))
   print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
   print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

.. code:: 

   ...
   Fitness value of the best solution = 83.75
   Index of the best solution : 0
   Best fitness value reached after 4 generations.

Making Predictions using the Trained Weights
--------------------------------------------

The ``pygad.cnn.predict()`` function can be used to make predictions
using the trained network. As printed, the network is able to predict
the labels correctly.

.. code:: python

   predictions = pygad.cnn.predict(last_layer=GANN_instance.population_networks[solution_idx], data_inputs=data_inputs)
   print("Predictions of the trained network : {predictions}".format(predictions=predictions))

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

.. code:: 

   Number of correct classifications : 67.
   Number of wrong classifications : 13.
   Classification accuracy : 83.75.

Examples
========

This section gives the complete code of some examples that build and
train neural networks using the genetic algorithm. Each subsection
builds a different network.

Image Classification
--------------------

This example is discussed in the **Steps to Build and Train CNN using
Genetic Algorithm** section that builds the an image classifier. Its
complete code is listed below.

.. code:: python

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

   def fitness_func(solution, sol_idx):
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

   data_inputs = numpy.load("dataset_inputs.npy")
   data_outputs = numpy.load("dataset_outputs.npy")

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

   ga_instance = pygad.GA(num_generations=num_generations, 
                          num_parents_mating=num_parents_mating, 
                          initial_population=initial_population,
                          fitness_func=fitness_func,
                          mutation_percent_genes=mutation_percent_genes,
                          on_generation=callback_generation)

   ga_instance.run()

   # After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
   ga_instance.plot_fitness()

   # Returning the details of the best solution.
   solution, solution_fitness, solution_idx = ga_instance.best_solution()
   print("Parameters of the best solution : {solution}".format(solution=solution))
   print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
   print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

   if ga_instance.best_solution_generation != -1:
       print("Best fitness value reached after {best_solution_generation} generations.".format(best_solution_generation=ga_instance.best_solution_generation))

   # Predicting the outputs of the data using the best solution.
   predictions = GACNN_instance.population_networks[solution_idx].predict(data_inputs=data_inputs)
   print("Predictions of the trained network : {predictions}".format(predictions=predictions))

   # Calculating some statistics
   num_wrong = numpy.where(predictions != data_outputs)[0]
   num_correct = data_outputs.size - num_wrong.size
   accuracy = 100 * (num_correct/data_outputs.size)
   print("Number of correct classifications : {num_correct}.".format(num_correct=num_correct))
   print("Number of wrong classifications : {num_wrong}.".format(num_wrong=num_wrong.size))
   print("Classification accuracy : {accuracy}.".format(accuracy=accuracy))
