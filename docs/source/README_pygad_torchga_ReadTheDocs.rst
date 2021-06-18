.. _pygadtorchga-module:

``pygad.torchga`` Module
========================

This section of the PyGAD's library documentation discusses the
**pygad.torchga** module.

The ``pygad.torchga`` module has helper a class and 2 functions to train
PyTorch models using the genetic algorithm (PyGAD).

The contents of this module are:

1. ``TorchGA``: A class for creating an initial population of all
   parameters in the PyTorch model.

2. ``model_weights_as_vector()``: A function to reshape the PyTorch
   model weights to a single vector.

3. ``model_weights_as_dict()``: A function to restore the PyTorch model
   weights from a vector.

4. ``predict()``: A function to make predictions based on the PyTorch
   model and a solution.

More details are given in the next sections.

Steps Summary
=============

The summary of the steps used to train a PyTorch model using PyGAD is as
follows:

1. Create a PyTorch model.

2. Create an instance of the ``pygad.torchga.TorchGA`` class.

3. Prepare the training data.

4. Build the fitness function.

5. Create an instance of the ``pygad.GA`` class.

6. Run the genetic algorithm.

Create PyTorch Model
====================

Before discussing training a PyTorch model using PyGAD, the first thing
to do is to create the PyTorch model. To get started, please check the
`PyTorch library
documentation <https://pytorch.org/docs/stable/index.html>`__.

Here is an example of a PyTorch model.

.. code:: python

   import torch

   input_layer = torch.nn.Linear(3, 5)
   relu_layer = torch.nn.ReLU()
   output_layer = torch.nn.Linear(5, 1)

   model = torch.nn.Sequential(input_layer,
                               relu_layer,
                               output_layer)

Feel free to add the layers of your choice.

.. _pygadtorchgatorchga-class:

``pygad.torchga.TorchGA`` Class
===============================

The ``pygad.torchga`` module has a class named ``TorchGA`` for creating
an initial population for the genetic algorithm based on a PyTorch
model. The constructor, methods, and attributes within the class are
discussed in this section.

.. _init:

``__init__()``
--------------

The ``pygad.torchga.TorchGA`` class constructor accepts the following
parameters:

-  ``model``: An instance of the PyTorch model.

-  ``num_solutions``: Number of solutions in the population. Each
   solution has different parameters of the model.

Instance Attributes
-------------------

All parameters in the ``pygad.torchga.TorchGA`` class constructor are
used as instance attributes in addition to adding a new attribute called
``population_weights``.

Here is a list of all instance attributes:

-  ``model``

-  ``num_solutions``

-  ``population_weights``: A nested list holding the weights of all
   solutions in the population.

Methods in the ``TorchGA`` Class
--------------------------------

This section discusses the methods available for instances of the
``pygad.torchga.TorchGA`` class.

.. _createpopulation:

``create_population()``
~~~~~~~~~~~~~~~~~~~~~~~

The ``create_population()`` method creates the initial population of the
genetic algorithm as a list of solutions where each solution represents
different model parameters. The list of networks is assigned to the
``population_weights`` attribute of the instance.

.. _functions-in-the-pygadtorchga-module:

Functions in the ``pygad.torchga`` Module
=========================================

This section discusses the functions in the ``pygad.torchga`` module.

.. _pygadtorchgamodelweightsasvector:

``pygad.torchga.model_weights_as_vector()`` 
-------------------------------------------

The ``model_weights_as_vector()`` function accepts a single parameter
named ``model`` representing the PyTorch model. It returns a vector
holding all model weights. The reason for representing the model weights
as a vector is that the genetic algorithm expects all parameters of any
solution to be in a 1D vector form.

The function accepts the following parameters:

-  ``model``: The PyTorch model.

It returns a 1D vector holding the model weights.

.. _pygadtorchmodelweightsasdict:

``pygad.torch.model_weights_as_dict()``
---------------------------------------

The ``model_weights_as_dict()`` function accepts the following
parameters:

1. ``model``: The PyTorch model.

2. ``weights_vector``: The model parameters as a vector.

It returns the restored model weights in the same form used by the
``state_dict()`` method. The returned dictionary is ready to be passed
to the ``load_state_dict()`` method for setting the PyTorch model's
parameters.

.. _pygadtorchgapredict:

``pygad.torchga.predict()``
---------------------------

The ``predict()`` function makes a prediction based on a solution. It
accepts the following parameters:

1. ``model``: The PyTorch model.

2. ``solution``: The solution evolved.

3. ``data``: The test data inputs.

It returns the predictions for the data samples.

Examples
========

This section gives the complete code of some examples that build and
train a PyTorch model using PyGAD. Each subsection builds a different
network.

Example 1: Regression Example
-----------------------------

The next code builds a simple PyTorch model for regression. The next
subsections discuss each part in the code.

.. code:: python

   import torch
   import torchga
   import pygad

   def fitness_func(solution, sol_idx):
       global data_inputs, data_outputs, torch_ga, model, loss_function

       predictions = pygad.torchga.predict(model=model, 
                                           solution=solution, 
                                           data=data_inputs)

       abs_error = loss_function(predictions, data_outputs).detach().numpy() + 0.00000001

       solution_fitness = 1.0 / abs_error

       return solution_fitness

   def callback_generation(ga_instance):
       print("Generation = {generation}".format(generation=ga_instance.generations_completed))
       print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

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

   # Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
   num_generations = 250 # Number of generations.
   num_parents_mating = 5 # Number of solutions to be selected as parents in the mating pool.
   initial_population = torch_ga.population_weights # Initial population of network weights

   ga_instance = pygad.GA(num_generations=num_generations, 
                          num_parents_mating=num_parents_mating, 
                          initial_population=initial_population,
                          fitness_func=fitness_func,
                          on_generation=callback_generation)

   ga_instance.run()

   # After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
   ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

   # Returning the details of the best solution.
   solution, solution_fitness, solution_idx = ga_instance.best_solution()
   print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
   print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

   # Make predictions based on the best solution.
   predictions = pygad.torchga.predict(model=model, 
                                       solution=solution, 
                                       data=data_inputs)
   print("Predictions : \n", predictions.detach().numpy())

   abs_error = loss_function(predictions, data_outputs)
   print("Absolute Error : ", abs_error.detach().numpy())

Create a PyTorch model
~~~~~~~~~~~~~~~~~~~~~~

According to the steps mentioned previously, the first step is to create
a PyTorch model. Here is the code that builds the model using the
Functional API.

.. code:: python

   import torch

   input_layer = torch.nn.Linear(3, 5)
   relu_layer = torch.nn.ReLU()
   output_layer = torch.nn.Linear(5, 1)

   model = torch.nn.Sequential(input_layer,
                               relu_layer,
                               output_layer)

.. _create-an-instance-of-the-pygadtorchgatorchga-class:

Create an Instance of the ``pygad.torchga.TorchGA`` Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The second step is to create an instance of the
``pygad.torchga.TorchGA`` class. There are 10 solutions per population.
Change this number according to your needs.

.. code:: python

   import pygad.torchga

   torch_ga = torchga.TorchGA(model=model,
                              num_solutions=10)

.. _prepare-the-training-data-1:

Prepare the Training Data
~~~~~~~~~~~~~~~~~~~~~~~~~

The third step is to prepare the training data inputs and outputs. Here
is an example where there are 4 samples. Each sample has 3 inputs and 1
output.

.. code:: python

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

Build the Fitness Function
~~~~~~~~~~~~~~~~~~~~~~~~~~

The fourth step is to build the fitness function. This function must
accept 2 parameters representing the solution and its index within the
population.

The next fitness function calculates the mean absolute error (MAE) of
the PyTorch model based on the parameters in the solution. The
reciprocal of the MAE is used as the fitness value. Feel free to use any
other loss function to calculate the fitness value.

.. code:: python

   loss_function = torch.nn.L1Loss()

   def fitness_func(solution, sol_idx):
       global data_inputs, data_outputs, torch_ga, model, loss_function

       predictions = pygad.torchga.predict(model=model, 
                                           solution=solution, 
                                           data=data_inputs)

       abs_error = loss_function(predictions, data_outputs).detach().numpy() + 0.00000001

       solution_fitness = 1.0 / abs_error

       return solution_fitness

.. _create-an-instance-of-the-pygadga-class:

Create an Instance of the ``pygad.GA`` Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The fifth step is to instantiate the ``pygad.GA`` class. Note how the
``initial_population`` parameter is assigned to the initial weights of
the PyTorch models.

For more information, please check the `parameters this class
accepts <https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#init>`__.

.. code:: python

   # Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
   num_generations = 250 # Number of generations.
   num_parents_mating = 5 # Number of solutions to be selected as parents in the mating pool.
   initial_population = torch_ga.population_weights # Initial population of network weights

   ga_instance = pygad.GA(num_generations=num_generations, 
                          num_parents_mating=num_parents_mating, 
                          initial_population=initial_population,
                          fitness_func=fitness_func,
                          on_generation=callback_generation)

Run the Genetic Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~

The sixth and last step is to run the genetic algorithm by calling the
``run()`` method.

.. code:: python

   ga_instance.run()

After the PyGAD completes its execution, then there is a figure that
shows how the fitness value changes by generation. Call the
``plot_fitness()`` method to show the figure.

.. code:: python

   ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

Here is the figure.

.. figure:: https://user-images.githubusercontent.com/16560492/103469779-22f5b480-4d37-11eb-80dc-95503065ebb1.png
   :alt: 

To get information about the best solution found by PyGAD, use the
``best_solution()`` method.

.. code:: python

   # Returning the details of the best solution.
   solution, solution_fitness, solution_idx = ga_instance.best_solution()
   print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
   print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

.. code:: python

   Fitness value of the best solution = 145.42425295191546
   Index of the best solution : 0

The next code restores the trained model weights using the
``model_weights_as_dict()`` function. The restored weights are used to
calculate the predicted values.

.. code:: python

   predictions = pygad.torchga.predict(model=model, 
                                       solution=solution, 
                                       data=data_inputs)
   print("Predictions : \n", predictions.detach().numpy())

.. code:: python

   Predictions : 
   [[0.08401088]
    [0.60939324]
    [1.3010881 ]
    [2.5010352 ]]

The next code measures the trained model error.

.. code:: python

   abs_error = loss_function(predictions, data_outputs)
   print("Absolute Error : ", abs_error.detach().numpy())

.. code:: 

   Absolute Error :  0.006876422

Example 2: XOR Binary Classification
------------------------------------

The next code creates a PyTorch model to build the XOR binary
classification problem. Let's highlight the changes compared to the
previous example.

.. code:: python

   import torch
   import torchga
   import pygad

   def fitness_func(solution, sol_idx):
       global data_inputs, data_outputs, torch_ga, model, loss_function

       predictions = pygad.torchga.predict(model=model, 
                                           solution=solution, 
                                           data=data_inputs)

       solution_fitness = 1.0 / (loss_function(predictions, data_outputs).detach().numpy() + 0.00000001)

       return solution_fitness

   def callback_generation(ga_instance):
       print("Generation = {generation}".format(generation=ga_instance.generations_completed))
       print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

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

   # Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
   num_generations = 250 # Number of generations.
   num_parents_mating = 5 # Number of solutions to be selected as parents in the mating pool.
   initial_population = torch_ga.population_weights # Initial population of network weights.

   # Create an instance of the pygad.GA class
   ga_instance = pygad.GA(num_generations=num_generations, 
                          num_parents_mating=num_parents_mating, 
                          initial_population=initial_population,
                          fitness_func=fitness_func,
                          on_generation=callback_generation)

   # Start the genetic algorithm evolution.
   ga_instance.run()

   # After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
   ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

   # Returning the details of the best solution.
   solution, solution_fitness, solution_idx = ga_instance.best_solution()
   print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
   print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

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

Compared to the previous regression example, here are the changes:

-  The PyTorch model is changed according to the nature of the problem.
   Now, it has 2 inputs and 2 outputs with an in-between hidden layer of
   4 neurons.

.. code:: python

   input_layer  = torch.nn.Linear(2, 4)
   relu_layer = torch.nn.ReLU()
   dense_layer = torch.nn.Linear(4, 2)
   output_layer = torch.nn.Softmax(1)

   model = torch.nn.Sequential(input_layer,
                               relu_layer,
                               dense_layer,
                               output_layer)

-  The train data is changed. Note that the output of each sample is a
   1D vector of 2 values, 1 for each class.

.. code:: python

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

-  The fitness value is calculated based on the binary cross entropy.

.. code:: python

   loss_function = torch.nn.BCELoss()

After the previous code completes, the next figure shows how the fitness
value change by generation.

.. figure:: https://user-images.githubusercontent.com/16560492/103469818-c646c980-4d37-11eb-98c3-d9d591acd5e2.png
   :alt: 

Here is some information about the trained model. Its fitness value is
``100000000.0``, loss is ``0.0`` and accuracy is 100%.

.. code:: python

   Fitness value of the best solution = 100000000.0

   Index of the best solution : 0

   Predictions : 
   [[1.0000000e+00 1.3627675e-10]
    [3.8521746e-09 1.0000000e+00]
    [4.2789325e-10 1.0000000e+00]
    [1.0000000e+00 3.3668417e-09]]

   Binary Crossentropy :  0.0

   Accuracy :  1.0

Example 3: Image Multi-Class Classification (Dense Layers)
----------------------------------------------------------

Here is the code.

.. code:: python

   import torch
   import torchga
   import pygad
   import numpy

   def fitness_func(solution, sol_idx):
       global data_inputs, data_outputs, torch_ga, model, loss_function

       predictions = pygad.torchga.predict(model=model, 
                                           solution=solution, 
                                           data=data_inputs)

       solution_fitness = 1.0 / (loss_function(predictions, data_outputs).detach().numpy() + 0.00000001)

       return solution_fitness

   def callback_generation(ga_instance):
       print("Generation = {generation}".format(generation=ga_instance.generations_completed))
       print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

   # Build the PyTorch model using the functional API.
   input_layer = torch.nn.Linear(360, 50)
   relu_layer = torch.nn.ReLU()
   dense_layer = torch.nn.Linear(50, 4)
   output_layer = torch.nn.Softmax(1)

   model = torch.nn.Sequential(input_layer,
                               relu_layer,
                               dense_layer,
                               output_layer)

   # Create an instance of the pygad.torchga.TorchGA class to build the initial population.
   torch_ga = torchga.TorchGA(model=model,
                              num_solutions=10)

   loss_function = torch.nn.CrossEntropyLoss()

   # Data inputs
   data_inputs = torch.from_numpy(numpy.load("dataset_features.npy")).float()

   # Data outputs
   data_outputs = torch.from_numpy(numpy.load("outputs.npy")).long()
   # The next 2 lines are equivelant to this Keras function to perform 1-hot encoding: tensorflow.keras.utils.to_categorical(data_outputs)
   # temp_outs = numpy.zeros((data_outputs.shape[0], numpy.unique(data_outputs).size), dtype=numpy.uint8)
   # temp_outs[numpy.arange(data_outputs.shape[0]), numpy.uint8(data_outputs)] = 1

   # Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
   num_generations = 200 # Number of generations.
   num_parents_mating = 5 # Number of solutions to be selected as parents in the mating pool.
   initial_population = torch_ga.population_weights # Initial population of network weights.

   # Create an instance of the pygad.GA class
   ga_instance = pygad.GA(num_generations=num_generations, 
                          num_parents_mating=num_parents_mating, 
                          initial_population=initial_population,
                          fitness_func=fitness_func,
                          on_generation=callback_generation)

   # Start the genetic algorithm evolution.
   ga_instance.run()

   # After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
   ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

   # Returning the details of the best solution.
   solution, solution_fitness, solution_idx = ga_instance.best_solution()
   print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
   print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

   # Fetch the parameters of the best solution.
   best_solution_weights = torchga.model_weights_as_dict(model=model,
                                                           weights_vector=solution)
   model.load_state_dict(best_solution_weights)
   predictions = model(data_inputs)
   # print("Predictions : \n", predictions)

   # Calculate the crossentropy loss of the trained model.
   print("Crossentropy : ", loss_function(predictions, data_outputs).detach().numpy())

   # Calculate the classification accuracy for the trained model.
   accuracy = torch.sum(torch.max(predictions, axis=1).indices == data_outputs) / len(data_outputs)
   print("Accuracy : ", accuracy.detach().numpy())

Compared to the previous binary classification example, this example has
multiple classes (4) and thus the loss is measured using cross entropy.

.. code:: python

   loss_function = torch.nn.CrossEntropyLoss()

.. _prepare-the-training-data-2:

Prepare the Training Data
~~~~~~~~~~~~~~~~~~~~~~~~~

Before building and training neural networks, the training data (input
and output) needs to be prepared. The inputs and the outputs of the
training data are NumPy arrays.

The data used in this example is available as 2 files:

1. `dataset_features.npy <https://github.com/ahmedfgad/NumPyANN/blob/master/dataset_features.npy>`__:
   Data inputs.
   https://github.com/ahmedfgad/NumPyANN/blob/master/dataset_features.npy

2. `outputs.npy <https://github.com/ahmedfgad/NumPyANN/blob/master/outputs.npy>`__:
   Class labels.
   https://github.com/ahmedfgad/NumPyANN/blob/master/outputs.npy

The data consists of 4 classes of images. The image shape is
``(100, 100, 3)``. The number of training samples is 1962. The feature
vector extracted from each image has a length 360.

.. code:: python

   import numpy

   data_inputs = numpy.load("dataset_features.npy")

   data_outputs = numpy.load("outputs.npy")

The next figure shows how the fitness value changes.

.. figure:: https://user-images.githubusercontent.com/16560492/103469855-5d138600-4d38-11eb-84b1-b5eff8faa7bc.png
   :alt: 

Here are some statistics about the trained model.

.. code:: 

   Fitness value of the best solution = 1.3446997034434534
   Index of the best solution : 0
   Crossentropy :  0.74366045
   Accuracy :  1.0

Example 4: Image Multi-Class Classification (Conv Layers)
---------------------------------------------------------

Compared to the previous example that uses only dense layers, this
example uses convolutional layers to classify the same dataset.

Here is the complete code.

.. code:: python

   import torch
   import torchga
   import pygad
   import numpy

   def fitness_func(solution, sol_idx):
       global data_inputs, data_outputs, torch_ga, model, loss_function

       predictions = pygad.torchga.predict(model=model, 
                                           solution=solution, 
                                           data=data_inputs)

       solution_fitness = 1.0 / (loss_function(predictions, data_outputs).detach().numpy() + 0.00000001)

       return solution_fitness

   def callback_generation(ga_instance):
       print("Generation = {generation}".format(generation=ga_instance.generations_completed))
       print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

   # Build the PyTorch model.
   input_layer = torch.nn.Conv2d(in_channels=3, out_channels=5, kernel_size=7)
   relu_layer1 = torch.nn.ReLU()
   max_pool1 = torch.nn.MaxPool2d(kernel_size=5, stride=5)

   conv_layer2 = torch.nn.Conv2d(in_channels=5, out_channels=3, kernel_size=3)
   relu_layer2 = torch.nn.ReLU()

   flatten_layer1 = torch.nn.Flatten()
   # The value 768 is pre-computed by tracing the sizes of the layers' outputs.
   dense_layer1 = torch.nn.Linear(in_features=768, out_features=15)
   relu_layer3 = torch.nn.ReLU()

   dense_layer2 = torch.nn.Linear(in_features=15, out_features=4)
   output_layer = torch.nn.Softmax(1)

   model = torch.nn.Sequential(input_layer,
                               relu_layer1,
                               max_pool1,
                               conv_layer2,
                               relu_layer2,
                               flatten_layer1,
                               dense_layer1,
                               relu_layer3,
                               dense_layer2,
                               output_layer)

   # Create an instance of the pygad.torchga.TorchGA class to build the initial population.
   torch_ga = torchga.TorchGA(model=model,
                              num_solutions=10)

   loss_function = torch.nn.CrossEntropyLoss()

   # Data inputs
   data_inputs = torch.from_numpy(numpy.load("dataset_inputs.npy")).float()
   data_inputs = data_inputs.reshape((data_inputs.shape[0], data_inputs.shape[3], data_inputs.shape[1], data_inputs.shape[2]))

   # Data outputs
   data_outputs = torch.from_numpy(numpy.load("dataset_outputs.npy")).long()

   # Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
   num_generations = 200 # Number of generations.
   num_parents_mating = 5 # Number of solutions to be selected as parents in the mating pool.
   initial_population = torch_ga.population_weights # Initial population of network weights.

   # Create an instance of the pygad.GA class
   ga_instance = pygad.GA(num_generations=num_generations, 
                          num_parents_mating=num_parents_mating, 
                          initial_population=initial_population,
                          fitness_func=fitness_func,
                          on_generation=callback_generation)

   # Start the genetic algorithm evolution.
   ga_instance.run()

   # After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
   ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

   # Returning the details of the best solution.
   solution, solution_fitness, solution_idx = ga_instance.best_solution()
   print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
   print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

   # Make predictions based on the best solution.
   predictions = pygad.torchga.predict(model=model, 
                                       solution=solution, 
                                       data=data_inputs)
   # print("Predictions : \n", predictions)

   # Calculate the crossentropy for the trained model.
   print("Crossentropy : ", loss_function(predictions, data_outputs).detach().numpy())

   # Calculate the classification accuracy for the trained model.
   accuracy = torch.sum(torch.max(predictions, axis=1).indices == data_outputs) / len(data_outputs)
   print("Accuracy : ", accuracy.detach().numpy())

Compared to the previous example, the only change is that the
architecture uses convolutional and max-pooling layers. The shape of
each input sample is 100x100x3.

.. code:: python

   input_layer = torch.nn.Conv2d(in_channels=3, out_channels=5, kernel_size=7)
   relu_layer1 = torch.nn.ReLU()
   max_pool1 = torch.nn.MaxPool2d(kernel_size=5, stride=5)

   conv_layer2 = torch.nn.Conv2d(in_channels=5, out_channels=3, kernel_size=3)
   relu_layer2 = torch.nn.ReLU()

   flatten_layer1 = torch.nn.Flatten()
   # The value 768 is pre-computed by tracing the sizes of the layers' outputs.
   dense_layer1 = torch.nn.Linear(in_features=768, out_features=15)
   relu_layer3 = torch.nn.ReLU()

   dense_layer2 = torch.nn.Linear(in_features=15, out_features=4)
   output_layer = torch.nn.Softmax(1)

   model = torch.nn.Sequential(input_layer,
                               relu_layer1,
                               max_pool1,
                               conv_layer2,
                               relu_layer2,
                               flatten_layer1,
                               dense_layer1,
                               relu_layer3,
                               dense_layer2,
                               output_layer)

.. _prepare-the-training-data-3:

Prepare the Training Data
~~~~~~~~~~~~~~~~~~~~~~~~~

The data used in this example is available as 2 files:

1. `dataset_inputs.npy <https://github.com/ahmedfgad/NumPyCNN/blob/master/dataset_inputs.npy>`__:
   Data inputs.
   https://github.com/ahmedfgad/NumPyCNN/blob/master/dataset_inputs.npy

2. `dataset_outputs.npy <https://github.com/ahmedfgad/NumPyCNN/blob/master/dataset_outputs.npy>`__:
   Class labels.
   https://github.com/ahmedfgad/NumPyCNN/blob/master/dataset_outputs.npy

The data consists of 4 classes of images. The image shape is
``(100, 100, 3)`` and there are 20 images per class for a total of 80
training samples. For more information about the dataset, check the
`Reading the
Data <https://pygad.readthedocs.io/en/latest/README_pygad_cnn_ReadTheDocs.html#reading-the-data>`__
section of the ``pygad.cnn`` module.

Simply download these 2 files and read them according to the next code.

.. code:: python

   import numpy

   data_inputs = numpy.load("dataset_inputs.npy")

   data_outputs = numpy.load("dataset_outputs.npy")

The next figure shows how the fitness value changes.

.. figure:: https://user-images.githubusercontent.com/16560492/103469887-c7c4c180-4d38-11eb-98a7-1c5e73e918d0.png
   :alt: 

Here are some statistics about the trained model. The model accuracy is
97.5% after the 200 generations. Note that just running the code again
may give different results.

.. code:: 

   Fitness value of the best solution = 1.3009520689219258
   Index of the best solution : 0
   Crossentropy :  0.7686678
   Accuracy :  0.975
