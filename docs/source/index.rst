.. PyGAD documentation master file, created by
   sphinx-quickstart on Sat May 16 15:14:25 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.





Welcome to PyGAD's documentation!
=================================

`PyGAD <https://pypi.org/project/pygad>`__ is an open-source Python 3
library for implementing the genetic algorithm and optimizing machine
learning algorithms.

`PyGAD <https://pypi.org/project/pygad>`__ supports different types of
crossover, mutation, and parent selection.
`PyGAD <https://pypi.org/project/pygad>`__ allows different types of
problems to be optimized using the genetic algorithm by customizing the
fitness function.

Besides building the genetic algorithm, it builds and optimizes machine
learning algorithms. Currently,
`PyGAD <https://pypi.org/project/pygad>`__ supports building and
training (using genetic algorithm) artificial neural networks for
classification problems.

The library is under active development and more features in the genetic
algorithm will be added like working with binary problems. This is in
addition to supporting more machine learning algorithms.

.. _header-n47:

Donation
========

You can donate to PyGAD via `Open
Collective <https://opencollective.com/pygad>`__:
`opencollective.com/pygad <https://opencollective.com/pygad>`__.

To donate using PayPal, use either this link:
`paypal.me/ahmedfgad <https://paypal.me/ahmedfgad>`__ or the e-mail
address ahmed.f.gad@gmail.com.

.. _header-n5:

Installation
============

To install `PyGAD <https://pypi.org/project/pygad>`__, simply use pip to
download and install the library from
`PyPI <https://pypi.org/project/pygad>`__ (Python Package Index). The
library lives a PyPI at this page https://pypi.org/project/pygad.

For Windows, issue the following command:

.. code:: python

   pip install pygad

For Linux and Mac, replace ``pip`` by use ``pip3`` because the library
only supports Python 3.

.. code:: python

   pip3 install pygad

PyGAD is developed in Python 3.7.3 and depends on NumPy for creating and
manipulating arrays and Matplotlib for creating figures. The exact NumPy
version used in developing PyGAD is 1.16.4. For Matplotlib, the version
is 3.1.0.

.. _header-n12:

Quick Start
===========

To get started with `PyGAD <https://pypi.org/project/pygad>`__, simply
import it.

.. code:: python

   import pygad

Using `PyGAD <https://pypi.org/project/pygad>`__, a wide range of
problems can be optimized. A quick and simple problem to be optimized
using the `PyGAD <https://pypi.org/project/pygad>`__ is finding the best
set of weights that satisfy the following function:

.. code:: 

   y = f(w1:w6) = w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + 6wx6
   where (x1,x2,x3,x4,x5,x6)=(4,-2,3.5,5,-11,-4.7) and y=44

The first step is to prepare the inputs and the outputs of this
equation.

.. code:: python

   function_inputs = [4,-2,3.5,5,-11,-4.7]
   desired_output = 44

A very important step is to implement the fitness function that will be
used for calculating the fitness value for each solution. Here is one.

.. code:: python

   def fitness_func(solution, solution_idx):
       output = numpy.sum(solution*function_inputs)
       fitness = 1.0 / numpy.abs(output - desired_output)
       return fitness

Next is to prepare the parameters of
`PyGAD <https://pypi.org/project/pygad>`__. Here is an example for a set
of parameters.

.. code:: python

   fitness_function = fitness_func

   num_generations = 50
   num_parents_mating = 4

   sol_per_pop = 8
   num_genes = len(function_inputs)

   init_range_low = -2
   init_range_high = 5

   parent_selection_type = "sss"
   keep_parents = 1

   crossover_type = "single_point"

   mutation_type = "random"
   mutation_percent_genes = 10

After the parameters are prepared, an instance of the **pygad.GA** class
is created.

.. code:: python

   ga_instance = pygad.GA(num_generations=num_generations,
                          num_parents_mating=num_parents_mating, 
                          fitness_func=fitness_function,
                          sol_per_pop=sol_per_pop, 
                          num_genes=num_genes,
                          init_range_low=init_range_low,
                          init_range_high=init_range_high,
                          parent_selection_type=parent_selection_type,
                          keep_parents=keep_parents,
                          crossover_type=crossover_type,
                          mutation_type=mutation_type,
                          mutation_percent_genes=mutation_percent_genes)

After creating the instance, the ``run()`` method is called to start the
optimization.

.. code:: python

   ga_instance.run()

After the ``run()`` method completes, information about the best
solution found by PyGAD can be accessed.

.. code:: python

   solution, solution_fitness, solution_idx = ga_instance.best_solution()
   print("Parameters of the best solution : {solution}".format(solution=solution))
   print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

   prediction = numpy.sum(numpy.array(function_inputs)*solution)
   print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

.. code:: 

   Parameters of the best solution : [3.92692328 -0.11554946 2.39873381 3.29579039 -0.74091476 1.05468517]
   Fitness value of the best solution = 157.37320042925006
   Predicted output based on the best solution : 44.00635432206546

There is more to do using PyGAD. Read its documentation to explore the
features of PyGAD.

.. _header-n31:

PyGAD's Modules
===============

`PyGAD <https://pypi.org/project/pygad>`__ has the following modules:

1. The main module has the same name as the library which is ``pygad``
   that builds the genetic algorithm.

2. The ``nn`` module builds artificial neural networks.

3. The ``gann`` module optimizes neural networks using the genetic
   algorithm.

4. The ``cnn`` module builds convolutional neural networks.

5. The ``gacnn`` module optimizes convolutional neural networks using
   the genetic algorithm.

The documentation discusses each of these modules.





.. _header-n4:

pygad Module
===============


.. toctree::
   :maxdepth: 4
   :caption: pygad Module TOC

   README_pygad_ReadTheDocs.rst




.. _header-n5:

pygad.nn Module
===============


.. toctree::
   :maxdepth: 4
   :caption: pygad.nn Module TOC

   README_pygad_nn_ReadTheDocs.rst





.. _header-n6:

pygad.gann Module
=================


.. toctree::
   :maxdepth: 4
   :caption: pygad.gann Module TOC

   README_pygad_gann_ReadTheDocs.rst









.. _header-n7:

pygad.cnn Module
=================


.. toctree::
   :maxdepth: 4
   :caption: pygad.cnn Module TOC

   README_pygad_cnn_ReadTheDocs.rst









.. _header-n8:

pygad.gacnn Module
=================


.. toctree::
   :maxdepth: 4
   :caption: pygad.gacnn Module TOC

   README_pygad_gacnn_ReadTheDocs.rst




.. _header-n9:

More Information
=================


.. toctree::
   :maxdepth: 4
   :caption: More Information

   Footer.rst





Indices and tables
==================

* :ref:`search`
