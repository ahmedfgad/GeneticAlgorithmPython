.. _header-n0:

Release History
===============

.. _header-n2:

PyGAD 1.0.17
------------

Release Date: 15 April 2020

1. The **pygad.GA** class accepts a new argument named ``fitness_func``
   which accepts a function to be used for calculating the fitness
   values for the solutions. This allows the project to be customized to
   any problem by building the right fitness function.

.. _header-n7:

PyGAD 1.0.20 
-------------

Release Date: 4 May 2020

1. The **pygad.GA** attributes are moved from the class scope to the
   instance scope.

2. Raising an exception for incorrect values of the passed parameters.

3. Two new parameters are added to the **pygad.GA** class constructor
   (``init_range_low`` and ``init_range_high``) allowing the user to
   customize the range from which the genes values in the initial
   population are selected.

4. The code object ``__code__`` of the passed fitness function is
   checked to ensure it has the right number of parameters.

.. _header-n18:

PyGAD 2.0.0 
------------

Release Date: 13 May 2020

1. The fitness function accepts a new argument named ``sol_idx``
   representing the index of the solution within the population.

2. A new parameter to the **pygad.GA** class constructor named
   ``initial_population`` is supported to allow the user to use a custom
   initial population to be used by the genetic algorithm. If not None,
   then the passed population will be used. If ``None``, then the
   genetic algorithm will create the initial population using the
   ``sol_per_pop`` and ``num_genes`` parameters.

3. The parameters ``sol_per_pop`` and ``num_genes`` are optional and set
   to ``None`` by default.

4. A new parameter named ``callback_generation`` is introduced in the
   **pygad.GA** class constructor. It accepts a function with a single
   parameter representing the **pygad.GA** class instance. This function
   is called after each generation. This helps the user to do
   post-processing or debugging operations after each generation.

.. _header-n29:

PyGAD 2.1.0
-----------

Release Date: 14 May 2020

1. The ``best_solution()`` method in the **pygad.GA** class returns a
   new output representing the index of the best solution within the
   population. Now, it returns a total of 3 outputs and their order is:
   best solution, best solution fitness, and best solution index. Here
   is an example:

.. code:: python

   solution, solution_fitness, solution_idx = ga_instance.best_solution()
   print("Parameters of the best solution :", solution)
   print("Fitness value of the best solution :", solution_fitness, "\n")
   print("Index of the best solution :", solution_idx, "\n")

1. | A new attribute named ``best_solution_generation`` is added to the
     instances of the **pygad.GA** class. it holds the generation number
     at which the best solution is reached. It is only assigned the
     generation number after the ``run()`` method completes. Otherwise,
     its value is -1.
   | Example:

.. code:: python

   print("Best solution reached after {best_solution_generation} generations.".format(best_solution_generation=ga_instance.best_solution_generation))

1. The ``best_solution_fitness`` attribute is renamed to
   ``best_solutions_fitness`` (plural solution).

2. Mutation is applied independently for the genes.

.. _header-n44:

PyGAD 2.2.1
-----------

Release Date: 17 May 2020

1. Adding 2 extra modules (pygad.nn and pygad.gann) for building and
   training neural networks with the genetic algorithm.

.. _header-n49:

PyGAD 2.2.2
-----------

Release Date: 18 May 2020

1. The initial value of the ``generations_completed`` attribute of
   instances from the pygad.GA class is ``0`` rather than ``None``.

2. An optional bool parameter named ``mutation_by_replacement`` is added
   to the constructor of the pygad.GA class. It works only when the
   selected type of mutation is random (``mutation_type="random"``). In
   this case, setting ``mutation_by_replacement=True`` means replace the
   gene by the randomly generated value. If ``False``, then it has no
   effect and random mutation works by adding the random value to the
   gene. This parameter should be used when the gene falls within a
   fixed range and its value must not go out of this range. Here are
   some examples:

   Assume there is a gene with the value 0.5.

   If ``mutation_type="random"`` and ``mutation_by_replacement=False``,
   then the generated random value (e.g. 0.1) will be added to the gene
   value. The new gene value is **0.5+0.1=0.6**.

   If ``mutation_type="random"`` and ``mutation_by_replacement=True``,
   then the generated random value (e.g. 0.1) will replace the gene
   value. The new gene value is **0.1**.

3. ``None`` value could be assigned to the ``mutation_type`` and
   ``crossover_type`` parameters of the pygad.GA class constructor. When
   ``None``, this means the step is bypassed and has no action.

.. _header-n155:

PyGAD Projects at GitHub
========================

The PyGAD library is available at PyPI at this page
https://pypi.org/project/pygad. PyGAD is built out of a number of
open-source GitHub projects. A brief note about these projects is given
in the next subsections.

.. _header-n51:

`GeneticAlgorithmPython <https://github.com/ahmedfgad/GeneticAlgorithmPython>`__
--------------------------------------------------------------------------------

GitHub Link: https://github.com/ahmedfgad/GeneticAlgorithmPython

`GeneticAlgorithmPython <https://github.com/ahmedfgad/GeneticAlgorithmPython>`__
is the first project which is an open-source Python 3 project for
implementing the genetic algorithm based on NumPy.

.. _header-n54:

`NumPyANN <https://github.com/ahmedfgad/NumPyANN>`__
----------------------------------------------------

GitHub Link: https://github.com/ahmedfgad/NumPyANN

`NumPyANN <https://github.com/ahmedfgad/NumPyANN>`__ builds artificial
neural networks in **Python 3** using **NumPy** from scratch. The
purpose of this project is to only implement the **forward pass** of a
neural network without using a training algorithm. Currently, it only
supports classification and later regression will be also supported.
Moreover, only one class is supported per sample.

.. _header-n57:

`NeuralGenetic <https://github.com/ahmedfgad/NeuralGenetic>`__
--------------------------------------------------------------

GitHub Link: https://github.com/ahmedfgad/NeuralGenetic

`NeuralGenetic <https://github.com/ahmedfgad/NeuralGenetic>`__ trains
neural networks using the genetic algorithm based on the previous 2
projects
`GeneticAlgorithmPython <https://github.com/ahmedfgad/GeneticAlgorithmPython>`__
and `NumPyANN <https://github.com/ahmedfgad/NumPyANN>`__.

.. _header-n60:

Submitting Issues
=================

If there is an issue using PyGAD, then use any of your preferred option
to discuss that issue.

One way is `submitting an
issue <https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/new>`__
into this GitHub project
(https://github.com/ahmedfgad/GeneticAlgorithmPython) in case something
is not working properly or to ask for questions.

If this is not a proper option for you, then check the **Contact Us**
section for more contact details.

.. _header-n64:

Ask for Feature
===============

PyGAD is actively developed with the goal of building a dynamic library
for suporting a wide-range of problems to be optimized using the genetic
algorithm.

To ask for a new feature, either `submit an
issue <https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/new>`__
into this GitHub project
(https://github.com/ahmedfgad/GeneticAlgorithmPython) or send an e-mail
to ahmed.f.gad@gmail.com.

Also check the **Contact Us** section for more contact details.

.. _header-n68:

Projects Built using PyGAD
==========================

If you created a project that uses PyGAD, then we can support you by
mentioning this project here in PyGAD's documentation.

To do that, please send a message at ahmed.f.gad@gmail.com or check the
**Contact Us** section for more contact details.

Within your message, please send the following details:

-  Project title

-  Brief description

-  Preferably, a link that directs the readers to your project

.. _header-n79:

For More Information
====================

There are different resources that can be used to get started with the
genetic algorithm and building it in Python.

.. _header-n81:

Tutorial: Implementing Genetic Algorithm in Python
--------------------------------------------------

To start with coding the genetic algorithm, you can check the tutorial
titled `Genetic Algorithm Implementation in
Python <https://www.linkedin.com/pulse/genetic-algorithm-implementation-python-ahmed-gad>`__
available at these links:

-  `LinkedIn <https://www.linkedin.com/pulse/genetic-algorithm-implementation-python-ahmed-gad>`__

-  `Towards Data
   Science <https://towardsdatascience.com/genetic-algorithm-implementation-in-python-5ab67bb124a6>`__

-  `KDnuggets <https://www.kdnuggets.com/2018/07/genetic-algorithm-implementation-python.html>`__

`This
tutorial <https://www.linkedin.com/pulse/genetic-algorithm-implementation-python-ahmed-gad>`__
is prepared based on a previous version of the project but it still a
good resource to start with coding the genetic algorithm.

|image0|

.. _header-n92:

Tutorial: Introduction to Genetic Algorithm
-------------------------------------------

Get started with the genetic algorithm by reading the tutorial titled
`Introduction to Optimization with Genetic
Algorithm <https://www.linkedin.com/pulse/introduction-optimization-genetic-algorithm-ahmed-gad>`__
which is available at these links:

-  `LinkedIn <https://www.linkedin.com/pulse/introduction-optimization-genetic-algorithm-ahmed-gad>`__

-  `Towards Data
   Science <https://www.kdnuggets.com/2018/03/introduction-optimization-with-genetic-algorithm.html>`__

-  `KDnuggets <https://towardsdatascience.com/introduction-to-optimization-with-genetic-algorithm-2f5001d9964b>`__

|image1|

.. _header-n102:

Tutorial: Build Neural Networks in Python
-----------------------------------------

Read about building neural networks in Python through the tutorial
titled `Artificial Neural Network Implementation using NumPy and
Classification of the Fruits360 Image
Dataset <https://www.linkedin.com/pulse/artificial-neural-network-implementation-using-numpy-fruits360-gad>`__
available at these links:

-  `LinkedIn <https://www.linkedin.com/pulse/artificial-neural-network-implementation-using-numpy-fruits360-gad>`__

-  `Towards Data
   Science <https://towardsdatascience.com/artificial-neural-network-implementation-using-numpy-and-classification-of-the-fruits360-image-3c56affa4491>`__

-  `KDnuggets <https://www.kdnuggets.com/2019/02/artificial-neural-network-implementation-using-numpy-and-image-classification.html>`__

|image2|

.. _header-n112:

Tutorial: Optimize Neural Networks with Genetic Algorithm
---------------------------------------------------------

Read about training neural networks using the genetic algorithm through
the tutorial titled `Artificial Neural Networks Optimization using
Genetic Algorithm with
Python <https://www.linkedin.com/pulse/artificial-neural-networks-optimization-using-genetic-ahmed-gad>`__
available at these links:

-  `LinkedIn <https://www.linkedin.com/pulse/artificial-neural-networks-optimization-using-genetic-ahmed-gad>`__

-  `Towards Data
   Science <https://towardsdatascience.com/artificial-neural-networks-optimization-using-genetic-algorithm-with-python-1fe8ed17733e>`__

-  `KDnuggets <https://www.kdnuggets.com/2019/03/artificial-neural-networks-optimization-genetic-algorithm-python.html>`__

|image3|

.. _header-n122:

Book: Practical Computer Vision Applications Using Deep Learning with CNNs
--------------------------------------------------------------------------

You can also check my book cited as `Ahmed Fawzy Gad 'Practical Computer
Vision Applications Using Deep Learning with CNNs'. Dec. 2018, Apress,
978-1-4842-4167-7 <https://www.amazon.com/Practical-Computer-Vision-Applications-Learning/dp/1484241665>`__
which discusses neural networks, convolutional neural networks, deep
learning, genetic algorithm, and more.

Find the book at these links:

-  `Amazon <https://www.amazon.com/Practical-Computer-Vision-Applications-Learning/dp/1484241665>`__

-  `Springer <https://link.springer.com/book/10.1007/978-1-4842-4167-7>`__

-  `Apress <https://www.apress.com/gp/book/9781484241660>`__

-  `O'Reilly <https://www.oreilly.com/library/view/practical-computer-vision/9781484241677>`__

-  `Google Books <https://books.google.com.eg/books?id=xLd9DwAAQBAJ>`__

.. figure:: https://user-images.githubusercontent.com/16560492/78830077-ae7c2800-79e7-11ea-980b-53b6bd879eeb.jpg
   :alt: 

.. _header-n137:

Contact Us
==========

-  E-mail: ahmed.f.gad@gmail.com

-  `LinkedIn <https://www.linkedin.com/in/ahmedfgad>`__

-  `Amazon Author Page <https://amazon.com/author/ahmedgad>`__

-  `Heartbeat <https://heartbeat.fritz.ai/@ahmedfgad>`__

-  `Paperspace <https://blog.paperspace.com/author/ahmed>`__

-  `KDnuggets <https://kdnuggets.com/author/ahmed-gad>`__

-  `TowardsDataScience <https://towardsdatascience.com/@ahmedfgad>`__

-  `GitHub <https://github.com/ahmedfgad>`__

.. |image0| image:: https://user-images.githubusercontent.com/16560492/78830052-a3c19300-79e7-11ea-8b9b-4b343ea4049c.png
   :target: https://www.linkedin.com/pulse/genetic-algorithm-implementation-python-ahmed-gad
.. |image1| image:: https://user-images.githubusercontent.com/16560492/82078259-26252d00-96e1-11ea-9a02-52a99e1054b9.jpg
   :target: https://www.linkedin.com/pulse/introduction-optimization-genetic-algorithm-ahmed-gad
.. |image2| image:: https://user-images.githubusercontent.com/16560492/82078281-30472b80-96e1-11ea-8017-6a1f4383d602.jpg
   :target: https://www.linkedin.com/pulse/artificial-neural-network-implementation-using-numpy-fruits360-gad
.. |image3| image:: https://user-images.githubusercontent.com/16560492/82078300-376e3980-96e1-11ea-821c-aa6b8ceb44d4.jpg
   :target: https://www.linkedin.com/pulse/artificial-neural-networks-optimization-using-genetic-ahmed-gad
