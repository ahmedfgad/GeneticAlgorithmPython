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
then the generated random value (e.g. 0.1) will replace the gene value.
The new gene value is **0.1**.

1. ``None`` value could be assigned to the ``mutation_type`` and
   ``crossover_type`` parameters of the pygad.GA class constructor. When
   ``None``, this means the step is bypassed and has no action.

.. _header-n62:

PyGAD 2.3.0
-----------

Release date: 1 June 2020

1. A new module named ``pygad.cnn`` is supported for building
   convolutional neural networks.

2. A new module named ``pygad.gacnn`` is supported for training
   convolutional neural networks using the genetic algorithm.

3. The ``pygad.plot_result()`` method has 3 optional parameters named
   ``title``, ``xlabel``, and ``ylabel`` to customize the plot title,
   x-axis label, and y-axis label, respectively.

4. The ``pygad.nn`` module supports the softmax activation function.

5. The name of the ``pygad.nn.predict_outputs()`` function is changed to
   ``pygad.nn.predict()``.

6. The name of the ``pygad.nn.train_network()`` function is changed to
   ``pygad.nn.train()``.

.. _header-n77:

PyGAD 2.4.0
-----------

Release date: 5 July 2020

1. A new parameter named ``delay_after_gen`` is added which accepts a
   non-negative number specifying the time in seconds to wait after a
   generation completes and before going to the next generation. It
   defaults to ``0.0`` which means no delay after the generation.

2. The passed function to the ``callback_generation`` parameter of the
   pygad.GA class constructor can terminate the execution of the genetic
   algorithm if it returns the string ``stop``. This causes the
   ``run()`` method to stop.

One important use case for that feature is to stop the genetic algorithm
when a condition is met before passing though all the generations. The
user may assigned a value of 100 to the ``num_generations`` parameter of
the pygad.GA class constructor. Assuming that at generation 50, for
example, a condition is met and the user wants to stop the execution
before waiting the remaining 50 generations. To do that, just make the
function passed to the ``callback_generation`` parameter to return the
string ``stop``.

Here is an example of a function to be passed to the
``callback_generation`` parameter which stops the execution if the
fitness value 70 is reached. The value 70 might be the best possible
fitness value. After being reached, then there is no need to pass
through more generations because no further improvement is possible.

.. code:: python

      def func_generation(ga_instance):
       if ga_instance.best_solution()[1] >= 70:
           return "stop"

.. _header-n87:

PyGAD 2.5.0
-----------

Release date: 19 July 2020

1. | 2 new optional parameters added to the constructor of the
     ``pygad.GA`` class which are ``crossover_probability`` and
     ``mutation_probability``. 
   | While applying the crossover operation, each parent has a random
     value generated between 0.0 and 1.0. If this random value is less
     than or equal to the value assigned to the
     ``crossover_probability`` parameter, then the parent is selected
     for the crossover operation.
   | For the mutation operation, a random value between 0.0 and 1.0 is
     generated for each gene in the solution. If this value is less than
     or equal to the value assigned to the ``mutation_probability``,
     then this gene is selected for mutation.

2. A new optional parameter named ``linewidth`` is added to the
   ``plot_result()`` method to specify the width of the curve in the
   plot. It defaults to 3.0.

3. Previously, the indices of the genes selected for mutation was
   randomly generated once for all solutions within the generation.
   Currently, the genes' indices are randomly generated for each
   solution in the population. If the population has 4 solutions, the
   indices are randomly generated 4 times inside the single generation,
   1 time for each solution.

4. Previously, the position of the point(s) for the single-point and
   two-points crossover was(were) randomly selected once for all
   solutions within the generation. Currently, the position(s) is(are)
   randomly selected for each solution in the population. If the
   population has 4 solutions, the position(s) is(are) randomly
   generated 4 times inside the single generation, 1 time for each
   solution.

5. A new optional parameter named ``gene_space`` as added to the
   ``pygad.GA`` class constructor. It is used to specify the possible
   values for each gene in case the user wants to restrict the gene
   values. It is useful if the gene space is restricted to a certain
   range or to discrete values.

Assuming that all genes have the same global space which include the
values 0.3, 5.2, -4, and 8, then those values can be assigned to the
``gene_space`` parameter as a list, tuple, or range. Here is a list
assigned to this parameter. By doing that, then the gene values are
restricted to those assigned to the ``gene_space`` parameter.

.. code:: python

   gene_space = [0.3, 5.2, -4, 8]

If some genes have different spaces, then ``gene_space`` should accept a
nested list or tuple. In this case, its elements could be:

1. List, tuple, or range: It holds the individual gene space.

2. Number (int/float): A single value to be assigned to the gene. This
   means this gene will have the same value across all generations.

3. ``None``: A gene with its space set to ``None`` is initialized
   randomly from the range specified by the 2 parameters
   ``init_range_low`` and ``init_range_high``. For mutation, its value
   is mutated based on a random value from the range specified by the 2
   parameters ``random_mutation_min_val`` and
   ``random_mutation_max_val``. If all elements in the ``gene_space``
   parameter are ``None``, the parameter will not have any effect.

Assuming that a chromosome has 2 genes and each gene has a different
value space. Then the ``gene_space`` could be assigned a nested
list/tuple where each element determines the space of a gene. According
to the next code, the space of the first gene is [0.4, -5] which has 2
values and the space for the second gene is [0.5, -3.2, 8.8, -9] which
has 4 values.

.. code:: python

   gene_space = [[0.4, -5], [0.5, -3.2, 8.2, -9]]

For a 2 gene chromosome, if the first gene space is restricted to the
discrete values from 0 to 4 and the second gene is restricted to the
values from 10 to 19, then it could be specified according to the next
code.

.. code:: python

   gene_space = [range(5), range(10, 20)]

If the user did not assign the initial population to the
``initial_population`` parameter, the initial population is created
randomly based on the ``gene_space`` parameter. Moreover, the mutation
is applied based on this parameter.

.. _header-n115:

PyGAD 2.6.0 
------------

Release Date: 6 August 2020

1. A bug fix in assigning the value to the ``initial_population``
   parameter.

2. A new parameter named ``gene_type`` is added to control the gene
   type. It can be either ``int`` or ``float``. It has an effect only
   when the parameter ``gene_space`` is ``None``.

3. 7 new parameters that accept callback functions: ``on_start``,
   ``on_fitness``, ``on_parents``, ``on_crossover``, ``on_mutation``,
   ``on_generation``, and ``on_stop``.

.. _header-n124:

PyGAD Projects at GitHub
========================

The PyGAD library is available at PyPI at this page
https://pypi.org/project/pygad. PyGAD is built out of a number of
open-source GitHub projects. A brief note about these projects is given
in the next subsections.

.. _header-n126:

`GeneticAlgorithmPython <https://github.com/ahmedfgad/GeneticAlgorithmPython>`__
--------------------------------------------------------------------------------

GitHub Link: https://github.com/ahmedfgad/GeneticAlgorithmPython

`GeneticAlgorithmPython <https://github.com/ahmedfgad/GeneticAlgorithmPython>`__
is the first project which is an open-source Python 3 project for
implementing the genetic algorithm based on NumPy.

.. _header-n129:

`NumPyANN <https://github.com/ahmedfgad/NumPyANN>`__
----------------------------------------------------

GitHub Link: https://github.com/ahmedfgad/NumPyANN

`NumPyANN <https://github.com/ahmedfgad/NumPyANN>`__ builds artificial
neural networks in **Python 3** using **NumPy** from scratch. The
purpose of this project is to only implement the **forward pass** of a
neural network without using a training algorithm. Currently, it only
supports classification and later regression will be also supported.
Moreover, only one class is supported per sample.

.. _header-n132:

`NeuralGenetic <https://github.com/ahmedfgad/NeuralGenetic>`__
--------------------------------------------------------------

GitHub Link: https://github.com/ahmedfgad/NeuralGenetic

`NeuralGenetic <https://github.com/ahmedfgad/NeuralGenetic>`__ trains
neural networks using the genetic algorithm based on the previous 2
projects
`GeneticAlgorithmPython <https://github.com/ahmedfgad/GeneticAlgorithmPython>`__
and `NumPyANN <https://github.com/ahmedfgad/NumPyANN>`__.

.. _header-n135:

`NumPyCNN <https://github.com/ahmedfgad/NumPyCNN>`__
----------------------------------------------------

GitHub Link: https://github.com/ahmedfgad/NumPyCNN

`NumPyCNN <https://github.com/ahmedfgad/NumPyCNN>`__ builds
convolutional neural networks using NumPy. The purpose of this project
is to only implement the **forward pass** of a convolutional neural
network without using a training algorithm.

.. _header-n138:

`CNNGenetic <https://github.com/ahmedfgad/CNNGenetic>`__
--------------------------------------------------------

GitHub Link: https://github.com/ahmedfgad/CNNGenetic

`CNNGenetic <https://github.com/ahmedfgad/CNNGenetic>`__ trains
convolutional neural networks using the genetic algorithm. It uses the
`GeneticAlgorithmPython <https://github.com/ahmedfgad/GeneticAlgorithmPython>`__
project for building the genetic algorithm.

.. _header-n141:

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

.. _header-n145:

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

.. _header-n149:

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

.. _header-n160:

For More Information
====================

There are different resources that can be used to get started with the
genetic algorithm and building it in Python.

.. _header-n162:

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

.. _header-n173:

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

.. _header-n183:

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

.. _header-n193:

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

.. _header-n203:

Tutorial: Building CNN in Python
--------------------------------

To start with coding the genetic algorithm, you can check the tutorial
titled `Building Convolutional Neural Network using NumPy from
Scratch <https://www.linkedin.com/pulse/building-convolutional-neural-network-using-numpy-from-ahmed-gad>`__
available at these links:

-  `LinkedIn <https://www.linkedin.com/pulse/building-convolutional-neural-network-using-numpy-from-ahmed-gad>`__

-  `Towards Data
   Science <https://towardsdatascience.com/building-convolutional-neural-network-using-numpy-from-scratch-b30aac50e50a>`__

-  `KDnuggets <https://www.kdnuggets.com/2018/04/building-convolutional-neural-network-numpy-scratch.html>`__

-  `Chinese Translation <http://m.aliyun.com/yunqi/articles/585741>`__

`This
tutorial <https://www.linkedin.com/pulse/building-convolutional-neural-network-using-numpy-from-ahmed-gad>`__)
is prepared based on a previous version of the project but it still a
good resource to start with coding CNNs.

|image4|

.. _header-n216:

Tutorial: Derivation of CNN from FCNN
-------------------------------------

Get started with the genetic algorithm by reading the tutorial titled
`Derivation of Convolutional Neural Network from Fully Connected Network
Step-By-Step <https://www.linkedin.com/pulse/derivation-convolutional-neural-network-from-fully-connected-gad>`__
which is available at these links:

-  `LinkedIn <https://www.linkedin.com/pulse/derivation-convolutional-neural-network-from-fully-connected-gad>`__

-  `Towards Data
   Science <https://towardsdatascience.com/derivation-of-convolutional-neural-network-from-fully-connected-network-step-by-step-b42ebafa5275>`__

-  `KDnuggets <https://www.kdnuggets.com/2018/04/derivation-convolutional-neural-network-fully-connected-step-by-step.html>`__

|image5|

.. _header-n226:

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

.. _header-n241:

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
.. |image4| image:: https://user-images.githubusercontent.com/16560492/82431022-6c3a1200-9a8e-11ea-8f1b-b055196d76e3.png
   :target: https://www.linkedin.com/pulse/building-convolutional-neural-network-using-numpy-from-ahmed-gad
.. |image5| image:: https://user-images.githubusercontent.com/16560492/82431369-db176b00-9a8e-11ea-99bd-e845192873fc.png
   :target: https://www.linkedin.com/pulse/derivation-convolutional-neural-network-from-fully-connected-gad
