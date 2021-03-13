.. _header-n0:

Release History
===============

.. figure:: https://user-images.githubusercontent.com/16560492/101267295-c74c0180-375f-11eb-9ad0-f8e37bd796ce.png
   :alt: 

.. _header-n3:

PyGAD 1.0.17
------------

Release Date: 15 April 2020

1. The **pygad.GA** class accepts a new argument named ``fitness_func``
   which accepts a function to be used for calculating the fitness
   values for the solutions. This allows the project to be customized to
   any problem by building the right fitness function.

.. _header-n8:

PyGAD 1.0.20 
------------

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

.. _header-n19:

PyGAD 2.0.0 
-----------

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

.. _header-n30:

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

.. _header-n45:

PyGAD 2.2.1
-----------

Release Date: 17 May 2020

1. Adding 2 extra modules (pygad.nn and pygad.gann) for building and
   training neural networks with the genetic algorithm.

.. _header-n50:

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

If ``mutation_type="random"`` and ``mutation_by_replacement=True``, then
the generated random value (e.g. 0.1) will replace the gene value. The
new gene value is **0.1**.

1. ``None`` value could be assigned to the ``mutation_type`` and
   ``crossover_type`` parameters of the pygad.GA class constructor. When
   ``None``, this means the step is bypassed and has no action.

.. _header-n63:

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

.. _header-n78:

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

.. _header-n88:

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
   range or to discrete values. Thanks to `Prof. Tamer A.
   Farrag <https://github.com/tfarrag2000>`__ for requesting this useful
   feature.

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

.. _header-n116:

PyGAD 2.6.0 
-----------

Release Date: 6 August 2020

1. A bug fix in assigning the value to the ``initial_population``
   parameter.

2. A new parameter named ``gene_type`` is added to control the gene
   type. It can be either ``int`` or ``float``. It has an effect only
   when the parameter ``gene_space`` is ``None``.

3. 7 new parameters that accept callback functions: ``on_start``,
   ``on_fitness``, ``on_parents``, ``on_crossover``, ``on_mutation``,
   ``on_generation``, and ``on_stop``.

.. _header-n125:

PyGAD 2.7.0
-----------

Release Date: 11 September 2020

1. The ``learning_rate`` parameter in the ``pygad.nn.train()`` function
   defaults to **0.01**.

2. Added support of building neural networks for regression using the
   new parameter named ``problem_type``. It is added as a parameter to
   both ``pygad.nn.train()`` and ``pygad.nn.predict()`` functions. The
   value of this parameter can be either **classification** or
   **regression** to define the problem type. It defaults to
   **classification**.

3. The activation function for a layer can be set to the string
   ``"None"`` to refer that there is no activation function at this
   layer. As a result, the supported values for the activation function
   are ``"sigmoid"``, ``"relu"``, ``"softmax"``, and ``"None"``.

To build a regression network using the ``pygad.nn`` module, just do the
following:

1. Set the ``problem_type`` parameter in the ``pygad.nn.train()`` and
   ``pygad.nn.predict()`` functions to the string ``"regression"``.

2. Set the activation function for the output layer to the string
   ``"None"``. This sets no limits on the range of the outputs as it
   will be from ``-infinity`` to ``+infinity``. If you are sure that all
   outputs will be nonnegative values, then use the ReLU function.

Check the documentation of the ``pygad.nn`` module for an example that
builds a neural network for regression. The regression example is also
available at `this GitHub
project <https://github.com/ahmedfgad/NumPyANN>`__:
https://github.com/ahmedfgad/NumPyANN

To build and train a regression network using the ``pygad.gann`` module,
do the following:

1. Set the ``problem_type`` parameter in the ``pygad.nn.train()`` and
   ``pygad.nn.predict()`` functions to the string ``"regression"``.

2. Set the ``output_activation`` parameter in the constructor of the
   ``pygad.gann.GANN`` class to ``"None"``.

Check the documentation of the ``pygad.gann`` module for an example that
builds and trains a neural network for regression. The regression
example is also available at `this GitHub
project <https://github.com/ahmedfgad/NeuralGenetic>`__:
https://github.com/ahmedfgad/NeuralGenetic

To build a classification network, either ignore the ``problem_type``
parameter or set it to ``"classification"`` (default value). In this
case, the activation function of the last layer can be set to any type
(e.g. softmax).

.. _header-n149:

PyGAD 2.7.1
-----------

Release Date: 11 September 2020

1. A bug fix when the ``problem_type`` argument is set to
   ``regression``.

.. _header-n154:

PyGAD 2.7.2
-----------

Release Date: 14 September 2020

1. Bug fix to support building and training regression neural networks
   with multiple outputs.

.. _header-n159:

PyGAD 2.8.0
-----------

Release Date: 20 September 2020

1. Support of a new module named ``kerasga`` so that the Keras models
   can be trained by the genetic algorithm using PyGAD.

.. _header-n164:

PyGAD 2.8.1
-----------

Release Date: 3 October 2020

1. Bug fix in applying the crossover operation when the
   ``crossover_probability`` parameter is used. Thanks to `Eng. Hamada
   Kassem, Research and Teaching Assistant, Construction Engineering and
   Management, Faculty of Engineering, Alexandria University,
   Egypt <https://www.linkedin.com/in/hamadakassem>`__.

.. _header-n169:

PyGAD 2.9.0 
-----------

Release Date: 06 December 2020

1. The fitness values of the initial population are considered in the
   ``best_solutions_fitness`` attribute.

2. An optional parameter named ``save_best_solutions`` is added. It
   defaults to ``False``. When it is ``True``, then the best solution
   after each generation is saved into an attribute named
   ``best_solutions``. If ``False``, then no solutions are saved and the
   ``best_solutions`` attribute will be empty.

3. Scattered crossover is supported. To use it, assign the
   ``crossover_type`` parameter the value ``"scattered"``.

4. NumPy arrays are now supported by the ``gene_space`` parameter.

5. The following parameters (``gene_type``, ``crossover_probability``,
   ``mutation_probability``, ``delay_after_gen``) can be assigned to a
   numeric value of any of these data types: ``int``, ``float``,
   ``numpy.int``, ``numpy.int8``, ``numpy.int16``, ``numpy.int32``,
   ``numpy.int64``, ``numpy.float``, ``numpy.float16``,
   ``numpy.float32``, or ``numpy.float64``.

.. _header-n182:

PyGAD 2.10.0
------------

Release Date: 03 January 2021

1.  Support of a new module ``pygad.torchga`` to train PyTorch models
    using PyGAD. Check `its
    documentation <https://pygad.readthedocs.io/en/latest/README_pygad_torchga_ReadTheDocs.html>`__.

2.  Support of adaptive mutation where the mutation rate is determined
    by the fitness value of each solution. Read the `Adaptive
    Mutation <https://pygad.readthedocs.io/en/latest/README_pygad_torchga_ReadTheDocs.html#adaptive-mutation>`__
    section for more details. Also, read this paper: `Libelli, S.
    Marsili, and P. Alba. "Adaptive mutation in genetic algorithms."
    Soft computing 4.2 (2000):
    76-80. <https://www.researchgate.net/publication/225642916_Adaptive_mutation_in_genetic_algorithms>`__

3.  Before the ``run()`` method completes or exits, the fitness value of
    the best solution in the current population is appended to the
    ``best_solution_fitness`` list attribute. Note that the fitness
    value of the best solution in the initial population is already
    saved at the beginning of the list. So, the fitness value of the
    best solution is saved before the genetic algorithm starts and after
    it ends.

4.  When the parameter ``parent_selection_type`` is set to ``sss``
    (steady-state selection), then a warning message is printed if the
    value of the ``keep_parents`` parameter is set to 0.

5.  More validations to the user input parameters.

6.  The default value of the ``mutation_percent_genes`` is set to the
    string ``"default"`` rather than the integer 10. This change helps
    to know whether the user explicitly passed a value to the
    ``mutation_percent_genes`` parameter or it is left to its default
    one. The ``"default"`` value is later translated into the integer
    10.

7.  The ``mutation_percent_genes`` parameter is no longer accepting the
    value 0. It must be ``>0`` and ``<=100``.

8.  The built-in ``warnings`` module is used to show warning messages
    rather than just using the ``print()`` function.

9.  A new ``bool`` parameter called ``suppress_warnings`` is added to
    the constructor of the ``pygad.GA`` class. It allows the user to
    control whether the warning messages are printed or not. It defaults
    to ``False`` which means the messages are printed.

10. A helper method called ``adaptive_mutation_population_fitness()`` is
    created to calculate the average fitness value used in adaptive
    mutation to filter the solutions.

11. The ``best_solution()`` method accepts a new optional parameter
    called ``pop_fitness``. It accepts a list of the fitness values of
    the solutions in the population. If ``None``, then the
    ``cal_pop_fitness()`` method is called to calculate the fitness
    values of the population.

.. _header-n207:

PyGAD 2.10.1
------------

Release Date: 10 January 2021

1. In the ``gene_space`` parameter, any ``None`` value (regardless of
   its index or axis), is replaced by a randomly generated number based
   on the 3 parameters ``init_range_low``, ``init_range_high``, and
   ``gene_type``. So, the ``None`` value in ``[..., None, ...]`` or
   ``[..., [..., None, ...], ...]`` are replaced with random values.
   This gives more freedom in building the space of values for the
   genes.

2. All the numbers passed to the ``gene_space`` parameter are casted to
   the type specified in the ``gene_type`` parameter.

3. The ``numpy.uint`` data type is supported for the parameters that
   accept integer values.

4. In the ``pygad.kerasga`` module, the ``model_weights_as_vector()``
   function uses the ``trainable`` attribute of the model's layers to
   only return the trainable weights in the network. So, only the
   trainable layers with their ``trainable`` attribute set to ``True``
   (``trainable=True``), which is the default value, have their weights
   evolved. All non-trainable layers with the ``trainable`` attribute
   set to ``False`` (``trainable=False``) will not be evolved. Thanks to
   `Prof. Tamer A. Farrag <https://github.com/tfarrag2000>`__ for
   pointing about that at
   `GitHub <https://github.com/ahmedfgad/KerasGA/issues/1>`__.

.. _header-n218:

PyGAD 2.10.2
------------

Release Date: 15 January 2021

1. A bug fix when ``save_best_solutions=True``. Refer to this issue for
   more information:
   https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/25

.. _header-n223:

PyGAD 2.11.0
------------

Release Date: 16 February 2021

1. In the ``gene_space`` argument, the user can use a dictionary to
   specify the lower and upper limits of the gene. This dictionary must
   have only 2 items with keys ``low`` and ``high`` to specify the low
   and high limits of the gene, respectively. This way, PyGAD takes care
   of not exceeding the value limits of the gene. For a problem with
   only 2 genes, then using
   ``gene_space=[{'low': 1, 'high': 5}, {'low': 0.2, 'high': 0.81}]``
   means the accepted values in the first gene start from 1 (inclusive)
   to 5 (exclusive) while the second one has values between 0.2
   (inclusive) and 0.85 (exclusive). For more information, please check
   the `Limit the Gene Value
   Range <https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#limit-the-gene-value-range>`__
   section of the documentation.

2. The ``plot_result()`` method returns the figure so that the user can
   save it.

3. Bug fixes in copying elements from the gene space.

4. For a gene with a set of discrete values (more than 1 value) in the
   ``gene_space`` parameter like ``[0, 1]``, it was possible that the
   gene value may not change after mutation. That is if the current
   value is 0, then the randomly selected value could also be 0. Now, it
   is verified that the new value is changed. So, if the current value
   is 0, then the new value after mutation will not be 0 but 1.

.. _header-n234:

PyGAD 2.12.0
------------

Release Date: 20 February 2021

1. 4 new instance attributes are added to hold temporary results after
   each generation: ``last_generation_fitness`` holds the fitness values
   of the solutions in the last generation, ``last_generation_parents``
   holds the parents selected from the last generation,
   ``last_generation_offspring_crossover`` holds the offspring generated
   after applying the crossover in the last generation, and
   ``last_generation_offspring_mutation`` holds the offspring generated
   after applying the mutation in the last generation. You can access
   these attributes inside the ``on_generation()`` method for example.

2. A bug fixed when the ``initial_population`` parameter is used. The
   bug occurred due to a mismatch between the data type of the array
   assigned to ``initial_population`` and the gene type in the
   ``gene_type`` attribute. Assuming that the array assigned to the
   ``initial_population`` parameter is
   ``((1, 1), (3, 3), (5, 5), (7, 7))`` which has type ``int``. When
   ``gene_type`` is set to ``float``, then the genes will not be float
   but casted to ``int`` because the defined array has ``int`` type. The
   bug is fixed by forcing the array assigned to ``initial_population``
   to have the data type in the ``gene_type`` attribute. Check the
   `issue at
   GitHub <https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/27>`__:
   https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/27

Thanks to Andrei Rozanski [PhD Bioinformatics Specialist, Department of
Tissue Dynamics and Regeneration, Max Planck Institute for Biophysical
Chemistry, Germany] for opening my eye to the first change.

Thanks to `Marios
Giouvanakis <https://www.researchgate.net/profile/Marios-Giouvanakis>`__,
a PhD candidate in Electrical & Computer Engineer, `Aristotle University
of Thessaloniki (Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης),
Greece <https://www.auth.gr/en>`__, for emailing me about the second
issue.

.. _header-n243:

PyGAD 2.13.0 
------------

Release Date: 12 March 2021

1. A new ``bool`` parameter called ``allow_duplicate_genes`` is
   supported. If ``True``, which is the default, then a
   solution/chromosome may have duplicate gene values. If ``False``,
   then each gene will have a unique value in its solution. Check the
   `Prevent Duplicates in Gene
   Values <https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#prevent-duplicates-in-gene-values>`__
   section for more details.

2. The ``last_generation_fitness`` is updated at the end of each
   generation not at the beginning. This keeps the fitness values of the
   most up-to-date population assigned to the
   ``last_generation_fitness`` parameter.

.. _header-n250:

PyGAD Projects at GitHub
========================

The PyGAD library is available at PyPI at this page
https://pypi.org/project/pygad. PyGAD is built out of a number of
open-source GitHub projects. A brief note about these projects is given
in the next subsections.

.. _header-n252:

`GeneticAlgorithmPython <https://github.com/ahmedfgad/GeneticAlgorithmPython>`__
--------------------------------------------------------------------------------

GitHub Link: https://github.com/ahmedfgad/GeneticAlgorithmPython

`GeneticAlgorithmPython <https://github.com/ahmedfgad/GeneticAlgorithmPython>`__
is the first project which is an open-source Python 3 project for
implementing the genetic algorithm based on NumPy.

.. _header-n255:

`NumPyANN <https://github.com/ahmedfgad/NumPyANN>`__
----------------------------------------------------

GitHub Link: https://github.com/ahmedfgad/NumPyANN

`NumPyANN <https://github.com/ahmedfgad/NumPyANN>`__ builds artificial
neural networks in **Python 3** using **NumPy** from scratch. The
purpose of this project is to only implement the **forward pass** of a
neural network without using a training algorithm. Currently, it only
supports classification and later regression will be also supported.
Moreover, only one class is supported per sample.

.. _header-n258:

`NeuralGenetic <https://github.com/ahmedfgad/NeuralGenetic>`__
--------------------------------------------------------------

GitHub Link: https://github.com/ahmedfgad/NeuralGenetic

`NeuralGenetic <https://github.com/ahmedfgad/NeuralGenetic>`__ trains
neural networks using the genetic algorithm based on the previous 2
projects
`GeneticAlgorithmPython <https://github.com/ahmedfgad/GeneticAlgorithmPython>`__
and `NumPyANN <https://github.com/ahmedfgad/NumPyANN>`__.

.. _header-n261:

`NumPyCNN <https://github.com/ahmedfgad/NumPyCNN>`__
----------------------------------------------------

GitHub Link: https://github.com/ahmedfgad/NumPyCNN

`NumPyCNN <https://github.com/ahmedfgad/NumPyCNN>`__ builds
convolutional neural networks using NumPy. The purpose of this project
is to only implement the **forward pass** of a convolutional neural
network without using a training algorithm.

.. _header-n264:

`CNNGenetic <https://github.com/ahmedfgad/CNNGenetic>`__
--------------------------------------------------------

GitHub Link: https://github.com/ahmedfgad/CNNGenetic

`CNNGenetic <https://github.com/ahmedfgad/CNNGenetic>`__ trains
convolutional neural networks using the genetic algorithm. It uses the
`GeneticAlgorithmPython <https://github.com/ahmedfgad/GeneticAlgorithmPython>`__
project for building the genetic algorithm.

.. _header-n267:

`KerasGA <https://github.com/ahmedfgad/KerasGA>`__
--------------------------------------------------

GitHub Link: https://github.com/ahmedfgad/KerasGA

`KerasGA <https://github.com/ahmedfgad/KerasGA>`__ trains
`Keras <https://keras.io>`__ models using the genetic algorithm. It uses
the
`GeneticAlgorithmPython <https://github.com/ahmedfgad/GeneticAlgorithmPython>`__
project for building the genetic algorithm.

.. _header-n270:

`TorchGA <https://github.com/ahmedfgad/TorchGA>`__
--------------------------------------------------

GitHub Link: https://github.com/ahmedfgad/TorchGA

`TorchGA <https://github.com/ahmedfgad/TorchGA>`__ trains
`PyTorch <https://pytorch.org>`__ models using the genetic algorithm. It
uses the
`GeneticAlgorithmPython <https://github.com/ahmedfgad/GeneticAlgorithmPython>`__
project for building the genetic algorithm.

`pygad.torchga <https://github.com/ahmedfgad/TorchGA>`__:
https://github.com/ahmedfgad/TorchGA

.. _header-n274:

Stackoverflow Questions about PyGAD
===================================

.. _header-n275:

`How can I save a matplotlib plot that is the output of a function in jupyter? <https://stackoverflow.com/questions/66055330/how-can-i-save-a-matplotlib-plot-that-is-the-output-of-a-function-in-jupyter>`__
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

.. _header-n276:

`How do I query the best solution of a pyGAD GA instance? <https://stackoverflow.com/questions/65757722/how-do-i-query-the-best-solution-of-a-pygad-ga-instance>`__
-------------------------------------------------------------------------------------------------------------------------------------------------------------------

.. _header-n277:

`How to solve TSP problem using pyGAD package? <https://stackoverflow.com/questions/66298595/how-to-solve-tsp-problem-using-pygad-package>`__
---------------------------------------------------------------------------------------------------------------------------------------------

.. _header-n278:

`Multi-Input Multi-Output in Genetic algorithm (python) <https://stackoverflow.com/questions/64943711/multi-input-multi-output-in-genetic-algorithm-python>`__
--------------------------------------------------------------------------------------------------------------------------------------------------------------

.. _header-n279:

Submitting Issues
=================

If there is an issue using PyGAD, then use any of your preferred option
to discuss that issue.

One way is `submitting an
issue <https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/new>`__
into this GitHub project
(`github.com/ahmedfgad/GeneticAlgorithmPython <https://github.com/ahmedfgad/GeneticAlgorithmPython>`__)
in case something is not working properly or to ask for questions.

If this is not a proper option for you, then check the `Contact
Us <https://pygad.readthedocs.io/en/latest/Footer.html#contact-us>`__
section for more contact details.

.. _header-n283:

Ask for Feature
===============

PyGAD is actively developed with the goal of building a dynamic library
for suporting a wide-range of problems to be optimized using the genetic
algorithm.

To ask for a new feature, either `submit an
issue <https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/new>`__
into this GitHub project
(`github.com/ahmedfgad/GeneticAlgorithmPython <https://github.com/ahmedfgad/GeneticAlgorithmPython>`__)
or send an e-mail to ahmed.f.gad@gmail.com.

Also check the `Contact
Us <https://pygad.readthedocs.io/en/latest/Footer.html#contact-us>`__
section for more contact details.

.. _header-n287:

Projects Built using PyGAD
==========================

If you created a project that uses PyGAD, then we can support you by
mentioning this project here in PyGAD's documentation.

To do that, please send a message at ahmed.f.gad@gmail.com or check the
`Contact
Us <https://pygad.readthedocs.io/en/latest/Footer.html#contact-us>`__
section for more contact details.

Within your message, please send the following details:

-  Project title

-  Brief description

-  Preferably, a link that directs the readers to your project

.. _header-n298:

Tutorials about PyGAD
=====================

.. _header-n299:

`Working with Different Genetic Algorithm Representations in Python <https://blog.paperspace.com/working-with-different-genetic-algorithm-representations-python>`__
--------------------------------------------------------------------------------------------------------------------------------------------------------------------

Depending on the nature of the problem being optimized, the genetic
algorithm (GA) supports two different gene representations: binary, and
decimal. The binary GA has only two values for its genes, which are 0
and 1. This is easier to manage as its gene values are limited compared
to the decimal GA, for which we can use different formats like float or
integer, and limited or unlimited ranges.

This tutorial discusses how the
`PyGAD <https://pygad.readthedocs.io/>`__ library supports the two GA
representations, binary and decimal.

.. _header-n302:

`5 Genetic Algorithm Applications Using PyGAD <https://blog.paperspace.com/genetic-algorithm-applications-using-pygad>`__
-------------------------------------------------------------------------------------------------------------------------

This tutorial introduces PyGAD, an open-source Python library for
implementing the genetic algorithm and training machine learning
algorithms. PyGAD supports 19 parameters for customizing the genetic
algorithm for various applications.

Within this tutorial we'll discuss 5 different applications of the
genetic algorithm and build them using PyGAD.

.. _header-n305:

`Train Neural Networks Using a Genetic Algorithm in Python with PyGAD <https://heartbeat.fritz.ai/train-neural-networks-using-a-genetic-algorithm-in-python-with-pygad-862905048429?gi=ba58ee6b4bbd>`__
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

The genetic algorithm (GA) is a biologically-inspired optimization
algorithm. It has in recent years gained importance, as it’s simple
while also solving complex problems like travel route optimization,
training machine learning algorithms, working with single and
multi-objective problems, game playing, and more.

Deep neural networks are inspired by the idea of how the biological
brain works. It’s a universal function approximator, which is capable of
simulating any function, and is now used to solve the most complex
problems in machine learning. What’s more, they’re able to work with all
types of data (images, audio, video, and text).

Both genetic algorithms (GAs) and neural networks (NNs) are similar, as
both are biologically-inspired techniques. This similarity motivates us
to create a hybrid of both to see whether a GA can train NNs with high
accuracy.

This tutorial uses `PyGAD <https://pygad.readthedocs.io/>`__, a Python
library that supports building and training NNs using a GA.
`PyGAD <https://pygad.readthedocs.io/>`__ offers both classification and
regression NNs.

.. _header-n310:

`How To Train Keras Models Using the Genetic Algorithm with PyGAD <https://blog.paperspace.com/train-keras-models-using-genetic-algorithm-with-pygad>`__
--------------------------------------------------------------------------------------------------------------------------------------------------------

PyGAD is an open-source Python library for building the genetic
algorithm and training machine learning algorithms. It offers a wide
range of parameters to customize the genetic algorithm to work with
different types of problems.

PyGAD has its own modules that support building and training neural
networks (NNs) and convolutional neural networks (CNNs). Despite these
modules working well, they are implemented in Python without any
additional optimization measures. This leads to comparatively high
computational times for even simple problems.

The latest PyGAD version, 2.8.0 (released on 20 September 2020),
supports a new module to train Keras models. Even though Keras is built
in Python, it's fast. The reason is that Keras uses TensorFlow as a
backend, and TensorFlow is highly optimized.

This tutorial discusses how to train Keras models using PyGAD. The
discussion includes building Keras models using either the Sequential
Model or the Functional API, building an initial population of Keras
model parameters, creating an appropriate fitness function, and more.

|image1|

.. _header-n316:

`Train PyTorch Models Using Genetic Algorithm with PyGAD <https://neptune.ai/blog/train-pytorch-models-using-genetic-algorithm-with-pygad>`__
---------------------------------------------------------------------------------------------------------------------------------------------

`PyGAD <https://pygad.readthedocs.io/>`__ is a genetic algorithm Python
3 library for solving optimization problems. One of these problems is
training machine learning algorithms.

PyGAD has a module called
`pygad.kerasga <https://github.com/ahmedfgad/KerasGA>`__. It trains
Keras models using the genetic algorithm. On January 3rd, 2021, a new
release of `PyGAD 2.10.0 <https://pygad.readthedocs.io/>`__ brought a
new module called
`pygad.torchga <https://github.com/ahmedfgad/TorchGA>`__ to train
PyTorch models. It’s very easy to use, but there are a few tricky steps.

So, in this tutorial, we’ll explore how to use PyGAD to train PyTorch
models.

|image2|

.. _header-n321:

PyGAD in Other Languages
========================

.. _header-n322:

French
------

`Cómo los algoritmos genéticos pueden competir con el descenso de
gradiente y el
backprop <https://www.hebergementwebs.com/nouvelles/comment-les-algorithmes-genetiques-peuvent-rivaliser-avec-la-descente-de-gradient-et-le-backprop>`__

Bien que la manière standard d'entraîner les réseaux de neurones soit la
descente de gradient et la rétropropagation, il y a d'autres joueurs
dans le jeu. L'un d'eux est les algorithmes évolutionnaires, tels que
les algorithmes génétiques.

Utiliser un algorithme génétique pour former un réseau de neurones
simple pour résoudre le OpenAI CartPole Jeu. Dans cet article, nous
allons former un simple réseau de neurones pour résoudre le OpenAI
CartPole . J'utiliserai PyTorch et PyGAD .

|image3|

.. _header-n327:

Spanish
-------

`Cómo los algoritmos genéticos pueden competir con el descenso de
gradiente y el
backprop <https://www.hebergementwebs.com/noticias/como-los-algoritmos-geneticos-pueden-competir-con-el-descenso-de-gradiente-y-el-backprop>`__

Aunque la forma estandar de entrenar redes neuronales es el descenso de
gradiente y la retropropagacion, hay otros jugadores en el juego, uno de
ellos son los algoritmos evolutivos, como los algoritmos geneticos.

Usa un algoritmo genetico para entrenar una red neuronal simple para
resolver el Juego OpenAI CartPole. En este articulo, entrenaremos una
red neuronal simple para resolver el OpenAI CartPole . Usare PyTorch y
PyGAD .

|image4|

.. _header-n332:

Korean
------

.. _header-n333:

`[PyGAD] Python 에서 Genetic Algorithm 을 사용해보기 <https://data-newbie.tistory.com/m/685>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

|image5|

파이썬에서 genetic algorithm을 사용하는 패키지들을 다 사용해보진
않았지만, 확장성이 있어보이고, 시도할 일이 있어서 살펴봤다.

이 패키지에서 가장 인상 깊었던 것은 neural network에서 hyper parameter
탐색을 gradient descent 방식이 아닌 GA로도 할 수 있다는 것이다.

개인적으로 이 부분이 어느정도 초기치를 잘 잡아줄 수 있는 역할로도 쓸 수
있고, Loss가 gradient descent 하기 어려운 구조에서 대안으로 쓸 수 있을
것으로도 생각된다.

일단 큰 흐름은 다음과 같이 된다.

사실 완전히 흐름이나 각 parameter에 대한 이해는 부족한 상황

.. _header-n340:

Turkish
-------

.. _header-n341:

`PyGAD ile Genetik Algoritmayı Kullanarak Keras Modelleri Nasıl Eğitilir <https://erencan34.medium.com/pygad-ile-genetik-algoritmay%C4%B1-kullanarak-keras-modelleri-nas%C4%B1l-e%C4%9Fitilir-cf92639a478c>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is a translation of an original English tutorial published at
Paperspace: `How To Train Keras Models Using the Genetic Algorithm with
PyGAD <https://blog.paperspace.com/train-keras-models-using-genetic-algorithm-with-pygad>`__

PyGAD, genetik algoritma oluşturmak ve makine öğrenimi algoritmalarını
eğitmek için kullanılan açık kaynaklı bir Python kitaplığıdır. Genetik
algoritmayı farklı problem türleri ile çalışacak şekilde özelleştirmek
için çok çeşitli parametreler sunar.

PyGAD, sinir ağları (NN’ler) ve evrişimli sinir ağları (CNN’ler)
oluşturmayı ve eğitmeyi destekleyen kendi modüllerine sahiptir. Bu
modüllerin iyi çalışmasına rağmen, herhangi bir ek optimizasyon önlemi
olmaksızın Python’da uygulanırlar. Bu, basit problemler için bile
nispeten yüksek hesaplama sürelerine yol açar.

En son PyGAD sürümü 2.8.0 (20 Eylül 2020'de piyasaya sürüldü), Keras
modellerini eğitmek için yeni bir modülü destekliyor. Keras Python’da
oluşturulmuş olsa da hızlıdır. Bunun nedeni, Keras’ın arka uç olarak
TensorFlow kullanması ve TensorFlow’un oldukça optimize edilmiş
olmasıdır.

Bu öğreticide, PyGAD kullanılarak Keras modellerinin nasıl eğitileceği
anlatılmaktadır. Tartışma, Sıralı Modeli veya İşlevsel API’yi kullanarak
Keras modellerini oluşturmayı, Keras model parametrelerinin ilk
popülasyonunu oluşturmayı, uygun bir uygunluk işlevi oluşturmayı ve daha
fazlasını içerir.

|image6|

.. _header-n348:

Hungarian
---------

.. _header-n349:

`Tensorflow alapozó 10. Neurális hálózatok tenyésztése genetikus algoritmussal PyGAD és OpenAI Gym használatával <https://thebojda.medium.com/tensorflow-alapoz%C3%B3-10-24f7767d4a2c>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hogy kontextusba helyezzem a genetikus algoritmusokat, ismételjük kicsit
át, hogy hogyan működik a gradient descent és a backpropagation, ami a
neurális hálók tanításának általános módszere. Az erről írt cikkemet itt
tudjátok elolvasni.

A hálózatok tenyésztéséhez a
`PyGAD <https://pygad.readthedocs.io/en/latest/>`__ nevű
programkönyvtárat használjuk, így mindenek előtt ezt kell telepítenünk,
valamint a Tensorflow-t és a Gym-et, amit Colabban már eleve telepítve
kapunk.

Maga a PyGAD egy teljesen általános genetikus algoritmusok futtatására
képes rendszer. Ennek a kiterjesztése a KerasGA, ami az általános motor
Tensorflow (Keras) neurális hálókon történő futtatását segíti. A 47.
sorban létrehozott KerasGA objektum ennek a kiterjesztésnek a része és
arra szolgál, hogy a paraméterként átadott modellből a második
paraméterben megadott számosságú populációt hozzon létre. Mivel a
hálózatunk 386 állítható paraméterrel rendelkezik, ezért a DNS-ünk itt
386 elemből fog állni. A populáció mérete 10 egyed, így a kezdő
populációnk egy 10x386 elemű mátrix lesz. Ezt adjuk át az 51. sorban az
initial_population paraméterben.

|image7|

.. _header-n354:

Russian
-------

.. _header-n355:

`PyGAD: библиотека для имплементации генетического алгоритма <https://neurohive.io/ru/frameworki/pygad-biblioteka-dlya-implementacii-geneticheskogo-algoritma>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyGAD — это библиотека для имплементации генетического алгоритма. Кроме
того, библиотека предоставляет доступ к оптимизированным реализациям
алгоритмов машинного обучения. PyGAD разрабатывали на Python 3.

Библиотека PyGAD поддерживает разные типы скрещивания, мутации и
селекции родителя. PyGAD позволяет оптимизировать проблемы с помощью
генетического алгоритма через кастомизацию целевой функции.

Кроме генетического алгоритма, библиотека содержит оптимизированные
имплементации алгоритмов машинного обучения. На текущий момент PyGAD
поддерживает создание и обучение нейросетей для задач классификации.

Библиотека находится в стадии активной разработки. Создатели планируют
добавление функционала для решения бинарных задач и имплементации новых
алгоритмов.

PyGAD разрабатывали на Python 3.7.3. Зависимости включают в себя NumPy
для создания и манипуляции массивами и Matplotlib для визуализации. Один
из изкейсов использования инструмента — оптимизация весов, которые
удовлетворяют заданной функции.

|image8|

.. _header-n362:

For More Information
====================

There are different resources that can be used to get started with the
genetic algorithm and building it in Python.

.. _header-n364:

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

|image9|

.. _header-n375:

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

|image10|

.. _header-n385:

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

|image11|

.. _header-n395:

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

|image12|

.. _header-n405:

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

|image13|

.. _header-n418:

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

|image14|

.. _header-n428:

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

.. _header-n443:

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

.. figure:: https://user-images.githubusercontent.com/16560492/101267295-c74c0180-375f-11eb-9ad0-f8e37bd796ce.png
   :alt: 

Thank you for using
`PyGAD <https://github.com/ahmedfgad/GeneticAlgorithmPython>`__ :)

.. |image1| image:: https://user-images.githubusercontent.com/16560492/111009628-2b372500-8362-11eb-90cf-01b47d831624.png
   :target: https://blog.paperspace.com/train-keras-models-using-genetic-algorithm-with-pygad
.. |image2| image:: https://user-images.githubusercontent.com/16560492/111009678-5457b580-8362-11eb-899a-39e2f96984df.png
   :target: https://neptune.ai/blog/train-pytorch-models-using-genetic-algorithm-with-pygad
.. |image3| image:: https://user-images.githubusercontent.com/16560492/111009275-3178d180-8361-11eb-9e86-7fb1519acde7.png
   :target: https://www.hebergementwebs.com/nouvelles/comment-les-algorithmes-genetiques-peuvent-rivaliser-avec-la-descente-de-gradient-et-le-backprop
.. |image4| image:: https://user-images.githubusercontent.com/16560492/111009257-232ab580-8361-11eb-99a5-7226efbc3065.png
   :target: https://www.hebergementwebs.com/noticias/como-los-algoritmos-geneticos-pueden-competir-con-el-descenso-de-gradiente-y-el-backprop
.. |image5| image:: https://user-images.githubusercontent.com/16560492/108586306-85bd0280-731b-11eb-874c-7ac4ce1326cd.jpg
   :target: https://data-newbie.tistory.com/m/685
.. |image6| image:: https://user-images.githubusercontent.com/16560492/108586601-85be0200-731d-11eb-98a4-161c75a1f099.jpg
   :target: https://erencan34.medium.com/pygad-ile-genetik-algoritmay%C4%B1-kullanarak-keras-modelleri-nas%C4%B1l-e%C4%9Fitilir-cf92639a478c
.. |image7| image:: https://user-images.githubusercontent.com/16560492/101267295-c74c0180-375f-11eb-9ad0-f8e37bd796ce.png
   :target: https://thebojda.medium.com/tensorflow-alapoz%C3%B3-10-24f7767d4a2c
.. |image8| image:: https://user-images.githubusercontent.com/16560492/101267295-c74c0180-375f-11eb-9ad0-f8e37bd796ce.png
   :target: https://neurohive.io/ru/frameworki/pygad-biblioteka-dlya-implementacii-geneticheskogo-algoritma
.. |image9| image:: https://user-images.githubusercontent.com/16560492/78830052-a3c19300-79e7-11ea-8b9b-4b343ea4049c.png
   :target: https://www.linkedin.com/pulse/genetic-algorithm-implementation-python-ahmed-gad
.. |image10| image:: https://user-images.githubusercontent.com/16560492/82078259-26252d00-96e1-11ea-9a02-52a99e1054b9.jpg
   :target: https://www.linkedin.com/pulse/introduction-optimization-genetic-algorithm-ahmed-gad
.. |image11| image:: https://user-images.githubusercontent.com/16560492/82078281-30472b80-96e1-11ea-8017-6a1f4383d602.jpg
   :target: https://www.linkedin.com/pulse/artificial-neural-network-implementation-using-numpy-fruits360-gad
.. |image12| image:: https://user-images.githubusercontent.com/16560492/82078300-376e3980-96e1-11ea-821c-aa6b8ceb44d4.jpg
   :target: https://www.linkedin.com/pulse/artificial-neural-networks-optimization-using-genetic-ahmed-gad
.. |image13| image:: https://user-images.githubusercontent.com/16560492/82431022-6c3a1200-9a8e-11ea-8f1b-b055196d76e3.png
   :target: https://www.linkedin.com/pulse/building-convolutional-neural-network-using-numpy-from-ahmed-gad
.. |image14| image:: https://user-images.githubusercontent.com/16560492/82431369-db176b00-9a8e-11ea-99bd-e845192873fc.png
   :target: https://www.linkedin.com/pulse/derivation-convolutional-neural-network-from-fully-connected-gad
