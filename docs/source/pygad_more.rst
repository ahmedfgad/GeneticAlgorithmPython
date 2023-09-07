More About PyGAD
================

Multi-Objective Optimization
============================

In `PyGAD
3.2.0 <https://pygad.readthedocs.io/en/latest/releases.html#pygad-3-2-0>`__,
the library supports multi-objective optimization using the
non-dominated sorting genetic algorithm II (NSGA-II). The code is
exactly similar to the regular code used for single-objective
optimization except for 1 difference. It is the return value of the
fitness function.

In single-objective optimization, the fitness function returns a single
numeric value. In this example, the variable ``fitness`` is expected to
be a numeric value.

.. code:: python

   def fitness_func(ga_instance, solution, solution_idx):
       ...
       return fitness

But in multi-objective optimization, the fitness function returns any of
these data types:

1. ``list``

2. ``tuple``

3. ``numpy.ndarray``

.. code:: python

   def fitness_func(ga_instance, solution, solution_idx):
       ...
       return [fitness1, fitness2, ..., fitnessN]

Whenever the fitness function returns an iterable of these data types,
then the problem is considered multi-objective. This holds even if there
is a single element in the returned iterable.

Other than the fitness function, everything else could be the same in
both single and multi-objective problems.

But it is recommended to use one of these 2 parent selection operators
to solve multi-objective problems:

1. ``nsga2``: This selects the parents based on non-dominated sorting
   and crowding distance.

2. ``tournament_nsga2``: This selects the parents using tournament
   selection which uses non-dominated sorting and crowding distance to
   rank the solutions.

This is a multi-objective optimization example that optimizes these 2
linear functions:

1. ``y1 = f(w1:w6) = w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + 6wx6``

2. ``y2 = f(w1:w6) = w1x7 + w2x8 + w3x9 + w4x10 + w5x11 + 6wx12``

Where:

1. ``(x1,x2,x3,x4,x5,x6)=(4,-2,3.5,5,-11,-4.7)`` and ``y=50``

2. ``(x7,x8,x9,x10,x11,x12)=(-2,0.7,-9,1.4,3,5)`` and ``y=30``

The 2 functions use the same parameters (weights) ``w1`` to ``w6``.

The goal is to use PyGAD to find the optimal values for such weights
that satisfy the 2 functions ``y1`` and ``y2``.

.. code:: python

   import pygad
   import numpy

   """
   Given these 2 functions:
       y1 = f(w1:w6) = w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + 6wx6
       y2 = f(w1:w6) = w1x7 + w2x8 + w3x9 + w4x10 + w5x11 + 6wx12
       where (x1,x2,x3,x4,x5,x6)=(4,-2,3.5,5,-11,-4.7) and y=50
       and   (x7,x8,x9,x10,x11,x12)=(-2,0.7,-9,1.4,3,5) and y=30
   What are the best values for the 6 weights (w1 to w6)? We are going to use the genetic algorithm to optimize these 2 functions.
   This is a multi-objective optimization problem.

   PyGAD considers the problem as multi-objective if the fitness function returns:
       1) List.
       2) Or tuple.
       3) Or numpy.ndarray.
   """

   function_inputs1 = [4,-2,3.5,5,-11,-4.7] # Function 1 inputs.
   function_inputs2 = [-2,0.7,-9,1.4,3,5] # Function 2 inputs.
   desired_output1 = 50 # Function 1 output.
   desired_output2 = 30 # Function 2 output.

   def fitness_func(ga_instance, solution, solution_idx):
       output1 = numpy.sum(solution*function_inputs1)
       output2 = numpy.sum(solution*function_inputs2)
       fitness1 = 1.0 / (numpy.abs(output1 - desired_output1) + 0.000001)
       fitness2 = 1.0 / (numpy.abs(output2 - desired_output2) + 0.000001)
       return [fitness1, fitness2]

   num_generations = 100
   num_parents_mating = 10

   sol_per_pop = 20
   num_genes = len(function_inputs1)

   ga_instance = pygad.GA(num_generations=num_generations,
                          num_parents_mating=num_parents_mating,
                          sol_per_pop=sol_per_pop,
                          num_genes=num_genes,
                          fitness_func=fitness_func,
                          parent_selection_type='nsga2')

   ga_instance.run()

   ga_instance.plot_fitness(label=['Obj 1', 'Obj 2'])

   solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
   print(f"Parameters of the best solution : {solution}")
   print(f"Fitness value of the best solution = {solution_fitness}")

   prediction = numpy.sum(numpy.array(function_inputs1)*solution)
   print(f"Predicted output 1 based on the best solution : {prediction}")
   prediction = numpy.sum(numpy.array(function_inputs2)*solution)
   print(f"Predicted output 2 based on the best solution : {prediction}")

This is the result of the print statements. The predicted outputs are
close to the desired outputs.

.. code:: 

   Parameters of the best solution : [ 0.79676439 -2.98823386 -4.12677662  5.70539445 -2.02797016 -1.07243922]
   Fitness value of the best solution = [  1.68090829 349.8591915 ]
   Predicted output 1 based on the best solution : 50.59491545442283
   Predicted output 2 based on the best solution : 29.99714270722312

This is the figure created by the ``plot_fitness()`` method. The fitness
of the first objective has the green color. The blue color is used for
the second objective fitness.

.. image:: https://github.com/ahmedfgad/GeneticAlgorithmPython/assets/16560492/7896f8d8-01c5-4ff9-8d15-52191c309b63
   :alt: 

.. _limit-the-gene-value-range-using-the-genespace-parameter:

Limit the Gene Value Range using the ``gene_space`` Parameter
=============================================================

In `PyGAD
2.11.0 <https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-11-0>`__,
the ``gene_space`` parameter supported a new feature to allow
customizing the range of accepted values for each gene. Let's take a
quick review of the ``gene_space`` parameter to build over it.

The ``gene_space`` parameter allows the user to feed the space of values
of each gene. This way the accepted values for each gene is retracted to
the user-defined values. Assume there is a problem that has 3 genes
where each gene has different set of values as follows:

1. Gene 1: ``[0.4, 12, -5, 21.2]``

2. Gene 2: ``[-2, 0.3]``

3. Gene 3: ``[1.2, 63.2, 7.4]``

Then, the ``gene_space`` for this problem is as given below. Note that
the order is very important.

.. code:: python

   gene_space = [[0.4, 12, -5, 21.2],
                 [-2, 0.3],
                 [1.2, 63.2, 7.4]]

In case all genes share the same set of values, then simply feed a
single list to the ``gene_space`` parameter as follows. In this case,
all genes can only take values from this list of 6 values.

.. code:: python

   gene_space = [33, 7, 0.5, 95. 6.3, 0.74]

The previous example restricts the gene values to just a set of fixed
number of discrete values. In case you want to use a range of discrete
values to the gene, then you can use the ``range()`` function. For
example, ``range(1, 7)`` means the set of allowed values for the gene
are ``1, 2, 3, 4, 5, and 6``. You can also use the ``numpy.arange()`` or
``numpy.linspace()`` functions for the same purpose.

The previous discussion only works with a range of discrete values not
continuous values. In `PyGAD
2.11.0 <https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-11-0>`__,
the ``gene_space`` parameter can be assigned a dictionary that allows
the gene to have values from a continuous range.

Assuming you want to restrict the gene within this half-open range [1 to
5) where 1 is included and 5 is not. Then simply create a dictionary
with 2 items where the keys of the 2 items are:

1. ``'low'``: The minimum value in the range which is 1 in the example.

2. ``'high'``: The maximum value in the range which is 5 in the example.

The dictionary will look like that:

.. code:: python

   {'low': 1,
    'high': 5}

It is not acceptable to add more than 2 items in the dictionary or use
other keys than ``'low'`` and ``'high'``.

For a 3-gene problem, the next code creates a dictionary for each gene
to restrict its values in a continuous range. For the first gene, it can
take any floating-point value from the range that starts from 1
(inclusive) and ends at 5 (exclusive).

.. code:: python

   gene_space = [{'low': 1, 'high': 5}, {'low': 0.3, 'high': 1.4}, {'low': -0.2, 'high': 4.5}]

.. _more-about-the-genespace-parameter:

More about the ``gene_space`` Parameter
=======================================

The ``gene_space`` parameter customizes the space of values of each
gene.

Assuming that all genes have the same global space which include the
values 0.3, 5.2, -4, and 8, then those values can be assigned to the
``gene_space`` parameter as a list, tuple, or range. Here is a list
assigned to this parameter. By doing that, then the gene values are
restricted to those assigned to the ``gene_space`` parameter.

.. code:: python

   gene_space = [0.3, 5.2, -4, 8]

If some genes have different spaces, then ``gene_space`` should accept a
nested list or tuple. In this case, the elements could be:

1. Number (of ``int``, ``float``, or ``NumPy`` data types): A single
   value to be assigned to the gene. This means this gene will have the
   same value across all generations.

2. ``list``, ``tuple``, ``numpy.ndarray``, or any range like ``range``,
   ``numpy.arange()``, or ``numpy.linspace``: It holds the space for
   each individual gene. But this space is usually discrete. That is
   there is a set of finite values to select from.

3. ``dict``: To sample a value for a gene from a continuous range. The
   dictionary must have 2 mandatory keys which are ``"low"`` and
   ``"high"`` in addition to an optional key which is ``"step"``. A
   random value is returned between the values assigned to the items
   with ``"low"`` and ``"high"`` keys. If the ``"step"`` exists, then
   this works as the previous options (i.e. discrete set of values).

4. ``None``: A gene with its space set to ``None`` is initialized
   randomly from the range specified by the 2 parameters
   ``init_range_low`` and ``init_range_high``. For mutation, its value
   is mutated based on a random value from the range specified by the 2
   parameters ``random_mutation_min_val`` and
   ``random_mutation_max_val``. If all elements in the ``gene_space``
   parameter are ``None``, the parameter will not have any effect.

Assuming that a chromosome has 2 genes and each gene has a different
value space. Then the ``gene_space`` could be assigned a nested
list/tuple where each element determines the space of a gene.

According to the next code, the space of the first gene is ``[0.4, -5]``
which has 2 values and the space for the second gene is
``[0.5, -3.2, 8.8, -9]`` which has 4 values.

.. code:: python

   gene_space = [[0.4, -5], [0.5, -3.2, 8.2, -9]]

For a 2 gene chromosome, if the first gene space is restricted to the
discrete values from 0 to 4 and the second gene is restricted to the
values from 10 to 19, then it could be specified according to the next
code.

.. code:: python

   gene_space = [range(5), range(10, 20)]

The ``gene_space`` can also be assigned to a single range, as given
below, where the values of all genes are sampled from the same range.

.. code:: python

   gene_space = numpy.arange(15)

The ``gene_space`` can be assigned a dictionary to sample a value from a
continuous range.

.. code:: python

   gene_space = {"low": 4, "high": 30}

A step also can be assigned to the dictionary. This works as if a range
is used.

.. code:: python

   gene_space = {"low": 4, "high": 30, "step": 2.5}

..

   Setting a ``dict`` like ``{"low": 0, "high": 10}`` in the
   ``gene_space`` means that random values from the continuous range [0,
   10) are sampled. Note that ``0`` is included but ``10`` is not
   included while sampling. Thus, the maximum value that could be
   returned is less than ``10`` like ``9.9999``. But if the user decided
   to round the genes using, for example, ``[float, 2]``, then this
   value will become 10. So, the user should be careful to the inputs.

If a ``None`` is assigned to only a single gene, then its value will be
randomly generated initially using the ``init_range_low`` and
``init_range_high`` parameters in the ``pygad.GA`` class's constructor.
During mutation, the value are sampled from the range defined by the 2
parameters ``random_mutation_min_val`` and ``random_mutation_max_val``.
This is an example where the second gene is given a ``None`` value.

.. code:: python

   gene_space = [range(5), None, numpy.linspace(10, 20, 300)]

If the user did not assign the initial population to the
``initial_population`` parameter, the initial population is created
randomly based on the ``gene_space`` parameter. Moreover, the mutation
is applied based on this parameter.

.. _how-mutation-works-with-the-genespace-parameter:

How Mutation Works with the ``gene_space`` Parameter?
-----------------------------------------------------

If a gene has its static space defined in the ``gene_space`` parameter,
then mutation works by replacing the gene value by a value randomly
selected from the gene space. This happens for both ``int`` and
``float`` data types.

For example, the following ``gene_space`` has the static space
``[1, 2, 3]`` defined for the first gene. So, this gene can only have a
value out of these 3 values.

.. code:: python

   Gene space: [[1, 2, 3],
                None]
   Solution: [1, 5]

For a solution like ``[1, 5]``, then mutation happens for the first gene
by simply replacing its current value by a randomly selected value
(other than its current value if possible). So, the value 1 will be
replaced by either 2 or 3.

For the second gene, its space is set to ``None``. So, traditional
mutation happens for this gene by:

1. Generating a random value from the range defined by the
   ``random_mutation_min_val`` and ``random_mutation_max_val``
   parameters.

2. Adding this random value to the current gene's value.

If its current value is 5 and the random value is ``-0.5``, then the new
value is 4.5. If the gene type is integer, then the value will be
rounded.

Stop at Any Generation
======================

In `PyGAD
2.4.0 <https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-4-0>`__,
it is possible to stop the genetic algorithm after any generation. All
you need to do it to return the string ``"stop"`` in the callback
function ``on_generation``. When this callback function is implemented
and assigned to the ``on_generation`` parameter in the constructor of
the ``pygad.GA`` class, then the algorithm immediately stops after
completing its current generation. Let's discuss an example.

Assume that the user wants to stop algorithm either after the 100
generations or if a condition is met. The user may assign a value of 100
to the ``num_generations`` parameter of the ``pygad.GA`` class
constructor.

The condition that stops the algorithm is written in a callback function
like the one in the next code. If the fitness value of the best solution
exceeds 70, then the string ``"stop"`` is returned.

.. code:: python

   def func_generation(ga_instance):
       if ga_instance.best_solution()[1] >= 70:
           return "stop"

Stop Criteria
=============

In `PyGAD
2.15.0 <https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-15-0>`__,
a new parameter named ``stop_criteria`` is added to the constructor of
the ``pygad.GA`` class. It helps to stop the evolution based on some
criteria. It can be assigned to one or more criterion.

Each criterion is passed as ``str`` that consists of 2 parts:

1. Stop word.

2. Number.

It takes this form:

.. code:: python

   "word_num"

The current 2 supported words are ``reach`` and ``saturate``.

The ``reach`` word stops the ``run()`` method if the fitness value is
equal to or greater than a given fitness value. An example for ``reach``
is ``"reach_40"`` which stops the evolution if the fitness is >= 40.

``saturate`` stops the evolution if the fitness saturates for a given
number of consecutive generations. An example for ``saturate`` is
``"saturate_7"`` which means stop the ``run()`` method if the fitness
does not change for 7 consecutive generations.

Here is an example that stops the evolution if either the fitness value
reached ``127.4`` or if the fitness saturates for ``15`` generations.

.. code:: python

   import pygad
   import numpy

   equation_inputs = [4, -2, 3.5, 8, 9, 4]
   desired_output = 44

   def fitness_func(ga_instance, solution, solution_idx):
       output = numpy.sum(solution * equation_inputs)

       fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)

       return fitness

   ga_instance = pygad.GA(num_generations=200,
                          sol_per_pop=10,
                          num_parents_mating=4,
                          num_genes=len(equation_inputs),
                          fitness_func=fitness_func,
                          stop_criteria=["reach_127.4", "saturate_15"])

   ga_instance.run()
   print(f"Number of generations passed is {ga_instance.generations_completed}")

Elitism Selection
=================

In `PyGAD
2.18.0 <https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-18-0>`__,
a new parameter called ``keep_elitism`` is supported. It accepts an
integer to define the number of elitism (i.e. best solutions) to keep in
the next generation. This parameter defaults to ``1`` which means only
the best solution is kept in the next generation.

In the next example, the ``keep_elitism`` parameter in the constructor
of the ``pygad.GA`` class is set to 2. Thus, the best 2 solutions in
each generation are kept in the next generation.

.. code:: python

   import numpy
   import pygad

   function_inputs = [4,-2,3.5,5,-11,-4.7]
   desired_output = 44

   def fitness_func(ga_instance, solution, solution_idx):
       output = numpy.sum(solution*function_inputs)
       fitness = 1.0 / numpy.abs(output - desired_output)
       return fitness

   ga_instance = pygad.GA(num_generations=2,
                          num_parents_mating=3,
                          fitness_func=fitness_func,
                          num_genes=6,
                          sol_per_pop=5,
                          keep_elitism=2)

   ga_instance.run()

The value passed to the ``keep_elitism`` parameter must satisfy 2
conditions:

1. It must be ``>= 0``.

2. It must be ``<= sol_per_pop``. That is its value cannot exceed the
   number of solutions in the current population.

In the previous example, if the ``keep_elitism`` parameter is set equal
to the value passed to the ``sol_per_pop`` parameter, which is 5, then
there will be no evolution at all as in the next figure. This is because
all the 5 solutions are used as elitism in the next generation and no
offspring will be created.

.. code:: python

   ...

   ga_instance = pygad.GA(...,
                          sol_per_pop=5,
                          keep_elitism=5)

   ga_instance.run()

.. image:: https://user-images.githubusercontent.com/16560492/189273225-67ffad41-97ab-45e1-9324-429705e17b20.png
   :alt: 

Note that if the ``keep_elitism`` parameter is effective (i.e. is
assigned a positive integer, not zero), then the ``keep_parents``
parameter will have no effect. Because the default value of the
``keep_elitism`` parameter is 1, then the ``keep_parents`` parameter has
no effect by default. The ``keep_parents`` parameter is only effective
when ``keep_elitism=0``.

Random Seed
===========

In `PyGAD
2.18.0 <https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-18-0>`__,
a new parameter called ``random_seed`` is supported. Its value is used
as a seed for the random function generators.

PyGAD uses random functions in these 2 libraries:

1. NumPy

2. random

The ``random_seed`` parameter defaults to ``None`` which means no seed
is used. As a result, different random numbers are generated for each
run of PyGAD.

If this parameter is assigned a proper seed, then the results will be
reproducible. In the next example, the integer 2 is used as a random
seed.

.. code:: python

   import numpy
   import pygad

   function_inputs = [4,-2,3.5,5,-11,-4.7]
   desired_output = 44

   def fitness_func(ga_instance, solution, solution_idx):
       output = numpy.sum(solution*function_inputs)
       fitness = 1.0 / numpy.abs(output - desired_output)
       return fitness

   ga_instance = pygad.GA(num_generations=2,
                          num_parents_mating=3,
                          fitness_func=fitness_func,
                          sol_per_pop=5,
                          num_genes=6,
                          random_seed=2)

   ga_instance.run()
   best_solution, best_solution_fitness, best_match_idx = ga_instance.best_solution()
   print(best_solution)
   print(best_solution_fitness)

This is the best solution found and its fitness value.

.. code:: 

   [ 2.77249188 -4.06570662  0.04196872 -3.47770796 -0.57502138 -3.22775267]
   0.04872203136549972

After running the code again, it will find the same result.

.. code:: 

   [ 2.77249188 -4.06570662  0.04196872 -3.47770796 -0.57502138 -3.22775267]
   0.04872203136549972

Continue without Loosing Progress
=================================

In `PyGAD
2.18.0 <https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-18-0>`__,
and thanks for `Felix Bernhard <https://github.com/FeBe95>`__ for
opening `this GitHub
issue <https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/123#issuecomment-1203035106>`__,
the values of these 4 instance attributes are no longer reset after each
call to the ``run()`` method.

1. ``self.best_solutions``

2. ``self.best_solutions_fitness``

3. ``self.solutions``

4. ``self.solutions_fitness``

This helps the user to continue where the last run stopped without
loosing the values of these 4 attributes.

Now, the user can save the model by calling the ``save()`` method.

.. code:: python

   import pygad

   def fitness_func(ga_instance, solution, solution_idx):
       ...
       return fitness

   ga_instance = pygad.GA(...)

   ga_instance.run()

   ga_instance.plot_fitness()

   ga_instance.save("pygad_GA")

Then the saved model is loaded by calling the ``load()`` function. After
calling the ``run()`` method over the loaded instance, then the data
from the previous 4 attributes are not reset but extended with the new
data.

.. code:: python

   import pygad

   def fitness_func(ga_instance, solution, solution_idx):
       ...
       return fitness

   loaded_ga_instance = pygad.load("pygad_GA")

   loaded_ga_instance.run()

   loaded_ga_instance.plot_fitness()

The plot created by the ``plot_fitness()`` method will show the data
collected from both the runs.

Note that the 2 attributes (``self.best_solutions`` and
``self.best_solutions_fitness``) only work if the
``save_best_solutions`` parameter is set to ``True``. Also, the 2
attributes (``self.solutions`` and ``self.solutions_fitness``) only work
if the ``save_solutions`` parameter is ``True``.

Prevent Duplicates in Gene Values
=================================

In `PyGAD
2.13.0 <https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-13-0>`__,
a new bool parameter called ``allow_duplicate_genes`` is supported to
control whether duplicates are supported in the chromosome or not. In
other words, whether 2 or more genes might have the same exact value.

If ``allow_duplicate_genes=True`` (which is the default case), genes may
have the same value. If ``allow_duplicate_genes=False``, then no 2 genes
will have the same value given that there are enough unique values for
the genes.

The next code gives an example to use the ``allow_duplicate_genes``
parameter. A callback generation function is implemented to print the
population after each generation.

.. code:: python

   import pygad

   def fitness_func(ga_instance, solution, solution_idx):
       return 0

   def on_generation(ga):
       print("Generation", ga.generations_completed)
       print(ga.population)

   ga_instance = pygad.GA(num_generations=5,
                          sol_per_pop=5,
                          num_genes=4,
                          mutation_num_genes=3,
                          random_mutation_min_val=-5,
                          random_mutation_max_val=5,
                          num_parents_mating=2,
                          fitness_func=fitness_func,
                          gene_type=int,
                          on_generation=on_generation,
                          allow_duplicate_genes=False)
   ga_instance.run()

Here are the population after the 5 generations. Note how there are no
duplicate values.

.. code:: python

   Generation 1
   [[ 2 -2 -3  3]
    [ 0  1  2  3]
    [ 5 -3  6  3]
    [-3  1 -2  4]
    [-1  0 -2  3]]
   Generation 2
   [[-1  0 -2  3]
    [-3  1 -2  4]
    [ 0 -3 -2  6]
    [-3  0 -2  3]
    [ 1 -4  2  4]]
   Generation 3
   [[ 1 -4  2  4]
    [-3  0 -2  3]
    [ 4  0 -2  1]
    [-4  0 -2 -3]
    [-4  2  0  3]]
   Generation 4
   [[-4  2  0  3]
    [-4  0 -2 -3]
    [-2  5  4 -3]
    [-1  2 -4  4]
    [-4  2  0 -3]]
   Generation 5
   [[-4  2  0 -3]
    [-1  2 -4  4]
    [ 3  4 -4  0]
    [-1  0  2 -2]
    [-4  2 -1  1]]

The ``allow_duplicate_genes`` parameter is configured with use with the
``gene_space`` parameter. Here is an example where each of the 4 genes
has the same space of values that consists of 4 values (1, 2, 3, and 4).

.. code:: python

   import pygad

   def fitness_func(ga_instance, solution, solution_idx):
       return 0

   def on_generation(ga):
       print("Generation", ga.generations_completed)
       print(ga.population)

   ga_instance = pygad.GA(num_generations=1,
                          sol_per_pop=5,
                          num_genes=4,
                          num_parents_mating=2,
                          fitness_func=fitness_func,
                          gene_type=int,
                          gene_space=[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
                          on_generation=on_generation,
                          allow_duplicate_genes=False)
   ga_instance.run()

Even that all the genes share the same space of values, no 2 genes
duplicate their values as provided by the next output.

.. code:: python

   Generation 1
   [[2 3 1 4]
    [2 3 1 4]
    [2 4 1 3]
    [2 3 1 4]
    [1 3 2 4]]
   Generation 2
   [[1 3 2 4]
    [2 3 1 4]
    [1 3 2 4]
    [2 3 4 1]
    [1 3 4 2]]
   Generation 3
   [[1 3 4 2]
    [2 3 4 1]
    [1 3 4 2]
    [3 1 4 2]
    [3 2 4 1]]
   Generation 4
   [[3 2 4 1]
    [3 1 4 2]
    [3 2 4 1]
    [1 2 4 3]
    [1 3 4 2]]
   Generation 5
   [[1 3 4 2]
    [1 2 4 3]
    [2 1 4 3]
    [1 2 4 3]
    [1 2 4 3]]

You should care of giving enough values for the genes so that PyGAD is
able to find alternatives for the gene value in case it duplicates with
another gene.

There might be 2 duplicate genes where changing either of the 2
duplicating genes will not solve the problem. For example, if
``gene_space=[[3, 0, 1], [4, 1, 2], [0, 2], [3, 2, 0]]`` and the
solution is ``[3 2 0 0]``, then the values of the last 2 genes
duplicate. There are no possible changes in the last 2 genes to solve
the problem.

This problem can be solved by randomly changing one of the
non-duplicating genes that may make a room for a unique value in one the
2 duplicating genes. For example, by changing the second gene from 2 to
4, then any of the last 2 genes can take the value 2 and solve the
duplicates. The resultant gene is then ``[3 4 2 0]``. But this option is
not yet supported in PyGAD.

Solve Duplicates using a Third Gene
-----------------------------------

When ``allow_duplicate_genes=False`` and a user-defined ``gene_space``
is used, it sometimes happen that there is no room to solve the
duplicates between the 2 genes by simply replacing the value of one gene
by another gene. In `PyGAD
3.1.0 <https://pygad.readthedocs.io/en/latest/releases.html#pygad-3-0-1>`__,
the duplicates are solved by looking for a third gene that will help in
solving the duplicates. The following examples explain how it works.

Example 1:

Let's assume that this gene space is used and there is a solution with 2
duplicate genes with the same value 4.

.. code:: python

   Gene space: [[2, 3],
                [3, 4],
                [4, 5],
                [5, 6]]
   Solution: [3, 4, 4, 5]

By checking the gene space, the second gene can have the values
``[3, 4]`` and the third gene can have the values ``[4, 5]``. To solve
the duplicates, we have the value of any of these 2 genes.

If the value of the second gene changes from 4 to 3, then it will be
duplicate with the first gene. If we are to change the value of the
third gene from 4 to 5, then it will duplicate with the fourth gene. As
a conclusion, trying to just selecting a different gene value for either
the second or third genes will introduce new duplicating genes.

When there are 2 duplicate genes but there is no way to solve their
duplicates, then the solution is to change a third gene that makes a
room to solve the duplicates between the 2 genes.

In our example, duplicates between the second and third genes can be
solved by, for example,:

-  Changing the first gene from 3 to 2 then changing the second gene
   from 4 to 3.

-  Or changing the fourth gene from 5 to 6 then changing the third gene
   from 4 to 5.

Generally, this is how to solve such duplicates:

1. For any duplicate gene **GENE1**, select another value.

2. Check which other gene **GENEX** has duplicate with this new value.

3. Find if **GENEX** can have another value that will not cause any more
   duplicates. If so, go to step 7.

4. If all the other values of **GENEX** will cause duplicates, then try
   another gene **GENEY**.

5. Repeat steps 3 and 4 until exploring all the genes.

6. If there is no possibility to solve the duplicates, then there is not
   way to solve the duplicates and we have to keep the duplicate value.

7. If a value for a gene **GENEM** is found that will not cause more
   duplicates, then use this value for the gene **GENEM**.

8. Replace the value of the gene **GENE1** by the old value of the gene
   **GENEM**. This solves the duplicates.

This is an example to solve the duplicate for the solution
``[3, 4, 4, 5]``:

1. Let's use the second gene with value 4. Because the space of this
   gene is ``[3, 4]``, then the only other value we can select is 3.

2. The first gene also have the value 3.

3. The first gene has another value 2 that will not cause more
   duplicates in the solution. Then go to step 7.

4. Skip.

5. Skip.

6. Skip.

7. The value of the first gene 3 will be replaced by the new value 2.
   The new solution is [2, 4, 4, 5].

8. Replace the value of the second gene 4 by the old value of the first
   gene which is 3. The new solution is [2, 3, 4, 5]. The duplicate is
   solved.

Example 2:

.. code:: python

   Gene space: [[0, 1], 
                [1, 2], 
                [2, 3],
                [3, 4]]
   Solution: [1, 2, 2, 3]

The quick summary is:

-  Change the value of the first gene from 1 to 0. The solution becomes
   [0, 2, 2, 3].

-  Change the value of the second gene from 2 to 1. The solution becomes
   [0, 1, 2, 3]. The duplicate is solved.

.. _more-about-the-genetype-parameter:

More about the ``gene_type`` Parameter
======================================

The ``gene_type`` parameter allows the user to control the data type for
all genes at once or each individual gene. In `PyGAD
2.15.0 <https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-15-0>`__,
the ``gene_type`` parameter also supports customizing the precision for
``float`` data types. As a result, the ``gene_type`` parameter helps to:

1. Select a data type for all genes with or without precision.

2. Select a data type for each individual gene with or without
   precision.

Let's discuss things by examples.

Data Type for All Genes without Precision
-----------------------------------------

The data type for all genes can be specified by assigning the numeric
data type directly to the ``gene_type`` parameter. This is an example to
make all genes of ``int`` data types.

.. code:: python

   gene_type=int

Given that the supported numeric data types of PyGAD include Python's
``int`` and ``float`` in addition to all numeric types of ``NumPy``,
then any of these types can be assigned to the ``gene_type`` parameter.

If no precision is specified for a ``float`` data type, then the
complete floating-point number is kept.

The next code uses an ``int`` data type for all genes where the genes in
the initial and final population are only integers.

.. code:: python

   import pygad
   import numpy

   equation_inputs = [4, -2, 3.5, 8, -2]
   desired_output = 2671.1234

   def fitness_func(ga_instance, solution, solution_idx):
       output = numpy.sum(solution * equation_inputs)
       fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
       return fitness

   ga_instance = pygad.GA(num_generations=10,
                          sol_per_pop=5,
                          num_parents_mating=2,
                          num_genes=len(equation_inputs),
                          fitness_func=fitness_func,
                          gene_type=int)

   print("Initial Population")
   print(ga_instance.initial_population)

   ga_instance.run()

   print("Final Population")
   print(ga_instance.population)

.. code:: python

   Initial Population
   [[ 1 -1  2  0 -3]
    [ 0 -2  0 -3 -1]
    [ 0 -1 -1  2  0]
    [-2  3 -2  3  3]
    [ 0  0  2 -2 -2]]

   Final Population
   [[ 1 -1  2  2  0]
    [ 1 -1  2  2  0]
    [ 1 -1  2  2  0]
    [ 1 -1  2  2  0]
    [ 1 -1  2  2  0]]

Data Type for All Genes with Precision
--------------------------------------

A precision can only be specified for a ``float`` data type and cannot
be specified for integers. Here is an example to use a precision of 3
for the ``float`` data type. In this case, all genes are of type
``float`` and their maximum precision is 3.

.. code:: python

   gene_type=[float, 3]

The next code uses prints the initial and final population where the
genes are of type ``float`` with precision 3.

.. code:: python

   import pygad
   import numpy

   equation_inputs = [4, -2, 3.5, 8, -2]
   desired_output = 2671.1234

   def fitness_func(ga_instance, solution, solution_idx):
       output = numpy.sum(solution * equation_inputs)
       fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)

       return fitness

   ga_instance = pygad.GA(num_generations=10,
                          sol_per_pop=5,
                          num_parents_mating=2,
                          num_genes=len(equation_inputs),
                          fitness_func=fitness_func,
                          gene_type=[float, 3])

   print("Initial Population")
   print(ga_instance.initial_population)

   ga_instance.run()

   print("Final Population")
   print(ga_instance.population)

.. code:: python

   Initial Population
   [[-2.417 -0.487  3.623  2.457 -2.362]
    [-1.231  0.079 -1.63   1.629 -2.637]
    [ 0.692 -2.098  0.705  0.914 -3.633]
    [ 2.637 -1.339 -1.107 -0.781 -3.896]
    [-1.495  1.378 -1.026  3.522  2.379]]

   Final Population
   [[ 1.714 -1.024  3.623  3.185 -2.362]
    [ 0.692 -1.024  3.623  3.185 -2.362]
    [ 0.692 -1.024  3.623  3.375 -2.362]
    [ 0.692 -1.024  4.041  3.185 -2.362]
    [ 1.714 -0.644  3.623  3.185 -2.362]]

Data Type for each Individual Gene without Precision
----------------------------------------------------

In `PyGAD
2.14.0 <https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-14-0>`__,
the ``gene_type`` parameter allows customizing the gene type for each
individual gene. This is by using a ``list``/``tuple``/``numpy.ndarray``
with number of elements equal to the number of genes. For each element,
a type is specified for the corresponding gene.

This is an example for a 5-gene problem where different types are
assigned to the genes.

.. code:: python

   gene_type=[int, float, numpy.float16, numpy.int8, float]

This is a complete code that prints the initial and final population for
a custom-gene data type.

.. code:: python

   import pygad
   import numpy

   equation_inputs = [4, -2, 3.5, 8, -2]
   desired_output = 2671.1234

   def fitness_func(ga_instance, solution, solution_idx):
       output = numpy.sum(solution * equation_inputs)
       fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
       return fitness

   ga_instance = pygad.GA(num_generations=10,
                          sol_per_pop=5,
                          num_parents_mating=2,
                          num_genes=len(equation_inputs),
                          fitness_func=fitness_func,
                          gene_type=[int, float, numpy.float16, numpy.int8, float])

   print("Initial Population")
   print(ga_instance.initial_population)

   ga_instance.run()

   print("Final Population")
   print(ga_instance.population)

.. code:: python

   Initial Population
   [[0 0.8615522360026828 0.7021484375 -2 3.5301821368185866]
    [-3 2.648189378595294 -3.830078125 1 -0.9586271572917742]
    [3 3.7729827570110714 1.2529296875 -3 1.395741994211889]
    [0 1.0490687178053282 1.51953125 -2 0.7243617940450235]
    [0 -0.6550158436937226 -2.861328125 -2 1.8212734549263097]]

   Final Population
   [[3 3.7729827570110714 2.055 0 0.7243617940450235]
    [3 3.7729827570110714 1.458 0 -0.14638754050305036]
    [3 3.7729827570110714 1.458 0 0.0869406120516778]
    [3 3.7729827570110714 1.458 0 0.7243617940450235]
    [3 3.7729827570110714 1.458 0 -0.14638754050305036]]

Data Type for each Individual Gene with Precision
-------------------------------------------------

The precision can also be specified for the ``float`` data types as in
the next line where the second gene precision is 2 and last gene
precision is 1.

.. code:: python

   gene_type=[int, [float, 2], numpy.float16, numpy.int8, [float, 1]]

This is a complete example where the initial and final populations are
printed where the genes comply with the data types and precisions
specified.

.. code:: python

   import pygad
   import numpy

   equation_inputs = [4, -2, 3.5, 8, -2]
   desired_output = 2671.1234

   def fitness_func(ga_instance, solution, solution_idx):
       output = numpy.sum(solution * equation_inputs)
       fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
       return fitness

   ga_instance = pygad.GA(num_generations=10,
                          sol_per_pop=5,
                          num_parents_mating=2,
                          num_genes=len(equation_inputs),
                          fitness_func=fitness_func,
                          gene_type=[int, [float, 2], numpy.float16, numpy.int8, [float, 1]])

   print("Initial Population")
   print(ga_instance.initial_population)

   ga_instance.run()

   print("Final Population")
   print(ga_instance.population)

.. code:: python

   Initial Population
   [[-2 -1.22 1.716796875 -1 0.2]
    [-1 -1.58 -3.091796875 0 -1.3]
    [3 3.35 -0.107421875 1 -3.3]
    [-2 -3.58 -1.779296875 0 0.6]
    [2 -3.73 2.65234375 3 -0.5]]

   Final Population
   [[2 -4.22 3.47 3 -1.3]
    [2 -3.73 3.47 3 -1.3]
    [2 -4.22 3.47 2 -1.3]
    [2 -4.58 3.47 3 -1.3]
    [2 -3.73 3.47 3 -1.3]]

Parallel Processing in PyGAD
============================

Starting from `PyGAD
2.17.0 <https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-17-0>`__,
parallel processing becomes supported. This section explains how to use
parallel processing in PyGAD.

According to the `PyGAD
lifecycle <https://pygad.readthedocs.io/en/latest/pygad.html#life-cycle-of-pygad>`__,
parallel processing can be parallelized in only 2 operations:

1. Population fitness calculation.

2. Mutation.

The reason is that the calculations in these 2 operations are
independent (i.e. each solution/chromosome is handled independently from
the others) and can be distributed across different processes or
threads.

For the mutation operation, it does not do intensive calculations on the
CPU. Its calculations are simple like flipping the values of some genes
from 0 to 1 or adding a random value to some genes. So, it does not take
much CPU processing time. Experiments proved that parallelizing the
mutation operation across the solutions increases the time instead of
reducing it. This is because running multiple processes or threads adds
overhead to manage them. Thus, parallel processing cannot be applied on
the mutation operation.

For the population fitness calculation, parallel processing can help
make a difference and reduce the processing time. But this is
conditional on the type of calculations done in the fitness function. If
the fitness function makes intensive calculations and takes much
processing time from the CPU, then it is probably that parallel
processing will help to cut down the overall time.

This section explains how parallel processing works in PyGAD and how to
use parallel processing in PyGAD

How to Use Parallel Processing in PyGAD
---------------------------------------

Starting from `PyGAD
2.17.0 <https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-17-0>`__,
a new parameter called ``parallel_processing`` added to the constructor
of the ``pygad.GA`` class.

.. code:: python

   import pygad
   ...
   ga_instance = pygad.GA(...,
                          parallel_processing=...)
   ...

This parameter allows the user to do the following:

1. Enable parallel processing.

2. Select whether processes or threads are used.

3. Specify the number of processes or threads to be used.

These are 3 possible values for the ``parallel_processing`` parameter:

1. ``None``: (Default) It means no parallel processing is used.

2. A positive integer referring to the number of threads to be used
   (i.e. threads, not processes, are used.

3. ``list``/``tuple``: If a list or a tuple of exactly 2 elements is
   assigned, then:

   1. The first element can be either ``'process'`` or ``'thread'`` to
      specify whether processes or threads are used, respectively.

   2. The second element can be:

      1. A positive integer to select the maximum number of processes or
         threads to be used

      2. ``0`` to indicate that 0 processes or threads are used. It
         means no parallel processing. This is identical to setting
         ``parallel_processing=None``.

      3. ``None`` to use the default value as calculated by the
         ``concurrent.futures module``.

These are examples of the values assigned to the ``parallel_processing``
parameter:

-  ``parallel_processing=4``: Because the parameter is assigned a
   positive integer, this means parallel processing is activated where 4
   threads are used.

-  ``parallel_processing=["thread", 5]``: Use parallel processing with 5
   threads. This is identical to ``parallel_processing=5``.

-  ``parallel_processing=["process", 8]``: Use parallel processing with
   8 processes.

-  ``parallel_processing=["process", 0]``: As the second element is
   given the value 0, this means do not use parallel processing. This is
   identical to ``parallel_processing=None``.

Examples
--------

The examples will help you know the difference between using processes
and threads. Moreover, it will give an idea when parallel processing
would make a difference and reduce the time. These are dummy examples
where the fitness function is made to always return 0.

The first example uses 10 genes, 5 solutions in the population where
only 3 solutions mate, and 9999 generations. The fitness function uses a
``for`` loop with 100 iterations just to have some calculations. In the
constructor of the ``pygad.GA`` class, ``parallel_processing=None``
means no parallel processing is used.

.. code:: python

   import pygad
   import time

   def fitness_func(ga_instance, solution, solution_idx):
       for _ in range(99):
           pass
       return 0

   ga_instance = pygad.GA(num_generations=9999,
                          num_parents_mating=3,
                          sol_per_pop=5,
                          num_genes=10,
                          fitness_func=fitness_func,
                          suppress_warnings=True,
                          parallel_processing=None)

   if __name__ == '__main__':
       t1 = time.time()

       ga_instance.run()

       t2 = time.time()
       print("Time is", t2-t1)

When parallel processing is not used, the time it takes to run the
genetic algorithm is ``1.5`` seconds.

In the comparison, let's do a second experiment where parallel
processing is used with 5 threads. In this case, it take ``5`` seconds.

.. code:: python

   ...
   ga_instance = pygad.GA(...,
                          parallel_processing=5)
   ...

For the third experiment, processes instead of threads are used. Also,
only 99 generations are used instead of 9999. The time it takes is
``99`` seconds.

.. code:: python

   ...
   ga_instance = pygad.GA(num_generations=99,
                          ...,
                          parallel_processing=["process", 5])
   ...

This is the summary of the 3 experiments:

1. No parallel processing & 9999 generations: 1.5 seconds.

2. Parallel processing with 5 threads & 9999 generations: 5 seconds

3. Parallel processing with 5 processes & 99 generations: 99 seconds

Because the fitness function does not need much CPU time, the normal
processing takes the least time. Running processes for this simple
problem takes 99 compared to only 5 seconds for threads because managing
processes is much heavier than managing threads. Thus, most of the CPU
time is for swapping the processes instead of executing the code.

In the second example, the loop makes 99999999 iterations and only 5
generations are used. With no parallelization, it takes 22 seconds.

.. code:: python

   import pygad
   import time

   def fitness_func(ga_instance, solution, solution_idx):
       for _ in range(99999999):
           pass
       return 0

   ga_instance = pygad.GA(num_generations=5,
                          num_parents_mating=3,
                          sol_per_pop=5,
                          num_genes=10,
                          fitness_func=fitness_func,
                          suppress_warnings=True,
                          parallel_processing=None)

   if __name__ == '__main__':
       t1 = time.time()
       ga_instance.run()
       t2 = time.time()
       print("Time is", t2-t1)

It takes 15 seconds when 10 processes are used.

.. code:: python

   ...
   ga_instance = pygad.GA(...,
                          parallel_processing=["process", 10])
   ...

This is compared to 20 seconds when 10 threads are used.

.. code:: python

   ...
   ga_instance = pygad.GA(...,
                          parallel_processing=["thread", 10])
   ...

Based on the second example, using parallel processing with 10 processes
takes the least time because there is much CPU work done. Generally,
processes are preferred over threads when most of the work in on the
CPU. Threads are preferred over processes in some situations like doing
input/output operations.

*Before releasing* `PyGAD
2.17.0 <https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-17-0>`__\ *,*
`László
Fazekas <https://www.linkedin.com/in/l%C3%A1szl%C3%B3-fazekas-2429a912>`__
*wrote an article to parallelize the fitness function with PyGAD. Check
it:* `How Genetic Algorithms Can Compete with Gradient Descent and
Backprop <https://hackernoon.com/how-genetic-algorithms-can-compete-with-gradient-descent-and-backprop-9m9t33bq>`__.

Print Lifecycle Summary
=======================

In `PyGAD
2.19.0 <https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-19-0>`__,
a new method called ``summary()`` is supported. It prints a Keras-like
summary of the PyGAD lifecycle showing the steps, callback functions,
parameters, etc.

This method accepts the following parameters:

-  ``line_length=70``: An integer representing the length of the single
   line in characters.

-  ``fill_character=" "``: A character to fill the lines.

-  ``line_character="-"``: A character for creating a line separator.

-  ``line_character2="="``: A secondary character to create a line
   separator.

-  ``columns_equal_len=False``: The table rows are split into
   equal-sized columns or split subjective to the width needed.

-  ``print_step_parameters=True``: Whether to print extra parameters
   about each step inside the step. If ``print_step_parameters=False``
   and ``print_parameters_summary=True``, then the parameters of each
   step are printed at the end of the table.

-  ``print_parameters_summary=True``: Whether to print parameters
   summary at the end of the table. If ``print_step_parameters=False``,
   then the parameters of each step are printed at the end of the table
   too.

This is a quick example to create a PyGAD example.

.. code:: python

   import pygad
   import numpy

   function_inputs = [4,-2,3.5,5,-11,-4.7]
   desired_output = 44

   def genetic_fitness(solution, solution_idx):
       output = numpy.sum(solution*function_inputs)
       fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
       return fitness

   def on_gen(ga):
       pass

   def on_crossover_callback(a, b):
       pass

   ga_instance = pygad.GA(num_generations=100,
                          num_parents_mating=10,
                          sol_per_pop=20,
                          num_genes=len(function_inputs),
                          on_crossover=on_crossover_callback,
                          on_generation=on_gen,
                          parallel_processing=2,
                          stop_criteria="reach_10",
                          fitness_batch_size=4,
                          crossover_probability=0.4,
                          fitness_func=genetic_fitness)

Then call the ``summary()`` method to print the summary with the default
parameters. Note that entries for the crossover and generation callback
function are created because their callback functions are implemented
through the ``on_crossover_callback()`` and ``on_gen()``, respectively.

.. code:: python

   ga_instance.summary()

.. code:: bash

   ----------------------------------------------------------------------
                              PyGAD Lifecycle                           
   ======================================================================
   Step                   Handler                            Output Shape
   ======================================================================
   Fitness Function       genetic_fitness()                  (1)      
   Fitness batch size: 4
   ----------------------------------------------------------------------
   Parent Selection       steady_state_selection()           (10, 6)  
   Number of Parents: 10
   ----------------------------------------------------------------------
   Crossover              single_point_crossover()           (10, 6)  
   Crossover probability: 0.4
   ----------------------------------------------------------------------
   On Crossover           on_crossover_callback()            None     
   ----------------------------------------------------------------------
   Mutation               random_mutation()                  (10, 6)  
   Mutation Genes: 1
   Random Mutation Range: (-1.0, 1.0)
   Mutation by Replacement: False
   Allow Duplicated Genes: True
   ----------------------------------------------------------------------
   On Generation          on_gen()                           None     
   Stop Criteria: [['reach', 10.0]]
   ----------------------------------------------------------------------
   ======================================================================
   Population Size: (20, 6)
   Number of Generations: 100
   Initial Population Range: (-4, 4)
   Keep Elitism: 1
   Gene DType: [<class 'float'>, None]
   Parallel Processing: ['thread', 2]
   Save Best Solutions: False
   Save Solutions: False
   ======================================================================

We can set the ``print_step_parameters`` and
``print_parameters_summary`` parameters to ``False`` to not print the
parameters.

.. code:: python

   ga_instance.summary(print_step_parameters=False,
                       print_parameters_summary=False)

.. code:: bash

   ----------------------------------------------------------------------
                              PyGAD Lifecycle                           
   ======================================================================
   Step                   Handler                            Output Shape
   ======================================================================
   Fitness Function       genetic_fitness()                  (1)      
   ----------------------------------------------------------------------
   Parent Selection       steady_state_selection()           (10, 6)  
   ----------------------------------------------------------------------
   Crossover              single_point_crossover()           (10, 6)  
   ----------------------------------------------------------------------
   On Crossover           on_crossover_callback()            None     
   ----------------------------------------------------------------------
   Mutation               random_mutation()                  (10, 6)  
   ----------------------------------------------------------------------
   On Generation          on_gen()                           None     
   ----------------------------------------------------------------------
   ======================================================================

Logging Outputs
===============

In `PyGAD
3.0.0 <https://pygad.readthedocs.io/en/latest/releases.html#pygad-3-0-0>`__,
the ``print()`` statement is no longer used and the outputs are printed
using the `logging <https://docs.python.org/3/library/logging.html>`__
module. A a new parameter called ``logger`` is supported to accept the
user-defined logger.

.. code:: python

   import logging

   logger = ...

   ga_instance = pygad.GA(...,
                          logger=logger,
                          ...)

The default value for this parameter is ``None``. If there is no logger
passed (i.e. ``logger=None``), then a default logger is created to log
the messages to the console exactly like how the ``print()`` statement
works.

Some advantages of using the the
`logging <https://docs.python.org/3/library/logging.html>`__ module
instead of the ``print()`` statement are:

1. The user has more control over the printed messages specially if
   there is a project that uses multiple modules where each module
   prints its messages. A logger can organize the outputs.

2. Using the proper ``Handler``, the user can log the output messages to
   files and not only restricted to printing it to the console. So, it
   is much easier to record the outputs.

3. The format of the printed messages can be changed by customizing the
   ``Formatter`` assigned to the Logger.

This section gives some quick examples to use the ``logging`` module and
then gives an example to use the logger with PyGAD.

Logging to the Console
----------------------

This is an example to create a logger to log the messages to the
console.

.. code:: python

   import logging

   # Create a logger
   logger = logging.getLogger(__name__)

   # Set the logger level to debug so that all the messages are printed.
   logger.setLevel(logging.DEBUG)

   # Create a stream handler to log the messages to the console.
   stream_handler = logging.StreamHandler()

   # Set the handler level to debug.
   stream_handler.setLevel(logging.DEBUG)

   # Create a formatter
   formatter = logging.Formatter('%(message)s')

   # Add the formatter to handler.
   stream_handler.setFormatter(formatter)

   # Add the stream handler to the logger
   logger.addHandler(stream_handler)

Now, we can log messages to the console with the format specified in the
``Formatter``.

.. code:: python

   logger.debug('Debug message.')
   logger.info('Info message.')
   logger.warning('Warn message.')
   logger.error('Error message.')
   logger.critical('Critical message.')

The outputs are identical to those returned using the ``print()``
statement.

.. code:: 

   Debug message.
   Info message.
   Warn message.
   Error message.
   Critical message.

By changing the format of the output messages, we can have more
information about each message.

.. code:: python

   formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

This is a sample output.

.. code:: python

   2023-04-03 18:46:27 DEBUG: Debug message.
   2023-04-03 18:46:27 INFO: Info message.
   2023-04-03 18:46:27 WARNING: Warn message.
   2023-04-03 18:46:27 ERROR: Error message.
   2023-04-03 18:46:27 CRITICAL: Critical message.

Note that you may need to clear the handlers after finishing the
execution. This is to make sure no cached handlers are used in the next
run. If the cached handlers are not cleared, then the single output
message may be repeated.

.. code:: python

   logger.handlers.clear()

Logging to a File
-----------------

This is another example to log the messages to a file named
``logfile.txt``. The formatter prints the following about each message:

1. The date and time at which the message is logged.

2. The log level.

3. The message.

4. The path of the file.

5. The lone number of the log message.

.. code:: python

   import logging

   level = logging.DEBUG
   name = 'logfile.txt'

   logger = logging.getLogger(name)
   logger.setLevel(level)

   file_handler = logging.FileHandler(name, 'a+', 'utf-8')
   file_handler.setLevel(logging.DEBUG)
   file_format = logging.Formatter('%(asctime)s %(levelname)s: %(message)s - %(pathname)s:%(lineno)d', datefmt='%Y-%m-%d %H:%M:%S')
   file_handler.setFormatter(file_format)
   logger.addHandler(file_handler)

This is how the outputs look like.

.. code:: python

   2023-04-03 18:54:03 DEBUG: Debug message. - c:\users\agad069\desktop\logger\example2.py:46
   2023-04-03 18:54:03 INFO: Info message. - c:\users\agad069\desktop\logger\example2.py:47
   2023-04-03 18:54:03 WARNING: Warn message. - c:\users\agad069\desktop\logger\example2.py:48
   2023-04-03 18:54:03 ERROR: Error message. - c:\users\agad069\desktop\logger\example2.py:49
   2023-04-03 18:54:03 CRITICAL: Critical message. - c:\users\agad069\desktop\logger\example2.py:50

Consider clearing the handlers if necessary.

.. code:: python

   logger.handlers.clear()

Log to Both the Console and a File
----------------------------------

This is an example to create a single Logger associated with 2 handlers:

1. A file handler.

2. A stream handler.

.. code:: python

   import logging

   level = logging.DEBUG
   name = 'logfile.txt'

   logger = logging.getLogger(name)
   logger.setLevel(level)

   file_handler = logging.FileHandler(name,'a+','utf-8')
   file_handler.setLevel(logging.DEBUG)
   file_format = logging.Formatter('%(asctime)s %(levelname)s: %(message)s - %(pathname)s:%(lineno)d', datefmt='%Y-%m-%d %H:%M:%S')
   file_handler.setFormatter(file_format)
   logger.addHandler(file_handler)

   console_handler = logging.StreamHandler()
   console_handler.setLevel(logging.INFO)
   console_format = logging.Formatter('%(message)s')
   console_handler.setFormatter(console_format)
   logger.addHandler(console_handler)

When a log message is executed, then it is both printed to the console
and saved in the ``logfile.txt``.

Consider clearing the handlers if necessary.

.. code:: python

   logger.handlers.clear()

PyGAD Example
-------------

To use the logger in PyGAD, just create your custom logger and pass it
to the ``logger`` parameter.

.. code:: python

   import logging
   import pygad
   import numpy

   level = logging.DEBUG
   name = 'logfile.txt'

   logger = logging.getLogger(name)
   logger.setLevel(level)

   file_handler = logging.FileHandler(name,'a+','utf-8')
   file_handler.setLevel(logging.DEBUG)
   file_format = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
   file_handler.setFormatter(file_format)
   logger.addHandler(file_handler)

   console_handler = logging.StreamHandler()
   console_handler.setLevel(logging.INFO)
   console_format = logging.Formatter('%(message)s')
   console_handler.setFormatter(console_format)
   logger.addHandler(console_handler)

   equation_inputs = [4, -2, 8]
   desired_output = 2671.1234

   def fitness_func(ga_instance, solution, solution_idx):
       output = numpy.sum(solution * equation_inputs)
       fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
       return fitness

   def on_generation(ga_instance):
       ga_instance.logger.info(f"Generation = {ga_instance.generations_completed}")
       ga_instance.logger.info(f"Fitness    = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}")

   ga_instance = pygad.GA(num_generations=10,
                          sol_per_pop=40,
                          num_parents_mating=2,
                          keep_parents=2,
                          num_genes=len(equation_inputs),
                          fitness_func=fitness_func,
                          on_generation=on_generation,
                          logger=logger)
   ga_instance.run()

   logger.handlers.clear()

By executing this code, the logged messages are printed to the console
and also saved in the text file.

.. code:: python

   2023-04-03 19:04:27 INFO: Generation = 1
   2023-04-03 19:04:27 INFO: Fitness    = 0.00038086960368076276
   2023-04-03 19:04:27 INFO: Generation = 2
   2023-04-03 19:04:27 INFO: Fitness    = 0.00038214871408010853
   2023-04-03 19:04:27 INFO: Generation = 3
   2023-04-03 19:04:27 INFO: Fitness    = 0.0003832795907974678
   2023-04-03 19:04:27 INFO: Generation = 4
   2023-04-03 19:04:27 INFO: Fitness    = 0.00038398612055017196
   2023-04-03 19:04:27 INFO: Generation = 5
   2023-04-03 19:04:27 INFO: Fitness    = 0.00038442348890867516
   2023-04-03 19:04:27 INFO: Generation = 6
   2023-04-03 19:04:27 INFO: Fitness    = 0.0003854406039137763
   2023-04-03 19:04:27 INFO: Generation = 7
   2023-04-03 19:04:27 INFO: Fitness    = 0.00038646083174063284
   2023-04-03 19:04:27 INFO: Generation = 8
   2023-04-03 19:04:27 INFO: Fitness    = 0.0003875169193024936
   2023-04-03 19:04:27 INFO: Generation = 9
   2023-04-03 19:04:27 INFO: Fitness    = 0.0003888816727311021
   2023-04-03 19:04:27 INFO: Generation = 10
   2023-04-03 19:04:27 INFO: Fitness    = 0.000389832593101348

Solve Non-Deterministic Problems
================================

PyGAD can be used to solve both deterministic and non-deterministic
problems. Deterministic are those that return the same fitness for the
same solution. For non-deterministic problems, a different fitness value
would be returned for the same solution.

By default, PyGAD settings are set to solve deterministic problems.
PyGAD can save the explored solutions and their fitness to reuse in the
future. These instances attributes can save the solutions:

1. ``solutions``: Exists if ``save_solutions=True``.

2. ``best_solutions``: Exists if ``save_best_solutions=True``.

3. ``last_generation_elitism``: Exists if ``keep_elitism`` > 0.

4. ``last_generation_parents``: Exists if ``keep_parents`` > 0 or
   ``keep_parents=-1``.

To configure PyGAD for non-deterministic problems, we have to disable
saving the previous solutions. This is by setting these parameters:

1. ``keep_elisitm=0``

2. ``keep_parents=0``

3. ``keep_solutions=False``

4. ``keep_best_solutions=False``

.. code:: python

   import pygad
   ...
   ga_instance = pygad.GA(...,
                          keep_elitism=0,
                          keep_parents=0,
                          save_solutions=False,
                          save_best_solutions=False,
                          ...)

This way PyGAD will not save any explored solution and thus the fitness
function have to be called for each individual solution.

Reuse the Fitness instead of Calling the Fitness Function
=========================================================

It may happen that a previously explored solution in generation X is
explored again in another generation Y (where Y > X). For some problems,
calling the fitness function takes much time.

For deterministic problems, it is better to not call the fitness
function for an already explored solutions. Instead, reuse the fitness
of the old solution. PyGAD supports some options to help you save time
calling the fitness function for a previously explored solution.

The parameters explored in this section can be set in the constructor of
the ``pygad.GA`` class.

The ``cal_pop_fitness()`` method of the ``pygad.GA`` class checks these
parameters to see if there is a possibility of reusing the fitness
instead of calling the fitness function.

.. _1-savesolutions:

1. ``save_solutions``
---------------------

It defaults to ``False``. If set to ``True``, then the population of
each generation is saved into the ``solutions`` attribute of the
``pygad.GA`` instance. In other words, every single solution is saved in
the ``solutions`` attribute.

.. _2-savebestsolutions:

2. ``save_best_solutions``
--------------------------

It defaults to ``False``. If ``True``, then it only saves the best
solution in every generation.

.. _3-keepelitism:

3. ``keep_elitism``
-------------------

It accepts an integer and defaults to 1. If set to a positive integer,
then it keeps the elitism of one generation available in the next
generation.

.. _4-keepparents:

4. ``keep_parents``
-------------------

It accepts an integer and defaults to -1. It set to ``-1`` or a positive
integer, then it keeps the parents of one generation available in the
next generation.

Why the Fitness Function is not Called for Solution at Index 0?
===============================================================

PyGAD has a parameter called ``keep_elitism`` which defaults to 1. This
parameter defines the number of best solutions in generation **X** to
keep in the next generation **X+1**. The best solutions are just copied
from generation **X** to generation **X+1** without making any change.

.. code:: python

   ga_instance = pygad.GA(...,
                          keep_elitism=1,
                          ...)

The best solutions are copied at the beginning of the population. If
``keep_elitism=1``, this means the best solution in generation X is kept
in the next generation X+1 at index 0 of the population. If
``keep_elitism=2``, this means the 2 best solutions in generation X are
kept in the next generation X+1 at indices 0 and 1 of the population of
generation 1.

Because the fitness of these best solutions are already calculated in
generation X, then their fitness values will not be recalculated at
generation X+1 (i.e. the fitness function will not be called for these
solutions again). Instead, their fitness values are just reused. This is
why you see that no solution with index 0 is passed to the fitness
function.

To force calling the fitness function for each solution in every
generation, consider setting ``keep_elitism`` and ``keep_parents`` to 0.
Moreover, keep the 2 parameters ``save_solutions`` and
``save_best_solutions`` to their default value ``False``.

.. code:: python

   ga_instance = pygad.GA(...,
                          keep_elitism=0,
                          keep_parents=0,
                          save_solutions=False,
                          save_best_solutions=False,
                          ...)

Batch Fitness Calculation
=========================

In `PyGAD
2.19.0 <https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-19-0>`__,
a new optional parameter called ``fitness_batch_size`` is supported. A
new optional parameter called ``fitness_batch_size`` is supported to
calculate the fitness function in batches. Thanks to `Linan
Qiu <https://github.com/linanqiu>`__ for opening the `GitHub issue
#136 <https://github.com/ahmedfgad/GeneticAlgorithmPython/issues/136>`__.

Its values can be:

-  ``1`` or ``None``: If the ``fitness_batch_size`` parameter is
   assigned the value ``1`` or ``None`` (default), then the normal flow
   is used where the fitness function is called for each individual
   solution. That is if there are 15 solutions, then the fitness
   function is called 15 times.

-  ``1 < fitness_batch_size <= sol_per_pop``: If the
   ``fitness_batch_size`` parameter is assigned a value satisfying this
   condition ``1 < fitness_batch_size <= sol_per_pop``, then the
   solutions are grouped into batches of size ``fitness_batch_size`` and
   the fitness function is called once for each batch. In this case, the
   fitness function must return a list/tuple/numpy.ndarray with a length
   equal to the number of solutions passed.

.. _example-without-fitnessbatchsize-parameter:

Example without ``fitness_batch_size`` Parameter
------------------------------------------------

This is an example where the ``fitness_batch_size`` parameter is given
the value ``None`` (which is the default value). This is equivalent to
using the value ``1``. In this case, the fitness function will be called
for each solution. This means the fitness function ``fitness_func`` will
receive only a single solution. This is an example of the passed
arguments to the fitness function:

.. code:: 

   solution: [ 2.52860734, -0.94178795, 2.97545704, 0.84131987, -3.78447118, 2.41008358]
   solution_idx: 3

The fitness function also must return a single numeric value as the
fitness for the passed solution.

As we have a population of ``20`` solutions, then the fitness function
is called 20 times per generation. For 5 generations, then the fitness
function is called ``20*5 = 100`` times. In PyGAD, the fitness function
is called after the last generation too and this adds additional 20
times. So, the total number of calls to the fitness function is
``20*5 + 20 = 120``.

Note that the ``keep_elitism`` and ``keep_parents`` parameters are set
to ``0`` to make sure no fitness values are reused and to force calling
the fitness function for each individual solution.

.. code:: python

   import pygad
   import numpy

   function_inputs = [4,-2,3.5,5,-11,-4.7]
   desired_output = 44

   number_of_calls = 0

   def fitness_func(ga_instance, solution, solution_idx):
       global number_of_calls
       number_of_calls = number_of_calls + 1
       output = numpy.sum(solution*function_inputs)
       fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
       return fitness

   ga_instance = pygad.GA(num_generations=5,
                          num_parents_mating=10,
                          sol_per_pop=20,
                          fitness_func=fitness_func,
                          fitness_batch_size=None,
                          # fitness_batch_size=1,
                          num_genes=len(function_inputs),
                          keep_elitism=0,
                          keep_parents=0)

   ga_instance.run()
   print(number_of_calls)

.. code:: 

   120

.. _example-with-fitnessbatchsize-parameter:

Example with ``fitness_batch_size`` Parameter
---------------------------------------------

This is an example where the ``fitness_batch_size`` parameter is used
and assigned the value ``4``. This means the solutions will be grouped
into batches of ``4`` solutions. The fitness function will be called
once for each patch (i.e. called once for each 4 solutions).

This is an example of the arguments passed to it:

.. code:: python

   solutions:
       [[ 3.1129432  -0.69123589  1.93792414  2.23772968 -1.54616001 -0.53930799]
        [ 3.38508121  0.19890812  1.93792414  2.23095014 -3.08955597  3.10194128]
        [ 2.37079504 -0.88819803  2.97545704  1.41742256 -3.95594055  2.45028256]
        [ 2.52860734 -0.94178795  2.97545704  0.84131987 -3.78447118  2.41008358]]
   solutions_indices:
       [16, 17, 18, 19]

As we have 20 solutions, then there are ``20/4 = 5`` patches. As a
result, the fitness function is called only 5 times per generation
instead of 20. For each call to the fitness function, it receives a
batch of 4 solutions.

As we have 5 generations, then the function will be called ``5*5 = 25``
times. Given the call to the fitness function after the last generation,
then the total number of calls is ``5*5 + 5 = 30``.

.. code:: python

   import pygad
   import numpy

   function_inputs = [4,-2,3.5,5,-11,-4.7]
   desired_output = 44

   number_of_calls = 0

   def fitness_func_batch(ga_instance, solutions, solutions_indices):
       global number_of_calls
       number_of_calls = number_of_calls + 1
       batch_fitness = []
       for solution in solutions:
           output = numpy.sum(solution*function_inputs)
           fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
           batch_fitness.append(fitness)
       return batch_fitness

   ga_instance = pygad.GA(num_generations=5,
                          num_parents_mating=10,
                          sol_per_pop=20,
                          fitness_func=fitness_func_batch,
                          fitness_batch_size=4,
                          num_genes=len(function_inputs),
                          keep_elitism=0,
                          keep_parents=0)

   ga_instance.run()
   print(number_of_calls)

.. code:: 

   30

When batch fitness calculation is used, then we saved ``120 - 30 = 90``
calls to the fitness function.

Use Functions and Methods to Build Fitness and Callbacks
========================================================

In PyGAD 2.19.0, it is possible to pass user-defined functions or
methods to the following parameters:

1. ``fitness_func``

2. ``on_start``

3. ``on_fitness``

4. ``on_parents``

5. ``on_crossover``

6. ``on_mutation``

7. ``on_generation``

8. ``on_stop``

This section gives 2 examples to assign these parameters user-defined:

1. Functions.

2. Methods.

Assign Functions
----------------

This is a dummy example where the fitness function returns a random
value. Note that the instance of the ``pygad.GA`` class is passed as the
last parameter of all functions.

.. code:: python

   import pygad
   import numpy

   def fitness_func(ga_instanse, solution, solution_idx):
       return numpy.random.rand()

   def on_start(ga_instanse):
       print("on_start")

   def on_fitness(ga_instanse, last_gen_fitness):
       print("on_fitness")

   def on_parents(ga_instanse, last_gen_parents):
       print("on_parents")

   def on_crossover(ga_instanse, last_gen_offspring):
       print("on_crossover")

   def on_mutation(ga_instanse, last_gen_offspring):
       print("on_mutation")

   def on_generation(ga_instanse):
       print("on_generation\n")

   def on_stop(ga_instanse, last_gen_fitness):
       print("on_stop")

   ga_instance = pygad.GA(num_generations=5,
                          num_parents_mating=4,
                          sol_per_pop=10,
                          num_genes=2,
                          on_start=on_start,
                          on_fitness=on_fitness,
                          on_parents=on_parents,
                          on_crossover=on_crossover,
                          on_mutation=on_mutation,
                          on_generation=on_generation,
                          on_stop=on_stop,
                          fitness_func=fitness_func)
       
   ga_instance.run()

Assign Methods
--------------

The next example has all the method defined inside the class ``Test``.
All of the methods accept an additional parameter representing the
method's object of the class ``Test``.

All methods accept ``self`` as the first parameter and the instance of
the ``pygad.GA`` class as the last parameter.

.. code:: python

   import pygad
   import numpy

   class Test:
       def fitness_func(self, ga_instanse, solution, solution_idx):
           return numpy.random.rand()

       def on_start(self, ga_instanse):
           print("on_start")

       def on_fitness(self, ga_instanse, last_gen_fitness):
           print("on_fitness")

       def on_parents(self, ga_instanse, last_gen_parents):
           print("on_parents")

       def on_crossover(self, ga_instanse, last_gen_offspring):
           print("on_crossover")

       def on_mutation(self, ga_instanse, last_gen_offspring):
           print("on_mutation")

       def on_generation(self, ga_instanse):
           print("on_generation\n")

       def on_stop(self, ga_instanse, last_gen_fitness):
           print("on_stop")

   ga_instance = pygad.GA(num_generations=5,
                          num_parents_mating=4,
                          sol_per_pop=10,
                          num_genes=2,
                          on_start=Test().on_start,
                          on_fitness=Test().on_fitness,
                          on_parents=Test().on_parents,
                          on_crossover=Test().on_crossover,
                          on_mutation=Test().on_mutation,
                          on_generation=Test().on_generation,
                          on_stop=Test().on_stop,
                          fitness_func=Test().fitness_func)
       
   ga_instance.run()
