.. _header-n0:

``pygad`` Module
================

This section of the PyGAD's library documentation discusses the
**pygad** module.

Using the ``pygad`` module, instances of the genetic algorithm can be
created, run, saved, and loaded.

.. _header-n4:

``pygad.GA`` Class
==================

The first module available in PyGAD is named ``pygad`` and contains a
class named ``GA`` for building the genetic algorithm. The constructor,
methods, function, and attributes within the class are discussed in this
section.

.. _header-n6:

``__init__()``
--------------

For creating an instance of the ``pygad.GA`` class, the constructor
accepts several parameters that allow the user to customize the genetic
algorithm.

The ``pygad.GA`` class constructor supported the following parameters:

-  ``num_generations``: Number of generations.

-  ``num_parents_mating``: Number of solutions to be selected as
   parents.

-  ``fitness_func``: Accepts a function that must accept 2 parameters (a
   single solution and its index in the population) and return the
   fitness value of the solution. Available starting from
   `PyGAD <https://pypi.org/project/pygad>`__ 1.0.17 until 1.0.20 with a
   single parameter representing the solution. Changed in
   `PyGAD <https://pypi.org/project/pygad>`__ 2.0.0 and higher to
   include a second parameter representing the solution index. Check the
   **Preparing the ``fitness_func`` Parameter** section for information
   about creating such a function.

-  ``initial_population``: A user-defined initial population. It is
   useful when the user wants to start the generations with a custom
   initial population. It defaults to ``None`` which means no initial
   population is specified by the user. In this case,
   `PyGAD <https://pypi.org/project/pygad>`__ creates an initial
   population using the ``sol_per_pop`` and ``num_genes`` parameters. An
   exception is raised if the ``initial_population`` is ``None`` while
   any of the 2 parameters (``sol_per_pop`` or ``num_genes``) is also
   ``None``. Introduced in PyGAD 2.0.0 and higher.

-  ``sol_per_pop``: Number of solutions (i.e. chromosomes) within the
   population. This parameter is not needed if the user feeds the
   initial population to the ``initial_population`` parameter.

-  ``num_genes``: Number of genes in the solution/chromosome. This
   parameter is not needed if the user feeds the initial population to
   the ``initial_population`` parameter.

-  ``init_range_low=-4``: The lower value of the random range from which
   the gene values in the initial population are selected.
   ``init_range_low`` defaults to ``-4``. Available in
   `PyGAD <https://pypi.org/project/pygad>`__ 1.0.20 and higher. This
   parameter is not needed if the user feeds the initial population to
   the ``initial_population`` parameter.

-  ``init_range_high=4``: The upper value of the random range from which
   the gene values in the initial population are selected.
   ``init_range_high`` defaults to ``+4``. Available in
   `PyGAD <https://pypi.org/project/pygad>`__ 1.0.20 and higher. This
   parameter is not needed if the user feeds the initial population to
   the ``initial_population`` parameter.

-  ``parent_selection_type="sss"``: The parent selection type. Supported
   types are ``sss`` (for steady-state selection), ``rws`` (for roulette
   wheel selection), ``sus`` (for stochastic universal selection),
   ``rank`` (for rank selection), ``random`` (for random selection), and
   ``tournament`` (for tournament selection).

-  ``keep_parents=-1``: Number of parents to keep in the current
   population. ``-1`` (default) means to keep all parents in the next
   population. ``0`` means keep no parents in the next population. A
   value ``greater than 0`` means keeps the specified number of parents
   in the next population. Note that the value assigned to
   ``keep_parents`` cannot be ``< - 1`` or greater than the number of
   solutions within the population ``sol_per_pop``.

-  ``K_tournament=3``: In case that the parent selection type is
   ``tournament``, the ``K_tournament`` specifies the number of parents
   participating in the tournament selection. It defaults to ``3``.

-  ``crossover_type="single_point"``: Type of the crossover operation.
   Supported types are ``single_point`` (for single-point crossover),
   ``two_points`` (for two points crossover), and ``uniform`` (for
   uniform crossover). It defaults to ``single_point``.

-  ``mutation_type="random"``: Type of the mutation operation. Supported
   types are ``random`` (for random mutation), ``swap`` (for swap
   mutation), ``inversion`` (for inversion mutation), and ``scramble``
   (for scramble mutation). It defaults to ``random``.

-  ``mutation_percent_genes=10``: Percentage of genes to mutate which
   defaults to ``10``. Out of this percentage, the number of genes to
   mutate is deduced. This parameter has no action if the parameter
   ``mutation_num_genes`` exists.

-  ``mutation_num_genes=None``: Number of genes to mutate which defaults
   to ``None`` meaning that no number is specified. If the parameter
   ``mutation_num_genes`` exists, then no need for the parameter
   ``mutation_percent_genes``.

-  ``random_mutation_min_val=-1.0``: For ``random`` mutation, the
   ``random_mutation_min_val`` parameter specifies the start value of
   the range from which a random value is selected to be added to the
   gene. It defaults to ``-1``.

-  ``random_mutation_max_val=1.0``: For ``random`` mutation, the
   ``random_mutation_max_val`` parameter specifies the end value of the
   range from which a random value is selected to be added to the gene.
   It defaults to ``+1``.

-  ``callback_generation``: If not ``None``, then it accepts a function
   to be called after each generation. This function must accept a
   **single parameter** representing the instance of the genetic
   algorithm.

The user doesn't have to specify all of such parameters while creating
an instance of the GA class. A very important parameter you must care
about is ``fitness_func`` which defines the fitness function.

It is OK to set the value of any of the 2 parameters ``init_range_low``
and ``init_range_high`` to be equal, higher, or lower than the other
parameter (i.e. ``init_range_low`` is not needed to be lower than
``init_range_high``).

The parameters are validated within the constructor. If at least a
parameter is not validated, an exception is thrown.

.. _header-n49:

Other Instance Attributes & Methods
-----------------------------------

All the parameters and functions passed to the **pygad.GA** class
constructor are used as attributes and methods in the instances of the
**pygad.GA** class. In addition to such attributes, there are other
attributes and methods added to the instances of the **pygad.GA** class:

The next 2 subsections list such attributes and methods.

.. _header-n52:

Other Attributes
~~~~~~~~~~~~~~~~

-  ``generations_completed``: Holds the number of the last completed
   generation.

-  ``population``: A NumPy array holding the initial population.

-  ``valid_parameters``: Set to ``True`` when all the parameters passed
   in the ``GA`` class constructor are valid.

-  ``run_completed``: Set to ``True`` only after the ``run()`` method
   completes gracefully.

-  ``pop_size``: The population size.

-  ``best_solutions_fitness``: A list holding the fitness values of the
   best solutions for all generations.

-  ``best_solution_generation``: The generation number at which the best
   fitness value is reached. It is only assigned the generation number
   after the ``run()`` method completes. Otherwise, its value is -1.

.. _header-n68:

Other Methods
~~~~~~~~~~~~~

-  ``cal_pop_fitness``: A method that calculates the fitness values for
   all solutions within the population by calling the function passed to
   the ``fitness_func`` parameter for each solution.

-  ``crossover``: Refers to the method that applies the crossover
   operator based on the selected type of crossover in the
   ``crossover_type`` property.

-  ``mutation``: Refers to the method that applies the mutation operator
   based on the selected type of mutation in the ``mutation_type``
   property.

-  ``select_parents``: Refers to a method that selects the parents based
   on the parent selection type specified in the
   ``parent_selection_type`` attribute.

The next sections discuss the methods available in the **pygad.GA**
class.

.. _header-n79:

``initialize_population()``
---------------------------

It creates an initial population randomly as a NumPy array. The array is
saved in the instance attribute named ``population``.

Accepts the following parameters:

-  ``low``: The lower value of the random range from which the gene
   values in the initial population are selected. It defaults to -4.
   Available in PyGAD 1.0.20 and higher.

-  ``high``: The upper value of the random range from which the gene
   values in the initial population are selected. It defaults to -4.
   Available in PyGAD 1.0.20.

This method assigns the values of the following 3 instance attributes:

1. ``pop_size``: Size of the population.

2. ``population``: Initially, it holds the initial population and later
   updated after each generation.

3. ``initial_population``: Keeping the initial population.

.. _header-n95:

``cal_pop_fitness()``
---------------------

Calculating the fitness values of all solutions in the current
population.

It works by iterating through the solutions and calling the function
assigned to the ``fitness_func`` parameter in the **pygad.GA** class
constructor for each solution.

It returns an array of the solutions' fitness values.

.. _header-n99:

``run()``
---------

Runs the genetic algorithm. This is the main method in which the genetic
algorithm is evolved through some generations. It accepts no parameters
as it uses the instance to access all of its requirements.

For each generation, the fitness values of all solutions within the
population are calculated according to the ``cal_pop_fitness()`` method
which internally just calls the function assigned to the
``fitness_func`` parameter in the **pygad.GA** class constructor for
each solution.

According to the fitness values of all solutions, the parents are
selected using the ``select_parents()`` method. This method behavior is
determined according to the parent selection type in the
``parent_selection_type`` parameter in the **pygad.GA** class
constructor

Based on the selected parents, offspring are generated by applying the
crossover and mutation operations using the ``crossover()`` and
``mutation()`` methods. The behavior of such 2 methods is defined
according to the ``crossover_type`` and ``mutation_type`` parameters in
the **pygad.GA** class constructor.

After the generation completes, the following takes place:

-  The ``population`` attribute is updated by the new population.

-  The ``generations_completed`` attribute is assigned by the number of
   the last completed generation.

-  If there is a callback function assigned to the
   ``callback_generation`` attribute, then it will be called.

After the ``run()`` method completes, the following takes place:

-  The ``best_solution_generation`` is assigned the generation number at
   which the best fitness value is reached.

-  The ``run_completed`` attribute is set to ``True``.

.. _header-n118:

Parent Selection Methods
------------------------

The **pygad.GA** class has several methods for selecting the parents
that will mate to produce the offspring. All of such methods accept the
same parameters which are:

-  ``fitness``: The fitness values of the solutions in the current
   population.

-  ``num_parents``: The number of parents to be selected.

All of such methods return an array of the selected parents.

The next subsections list the supported methods for parent selection.

.. _header-n127:

``steady_state_selection()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Selects the parents using the steady-state selection technique.

.. _header-n129:

``rank_selection()``
~~~~~~~~~~~~~~~~~~~~

Selects the parents using the rank selection technique.

.. _header-n131:

``random_selection()``
~~~~~~~~~~~~~~~~~~~~~~

Selects the parents randomly.

.. _header-n133:

``tournament_selection()``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Selects the parents using the tournament selection technique.

.. _header-n135:

``roulette_wheel_selection()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Selects the parents using the roulette wheel selection technique.

.. _header-n137:

``stochastic_universal_selection()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Selects the parents using the stochastic universal selection technique.

.. _header-n139:

Crossover Methods
-----------------

The **pygad.GA** class supports several methods for applying crossover
between the selected parents. All of these methods accept the same
parameters which are:

-  ``parents``: The parents to mate for producing the offspring.

-  ``offspring_size``: The size of the offspring to produce.

All of such methods return an array of the produced offspring.

The next subsections list the supported methods for crossover.

.. _header-n148:

``single_point_crossover()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Applies the single-point crossover. It selects a point randomly at which
crossover takes place between the pairs of parents.

.. _header-n150:

``two_points_crossover()``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Applies the 2 points crossover. It selects the 2 points randomly at
which crossover takes place between the pairs of parents.

.. _header-n152:

``uniform_crossover()``
~~~~~~~~~~~~~~~~~~~~~~~

Applies the uniform crossover. For each gene, a parent out of the 2
mating parents is selected randomly and the gene is copied from it.

.. _header-n154:

Mutation Methods
----------------

The **pygad.GA** class supports several methods for applying mutation.
All of these methods accept the same parameter which is:

-  ``offspring``: The offspring to mutate.

All of such methods return an array of the mutated offspring.

The next subsections list the supported methods for mutation.

.. _header-n161:

``random_mutation()``
~~~~~~~~~~~~~~~~~~~~~

Applies the random mutation which changes the values of some genes
randomly. The number of genes is specified according to either the
``mutation_num_genes`` or the ``mutation_percent_genes`` attributes.

For each gene, a random value is selected according to the range
specified by the 2 attributes ``random_mutation_min_val`` and
``random_mutation_max_val``. The random value is added to the selected
gene.

.. _header-n164:

``swap_mutation()``
~~~~~~~~~~~~~~~~~~~

Applies the swap mutation which interchanges the values of 2 randomly
selected genes.

.. _header-n166:

``inversion_mutation()``
~~~~~~~~~~~~~~~~~~~~~~~~

Applies the inversion mutation which selects a subset of genes and
inverts them.

.. _header-n168:

``scramble_mutation()``
~~~~~~~~~~~~~~~~~~~~~~~

Applies the scramble mutation which selects a subset of genes and
shuffles their order randomly.

.. _header-n170:

``best_solution()``
-------------------

Returns information about the best solution found by the genetic
algorithm. It can only be called after completing at least 1 generation.

If no generation is completed, an exception is raised. Otherwise, the
following is returned:

-  ``best_solution``: Best solution in the current population.

-  ``best_solution_fitness``: Fitness value of the best solution.

-  ``best_match_idx``: Index of the best solution in the current
   population.

.. _header-n180:

``plot_result()``
-----------------

Creates and shows a plot that summarizes how the fitness value evolved
by generation. It can only be called after completing at least 1
generation.

If no generation is completed (at least 1), an exception is raised.

.. _header-n183:

``save()``
----------

Saves the genetic algorithm instance

Accepts the following parameter:

-  ``filename``: Name of the file to save the instance. No extension is
   needed.

.. _header-n189:

Functions in ``pygad``
======================

Besides the methods available in the **pygad.GA** class, this section
discusses the functions available in pygad. Up to this time, there is
only a single function named ``load()``.

.. _header-n191:

``pygad.load()``
----------------

Reads a saved instance of the genetic algorithm. This is **not a
method** but a **function** that is indented under the ``pygad`` module.
So, it could be called by the **pygad** module as follows:
``pygad.load(filename)``.

Accepts the following parameter:

-  ``filename``: Name of the file holding the saved instance of the
   genetic algorithm. No extension is needed.

Returns the genetic algorithm instance.

.. _header-n198:

Steps to Use ``pygad``
======================

To use the ``pygad`` module, here is a summary of the required steps:

1. Preparing the ``fitness_func`` parameter.

2. Preparing Other Parameters.

3. Import pygad.

4. Create an Instance of the **pygad.GA** Class.

5. Run the Genetic Algorithm.

6. Plotting Results.

7. Information about the Best Solution.

8. Saving & Loading the Results.

Let's discuss how to do each of these steps.

.. _header-n218:

Preparing the ``fitness_func`` Parameter 
-----------------------------------------

Even there are some steps in the genetic algorithm pipeline that can
work the same regardless of the problem being solved, one critical step
is the calculation of the fitness value. There is no unique way of
calculating the fitness value and it changes from one problem to
another.

On **``15 April 2020``**, a new argument named ``fitness_func`` is added
to PyGAD 1.0.17 that allows the user to specify a custom function to be
used as a fitness function. This function must be a **maximization
function** so that a solution with a high fitness value returned is
selected compared to a solution with a low value. Doing that allows the
user to freely use PyGAD to solve any problem by passing the appropriate
fitness function. It is very important to understand the problem well
for creating this function.

Let's discuss an example:

   | Given the following function:
   |  y = f(w1:w6) = w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + 6wx6
   |  where (x1,x2,x3,x4,x5,x6)=(4,-2,3.5,5,-11,-4.7) and y=44
   | What are the best values for the 6 weights (w1 to w6)? We are going
     to use the genetic algorithm to optimize this function.

So, the task is about using the genetic algorithm to find the best
values for the 6 weight ``W1`` to ``W6``. Thinking of the problem, it is
clear that the best solution is that returning an output that is close
to the desired output ``y=44``. So, the fitness function should return a
value that gets higher when the solution's output is closer to ``y=44``.
Here is a function that does that:

.. code:: python

   function_inputs = [4,-2,3.5,5,-11,-4.7] # Function inputs.
   desired_output = 44 # Function output.

   def fitness_func(solution, solution_idx):
       output = numpy.sum(solution*function_inputs)
       fitness = 1.0 / numpy.abs(output - desired_output)
       return fitness

Such a user-defined function must accept 2 parameters:

1. 1D vector representing a single solution. Introduced in PyGAD 1.0.17
   and higher.

2. Solution index within the population. Introduced in PyGAD 2.0.0 and
   higher.

The ``__code__`` object is used to check if this function accepts the
required number of parameters. If more or fewer parameters are passed,
an exception is thrown.

By creating this function, you almost did an awesome step towards using
PyGAD.

.. _header-n234:

Preparing Other Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

Here is an example for preparing the other parameters:

.. code:: python

   num_generations = 50
   num_parents_mating = 4

   fitness_function = fitness_func

   sol_per_pop = 8
   num_genes = len(function_inputs)

   init_range_low = -2
   init_range_high = 5

   parent_selection_type = "sss"
   keep_parents = 1

   crossover_type = "single_point"

   mutation_type = "random"
   mutation_percent_genes = 10

.. _header-n237:

The ``callback_generation`` Parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In PyGAD 2.0.0 and higher, an optional parameter named
``callback_generation`` is supported which allows the user to call a
function (with a single parameter) after each generation. Here is a
simple function that just prints the current generation number and the
fitness value of the best solution in the current generation. The
``generations_completed`` attribute of the GA class returns the number
of the last completed generation.

.. code:: python

   def callback_gen(ga_instance):
       print("Generation : ", ga_instance.generations_completed)
       print("Fitness of the best solution :", ga_instance.best_solution()[1])

After being defined, the function is assigned to the
``callback_generation`` parameter of the GA class constructor. By doing
that, the ``callback_gen()`` function will be called after each
generation.

.. code:: python

   ga_instance = pygad.GA(..., 
                          callback_generation=callback_gen,
                          ...)

After the parameters are prepared, we can import PyGAD and build an
instance of the **pygad.GA** class.

.. _header-n243:

Import the ``pygad``
--------------------

The next step is to import PyGAD as follows:

.. code:: python

   import pygad

The **pygad.GA** class holds the implementation of all methods for
running the genetic algorithm.

.. _header-n247:

Create an Instance of the ``pygad.GA`` Class
--------------------------------------------

The **pygad.GA** class is instantiated where the previously prepared
parameters are fed to its constructor. The constructor is responsible
for creating the initial population.

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

.. _header-n250:

Run the Genetic Algorithm
-------------------------

After an instance of the **pygad.GA** class is created, the next step is
to call the ``run()`` method as follows:

.. code:: python

   ga_instance.run()

Inside this method, the genetic algorithm evolves over some generations
by doing the following tasks:

1. Calculating the fitness values of the solutions within the current
   population.

2. Select the best solutions as parents in the mating pool.

3. Apply the crossover & mutation operation

4. Repeat the process for the specified number of generations.

.. _header-n263:

Plotting Results
----------------

There is a method named ``plot_result()`` which creates a figure
summarizing how the fitness values of the solutions change with the
generations.

.. code:: python

   ga_instance.plot_result()

.. figure:: https://user-images.githubusercontent.com/16560492/78830005-93111d00-79e7-11ea-9d8e-a8d8325a6101.png
   :alt: 

.. _header-n267:

Information about the Best Solution
-----------------------------------

The following information about the best solution in the last population
is returned using the ``best_solution()`` method.

-  Solution

-  Fitness value of the solution

-  Index of the solution within the population

.. code:: python

   solution, solution_fitness, solution_idx = ga_instance.best_solution()
   print("Parameters of the best solution : {solution}".format(solution=solution))
   print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
   print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

Using the ``best_solution_generation`` attribute of the instance from
the **pygad.GA** class, the generation number at which the **best
fitness** is reached could be fetched.

.. code:: python

   if ga_instance.best_solution_generation != -1:
       print("Best fitness value reached after {best_solution_generation} generations.".format(best_solution_generation=ga_instance.best_solution_generation))

.. _header-n279:

Saving & Loading the Results
----------------------------

After the ``run()`` method completes, it is possible to save the current
instance of the genetic algorithm to avoid losing the progress made. The
``save()`` method is available for that purpose. Just pass the file name
to it without an extension. According to the next code, a file named
``genetic.pkl`` will be created and saved in the current directory.

.. code:: python

   filename = 'genetic'
   ga_instance.save(filename=filename)

You can also load the saved model using the ``load()`` function and
continue using it. For example, you might run the genetic algorithm for
some generations, save its current state using the ``save()`` method,
load the model using the ``load()`` function, and then call the
``run()`` method again.

.. code:: python

   loaded_ga_instance = pygad.load(filename=filename)

After the instance is loaded, you can use it to run any method or access
any property.

.. code:: python

   print(loaded_ga_instance.best_solution())

.. _header-n286:

Crossover, Mutation, and Parent Selection
=========================================

PyGAD supports different types for selecting the parents and applying
the crossover & mutation operators. More features will be added in the
future. To ask for a new feature, please check the **Ask for Feature**
section.

.. _header-n288:

Supported Crossover Operations
------------------------------

The supported crossover operations at this time are:

1. Single point: Implemented using the ``single_point_crossover()``
   method.

2. Two points: Implemented using the ``two_points_crossover()`` method.

3. Uniform: Implemented using the ``uniform_crossover()`` method.

.. _header-n297:

Supported Mutation Operations
-----------------------------

The supported mutation operations at this time are:

1. Random: Implemented using the ``random_mutation()`` method.

2. Swap: Implemented using the ``swap_mutation()`` method.

3. Inversion: Implemented using the ``inversion_mutation()`` method.

4. Scramble: Implemented using the ``scramble_mutation()`` method.

.. _header-n308:

Supported Parent Selection Operations
-------------------------------------

The supported parent selection techniques at this time are:

1. Steady-state: Implemented using the ``steady_state_selection()``
   method.

2. Roulette wheel: Implemented using the ``roulette_wheel_selection()``
   method.

3. Stochastic universal: Implemented using the
   ``stochastic_universal_selection()``\ method.

4. Rank: Implemented using the ``rank_selection()`` method.

5. Random: Implemented using the ``random_selection()`` method.

6. Tournament: Implemented using the ``tournament_selection()`` method.

.. _header-n323:

Examples
========

This section gives the complete code of some examples that use
``pygad``. Each subsection builds a different example.

.. _header-n325:

Linear Model Optimization
-------------------------

This example is discussed in the **Steps to Use ``pygad``** section
which optimizes a linear model. Its complete code is listed below.

.. code:: python

   import pygad
   import numpy

   """
   Given the following function:
       y = f(w1:w6) = w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + 6wx6
       where (x1,x2,x3,x4,x5,x6)=(4,-2,3.5,5,-11,-4.7) and y=44
   What are the best values for the 6 weights (w1 to w6)? We are going to use the genetic algorithm to optimize this function.
   """

   function_inputs = [4,-2,3.5,5,-11,-4.7] # Function inputs.
   desired_output = 44 # Function output.

   def fitness_func(solution, solution_idx):
       # Calculating the fitness value of each solution in the current population.
       # The fitness function calulates the sum of products between each input and its corresponding weight.
       output = numpy.sum(solution*function_inputs)
       fitness = 1.0 / numpy.abs(output - desired_output)
       return fitness

   fitness_function = fitness_func

   num_generations = 50 # Number of generations.
   num_parents_mating = 4 # Number of solutions to be selected as parents in the mating pool.

   # To prepare the initial population, there are 2 ways:
   # 1) Prepare it yourself and pass it to the initial_population parameter. This way is useful when the user wants to start the genetic algorithm with a custom initial population.
   # 2) Assign valid integer values to the sol_per_pop and num_genes parameters. If the initial_population parameter exists, then the sol_per_pop and num_genes parameters are useless.
   sol_per_pop = 8 # Number of solutions in the population.
   num_genes = len(function_inputs)

   init_range_low = -2
   init_range_high = 5

   parent_selection_type = "sss" # Type of parent selection.
   keep_parents = 1 # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.

   crossover_type = "single_point" # Type of the crossover operator.

   # Parameters of the mutation operation.
   mutation_type = "random" # Type of the mutation operator.
   mutation_percent_genes = 10 # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists.

   last_fitness = 0
   def callback_generation(ga_instance):
       global last_fitness
       print("Generation = {generation}".format(generation=ga_instance.generations_completed))
       print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
       print("Change     = {change}".format(change=ga_instance.best_solution()[1] - last_fitness))

   # Creating an instance of the GA class inside the ga module. Some parameters are initialized within the constructor.
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
                          mutation_percent_genes=mutation_percent_genes,
                          callback_generation=callback_generation)

   # Running the GA to optimize the parameters of the function.
   ga_instance.run()

   # After the generations complete, some plots are showed that summarize the how the outputs/fitenss values evolve over generations.
   ga_instance.plot_result()

   # Returning the details of the best solution.
   solution, solution_fitness, solution_idx = ga_instance.best_solution()
   print("Parameters of the best solution : {solution}".format(solution=solution))
   print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
   print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

   prediction = numpy.sum(numpy.array(function_inputs)*solution)
   print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

   if ga_instance.best_solution_generation != -1:
       print("Best fitness value reached after {best_solution_generation} generations.".format(best_solution_generation=ga_instance.best_solution_generation))

   # Saving the GA instance.
   filename = 'genetic' # The filename to which the instance is saved. The name is without extension.
   ga_instance.save(filename=filename)

   # Loading the saved GA instance.
   loaded_ga_instance = pygad.load(filename=filename)
   loaded_ga_instance.plot_result()
