.. _header-n513:

``pygad`` Module
================

This section of the PyGAD's library documentation discusses the
**pygad** module.

Using the ``pygad`` module, instances of the genetic algorithm can be
created, run, saved, and loaded.

.. _header-n517:

``pygad.GA`` Class
==================

The first module available in PyGAD is named ``pygad`` and contains a
class named ``GA`` for building the genetic algorithm. The constructor,
methods, function, and attributes within the class are discussed in this
section.

.. _header-n519:

``__init__()``
--------------

For creating an instance of the ``pygad.GA`` class, the constructor
accepts several parameters that allow the user to customize the genetic
algorithm to different types of applications.

The ``pygad.GA`` class constructor supports the following parameters:

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
   population. This parameter has no action if ``initial_population``
   parameter exists.

-  ``num_genes``: Number of genes in the solution/chromosome. This
   parameter is not needed if the user feeds the initial population to
   the ``initial_population`` parameter.

-  ``gene_type=float``: Controls the gene type. It has an effect only
   when the parameter ``gene_space`` is ``None`` (which is its default
   value). Starting from PyGAD 2.9.0, the ``gene_type`` parameter can be
   assigned to a numeric value of any of these types: ``int``,
   ``float``, and ``numpy.int/uint/float(8-64)``.

-  ``init_range_low=-4``: The lower value of the random range from which
   the gene values in the initial population are selected.
   ``init_range_low`` defaults to ``-4``. Available in
   `PyGAD <https://pypi.org/project/pygad>`__ 1.0.20 and higher. This
   parameter has no action if the ``initial_population`` parameter
   exists.

-  ``init_range_high=4``: The upper value of the random range from which
   the gene values in the initial population are selected.
   ``init_range_high`` defaults to ``+4``. Available in
   `PyGAD <https://pypi.org/project/pygad>`__ 1.0.20 and higher. This
   parameter has no action if the ``initial_population`` parameter
   exists.

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
   ``two_points`` (for two points crossover), ``uniform`` (for uniform
   crossover), and ``scattered`` (for scattered crossover). Scattered
   crossover is supported from PyGAD 2.9.0 and higher. It defaults to
   ``single_point``. Starting from PyGAD 2.2.2 and higher, if
   ``crossover_type=None``, then the crossover step is bypassed which
   means no crossover is applied and thus no offspring will be created
   in the next generations. The next generation will use the solutions
   in the current population.

-  ``crossover_probability=None``: The probability of selecting a parent
   for applying the crossover operation. Its value must be between 0.0
   and 1.0 inclusive. For each parent, a random value between 0.0 and
   1.0 is generated. If this random value is less than or equal to the
   value assigned to the ``crossover_probability`` parameter, then the
   parent is selected. Added in PyGAD 2.5.0 and higher.

-  ``mutation_type="random"``: Type of the mutation operation. Supported
   types are ``random`` (for random mutation), ``swap`` (for swap
   mutation), ``inversion`` (for inversion mutation), ``scramble`` (for
   scramble mutation), and ``adaptive`` (for adaptive mutation). It
   defaults to ``random``. Starting from PyGAD 2.2.2 and higher, if
   ``mutation_type=None``, then the mutation step is bypassed which
   means no mutation is applied and thus no changes are applied to the
   offspring created using the crossover operation. The offspring will
   be used unchanged in the next generation. ``Adaptive`` mutation is
   supported starting from PyGAD 2.10.0. For more information about
   adaptive mutation, go the the `Adaptive
   Mutation <https://pygad.readthedocs.io/en/latest/README_pygad_torchga_ReadTheDocs.html#adaptive-mutation>`__
   section. For example about using adaptive mutation, check the `Use
   Adaptive Mutation in
   PyGAD <https://pygad.readthedocs.io/en/latest/README_pygad_torchga_ReadTheDocs.html#use-adaptive-mutation-in-pygad>`__
   section.

-  ``mutation_probability=None``: The probability of selecting a gene
   for applying the mutation operation. Its value must be between 0.0
   and 1.0 inclusive. For each gene in a solution, a random value
   between 0.0 and 1.0 is generated. If this random value is less than
   or equal to the value assigned to the ``mutation_probability``
   parameter, then the gene is selected. If this parameter exists, then
   there is no need for the 2 parameters ``mutation_percent_genes`` and
   ``mutation_num_genes``. Added in PyGAD 2.5.0 and higher.

-  ``mutation_by_replacement=False``: An optional bool parameter. It
   works only when the selected type of mutation is random
   (``mutation_type="random"``). In this case,
   ``mutation_by_replacement=True`` means replace the gene by the
   randomly generated value. If False, then it has no effect and random
   mutation works by adding the random value to the gene. Supported in
   PyGAD 2.2.2 and higher. Check the changes in PyGAD 2.2.2 under the
   Release History section for an example.

-  ``mutation_percent_genes="default"``: Percentage of genes to mutate.
   It defaults to the string ``"default"`` which is later translated
   into the integer ``10`` which means 10% of the genes will be mutated.
   It must be ``>0`` and ``<=100``. Out of this percentage, the number
   of genes to mutate is deduced which is assigned to the
   ``mutation_num_genes`` parameter. The ``mutation_percent_genes``
   parameter has no action if ``mutation_probability`` or
   ``mutation_num_genes`` exist. Starting from PyGAD 2.2.2 and higher,
   this parameter has no action if ``mutation_type`` is ``None``.

-  ``mutation_num_genes=None``: Number of genes to mutate which defaults
   to ``None`` meaning that no number is specified. The
   ``mutation_num_genes`` parameter has no action if the parameter
   ``mutation_probability`` exists. Starting from PyGAD 2.2.2 and
   higher, this parameter has no action if ``mutation_type`` is
   ``None``.

-  ``random_mutation_min_val=-1.0``: For ``random`` mutation, the
   ``random_mutation_min_val`` parameter specifies the start value of
   the range from which a random value is selected to be added to the
   gene. It defaults to ``-1``. Starting from PyGAD 2.2.2 and higher,
   this parameter has no action if ``mutation_type`` is ``None``.

-  ``random_mutation_max_val=1.0``: For ``random`` mutation, the
   ``random_mutation_max_val`` parameter specifies the end value of the
   range from which a random value is selected to be added to the gene.
   It defaults to ``+1``. Starting from PyGAD 2.2.2 and higher, this
   parameter has no action if ``mutation_type`` is ``None``.

-  ``gene_space=None``: It is used to specify the possible values for
   each gene in case the user wants to restrict the gene values. It is
   useful if the gene space is restricted to a certain range or to
   discrete values. It accepts a ``list``, ``tuple``, ``range``, or
   ``numpy.ndarray``. When all genes have the same global space, specify
   their values as a list/tuple/range/numpy.ndarray. For example,
   ``gene_space = [0.3, 5.2, -4, 8]`` restricts the gene values to the 4
   specified values. If each gene has its own space, then the
   ``gene_space`` parameter can be nested like
   ``[[0.4, -5], [0.5, -3.2, 8.2, -9], ...]`` where the first sublist
   determines the values for the first gene, the second sublist for the
   second gene, and so on. If the nested list/tuple has a ``None``
   value, then the gene's initial value is selected randomly from the
   range specified by the 2 parameters ``init_range_low`` and
   ``init_range_high`` and its mutation value is selected randomly from
   the range specified by the 2 parameters ``random_mutation_min_val``
   and ``random_mutation_max_val``. ``gene_space`` is added in PyGAD
   2.5.0. Check the **Release History** section of the documentation for
   more details. In PyGAD 2.9.0, NumPy arrays can be assigned to the
   ``gene_space`` parameter.

-  ``on_start=None``: Accepts a function to be called only once before
   the genetic algorithm starts its evolution. This function must accept
   a single parameter representing the instance of the genetic
   algorithm. Added in PyGAD 2.6.0.

-  ``on_fitness=None``: Accepts a function to be called after
   calculating the fitness values of all solutions in the population.
   This function must accept 2 parameters: the first one represents the
   instance of the genetic algorithm and the second one is a list of all
   solutions' fitness values. Added in PyGAD 2.6.0.

-  ``on_parents=None``: Accepts a function to be called after selecting
   the parents that mates. This function must accept 2 parameters: the
   first one represents the instance of the genetic algorithm and the
   second one represents the selected parents. Added in PyGAD 2.6.0.

-  ``on_crossover=None``: Accepts a function to be called each time the
   crossover operation is applied. This function must accept 2
   parameters: the first one represents the instance of the genetic
   algorithm and the second one represents the offspring generated using
   crossover. Added in PyGAD 2.6.0.

-  ``on_mutation=None``: Accepts a function to be called each time the
   mutation operation is applied. This function must accept 2
   parameters: the first one represents the instance of the genetic
   algorithm and the second one represents the offspring after applying
   the mutation. Added in PyGAD 2.6.0.

-  ``callback_generation=None``: Accepts a function to be called after
   each generation. This function must accept a single parameter
   representing the instance of the genetic algorithm. Supported in
   PyGAD 2.0.0 and higher. In PyGAD 2.4.0, if this function returned the
   string ``stop``, then the ``run()`` method stops at the current
   generation without completing the remaining generations. Check the
   **Release History** section of the documentation for an example.
   Starting from PyGAD 2.6.0, the ``callback_generation`` parameter is
   deprecated and should be replaced by the ``on_generation`` parameter.
   The ``callback_generation`` parameter will be removed in a later
   version.

-  ``on_generation=None``: Accepts a function to be called after each
   generation. This function must accept a single parameter representing
   the instance of the genetic algorithm. If the function returned the
   string ``stop``, then the ``run()`` method stops without completing
   the other generations. Added in PyGAD 2.6.0.

-  ``on_stop=None``: Accepts a function to be called only once exactly
   before the genetic algorithm stops or when it completes all the
   generations. This function must accept 2 parameters: the first one
   represents the instance of the genetic algorithm and the second one
   is a list of fitness values of the last population's solutions. Added
   in PyGAD 2.6.0.

-  ``delay_after_gen=0.0``: It accepts a non-negative number specifying
   the time in seconds to wait after a generation completes and before
   going to the next generation. It defaults to ``0.0`` which means no
   delay after the generation. Available in PyGAD 2.4.0 and higher.

-  ``save_best_solutions=False``: When ``True``, then the best solution
   after each generation is saved into an attribute named
   ``best_solutions``. If ``False`` (default), then no solutions are
   saved and the ``best_solutions`` attribute will be empty. Supported
   in PyGAD 2.9.0.

-  ``suppress_warnings=False``: A bool parameter to control whether the
   warning messages are printed or not. It defaults to ``False``.

The user doesn't have to specify all of such parameters while creating
an instance of the GA class. A very important parameter you must care
about is ``fitness_func`` which defines the fitness function.

It is OK to set the value of any of the 2 parameters ``init_range_low``
and ``init_range_high`` to be equal, higher, or lower than the other
parameter (i.e. ``init_range_low`` is not needed to be lower than
``init_range_high``). The same holds for the ``random_mutation_min_val``
and ``random_mutation_max_val`` parameters.

If the 2 parameters ``mutation_type`` and ``crossover_type`` are
``None``, this disables any type of evolution the genetic algorithm can
make. As a result, the genetic algorithm cannot find a better solution
that the best solution in the initial population.

The parameters are validated within the constructor. If at least a
parameter is not validated, an exception is thrown.

.. _header-n593:

Other Instance Attributes & Methods
-----------------------------------

All the parameters and functions passed to the **pygad.GA** class
constructor are used as attributes and methods in the instances of the
**pygad.GA** class. In addition to such attributes, there are other
attributes and methods added to the instances of the **pygad.GA** class:

The next 2 subsections list such attributes and methods.

.. _header-n596:

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

-  ``best_solutions``: A NumPy array holding the best solution per each
   generation. It only exists when the ``save_best_solutions`` parameter
   in the ``pygad.GA`` class constructor is set to ``True``.

.. _header-n614:

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

-  ``adaptive_mutation_population_fitness``: Returns the average fitness
   value used in the adaptive mutation to filter the solutions.

The next sections discuss the methods available in the **pygad.GA**
class.

.. _header-n627:

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

.. _header-n643:

``cal_pop_fitness()``
---------------------

Calculating the fitness values of all solutions in the current
population.

It works by iterating through the solutions and calling the function
assigned to the ``fitness_func`` parameter in the **pygad.GA** class
constructor for each solution.

It returns an array of the solutions' fitness values.

.. _header-n647:

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

.. _header-n666:

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

.. _header-n675:

``steady_state_selection()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Selects the parents using the steady-state selection technique.

.. _header-n677:

``rank_selection()``
~~~~~~~~~~~~~~~~~~~~

Selects the parents using the rank selection technique.

.. _header-n679:

``random_selection()``
~~~~~~~~~~~~~~~~~~~~~~

Selects the parents randomly.

.. _header-n681:

``tournament_selection()``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Selects the parents using the tournament selection technique.

.. _header-n683:

``roulette_wheel_selection()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Selects the parents using the roulette wheel selection technique.

.. _header-n685:

``stochastic_universal_selection()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Selects the parents using the stochastic universal selection technique.

.. _header-n687:

Crossover Methods
-----------------

The **pygad.GA** class supports several methods for applying crossover
between the selected parents. All of these methods accept the same
parameters which are:

-  ``parents``: The parents to mate for producing the offspring.

-  ``offspring_size``: The size of the offspring to produce.

All of such methods return an array of the produced offspring.

The next subsections list the supported methods for crossover.

.. _header-n696:

``single_point_crossover()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Applies the single-point crossover. It selects a point randomly at which
crossover takes place between the pairs of parents.

.. _header-n698:

``two_points_crossover()``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Applies the 2 points crossover. It selects the 2 points randomly at
which crossover takes place between the pairs of parents.

.. _header-n700:

``uniform_crossover()``
~~~~~~~~~~~~~~~~~~~~~~~

Applies the uniform crossover. For each gene, a parent out of the 2
mating parents is selected randomly and the gene is copied from it.

.. _header-n702:

``scattered_crossover()``
~~~~~~~~~~~~~~~~~~~~~~~~~

Applies the scattered crossover. It randomly selects the gene from one
of the 2 parents.

.. _header-n704:

Mutation Methods
----------------

The **pygad.GA** class supports several methods for applying mutation.
All of these methods accept the same parameter which is:

-  ``offspring``: The offspring to mutate.

All of such methods return an array of the mutated offspring.

The next subsections list the supported methods for mutation.

.. _header-n711:

``random_mutation()``
~~~~~~~~~~~~~~~~~~~~~

Applies the random mutation which changes the values of some genes
randomly. The number of genes is specified according to either the
``mutation_num_genes`` or the ``mutation_percent_genes`` attributes.

For each gene, a random value is selected according to the range
specified by the 2 attributes ``random_mutation_min_val`` and
``random_mutation_max_val``. The random value is added to the selected
gene.

.. _header-n714:

``swap_mutation()``
~~~~~~~~~~~~~~~~~~~

Applies the swap mutation which interchanges the values of 2 randomly
selected genes.

.. _header-n716:

``inversion_mutation()``
~~~~~~~~~~~~~~~~~~~~~~~~

Applies the inversion mutation which selects a subset of genes and
inverts them.

.. _header-n718:

``scramble_mutation()``
~~~~~~~~~~~~~~~~~~~~~~~

Applies the scramble mutation which selects a subset of genes and
shuffles their order randomly.

.. _header-n720:

``adaptive_mutation()``
~~~~~~~~~~~~~~~~~~~~~~~

Applies the adaptive mutation which selects a subset of genes and
shuffles their order randomly.

.. _header-n722:

``best_solution()``
-------------------

Returns information about the best solution found by the genetic
algorithm.

It accepts the following parameters:

-  ``pop_fitness=None``: An optional parameter that accepts a list of
   the fitness values of the solutions in the population. If ``None``,
   then the ``cal_pop_fitness()`` method is called to calculate the
   fitness values of the population.

It returns the following:

-  ``best_solution``: Best solution in the current population.

-  ``best_solution_fitness``: Fitness value of the best solution.

-  ``best_match_idx``: Index of the best solution in the current
   population.

.. _header-n736:

``plot_result()``
-----------------

Creates and shows a plot that summarizes how the fitness value evolved
by generation. It can only be called after completing at least 1
generation.

If no generation is completed (at least 1), an exception is raised.

In PyGAD 2.3.0 and higher, this function accepts 3 optional parameters:

1. ``title``: Title of the figure.

2. ``xlabel``: X-axis label.

3. ``ylabel``: Y-axis label.

Starting from PyGAD 2.5.0, a new optional parameter named ``linewidth``
is added to specify the width of the curve in the plot. It defaults to
``3.0``.

.. _header-n748:

``save()``
----------

Saves the genetic algorithm instance

Accepts the following parameter:

-  ``filename``: Name of the file to save the instance. No extension is
   needed.

.. _header-n754:

Functions in ``pygad``
======================

Besides the methods available in the **pygad.GA** class, this section
discusses the functions available in pygad. Up to this time, there is
only a single function named ``load()``.

.. _header-n756:

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

.. _header-n763:

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

.. _header-n783:

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

.. _header-n799:

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

.. _header-n802:

The ``callback_generation`` Parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This parameter should be replaced by ``on_generation``. The
``callback_generation`` parameter will be removed in a later release of
PyGAD.

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

.. _header-n809:

Import the ``pygad``
--------------------

The next step is to import PyGAD as follows:

.. code:: python

   import pygad

The **pygad.GA** class holds the implementation of all methods for
running the genetic algorithm.

.. _header-n813:

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

.. _header-n816:

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

.. _header-n829:

Plotting Results
----------------

There is a method named ``plot_result()`` which creates a figure
summarizing how the fitness values of the solutions change with the
generations.

.. code:: python

   ga_instance.plot_result()

.. figure:: https://user-images.githubusercontent.com/16560492/78830005-93111d00-79e7-11ea-9d8e-a8d8325a6101.png
   :alt: 

.. _header-n833:

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

.. _header-n845:

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

.. _header-n852:

Crossover, Mutation, and Parent Selection
=========================================

PyGAD supports different types for selecting the parents and applying
the crossover & mutation operators. More features will be added in the
future. To ask for a new feature, please check the **Ask for Feature**
section.

.. _header-n854:

Supported Crossover Operations
------------------------------

The supported crossover operations at this time are:

1. Single point: Implemented using the ``single_point_crossover()``
   method.

2. Two points: Implemented using the ``two_points_crossover()`` method.

3. Uniform: Implemented using the ``uniform_crossover()`` method.

.. _header-n863:

Supported Mutation Operations
-----------------------------

The supported mutation operations at this time are:

1. Random: Implemented using the ``random_mutation()`` method.

2. Swap: Implemented using the ``swap_mutation()`` method.

3. Inversion: Implemented using the ``inversion_mutation()`` method.

4. Scramble: Implemented using the ``scramble_mutation()`` method.

.. _header-n874:

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

.. _header-n889:

Life Cycle of PyGAD
===================

The next figure lists the different stages in the lifecycle of an
instance of the ``pygad.GA`` class. Note that PyGAD stops when either
all generations are completed or when the function passed to the
``on_generation`` parameter returns the string ``stop``.

.. figure:: https://user-images.githubusercontent.com/16560492/89446279-9c6f8380-d754-11ea-83fd-a60ea2f53b85.jpg
   :alt: 

The next code implements all the callback functions to trace the
execution of the genetic algorithm. Each callback function prints its
name.

.. code:: python

   import pygad
   import numpy

   function_inputs = [4,-2,3.5,5,-11,-4.7]
   desired_output = 44

   def fitness_func(solution, solution_idx):
       output = numpy.sum(solution*function_inputs)
       fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
       return fitness

   fitness_function = fitness_func

   def on_start(ga_instance):
       print("on_start()")

   def on_fitness(ga_instance, population_fitness):
       print("on_fitness()")

   def on_parents(ga_instance, selected_parents):
       print("on_parents()")

   def on_crossover(ga_instance, offspring_crossover):
       print("on_crossover()")

   def on_mutation(ga_instance, offspring_mutation):
       print("on_mutation()")

   def on_generation(ga_instance):
       print("on_generation()")

   def on_stop(ga_instance, last_population_fitness):
       print("on_stop()")

   ga_instance = pygad.GA(num_generations=3,
                          num_parents_mating=5,
                          fitness_func=fitness_function,
                          sol_per_pop=10,
                          num_genes=len(function_inputs),
                          on_start=on_start,
                          on_fitness=on_fitness,
                          on_parents=on_parents,
                          on_crossover=on_crossover,
                          on_mutation=on_mutation,
                          on_generation=on_generation,
                          on_stop=on_stop)

   ga_instance.run()

Based on the used 3 generations as assigned to the ``num_generations``
argument, here is the output.

.. code:: 

   on_start()

   on_fitness()
   on_parents()
   on_crossover()
   on_mutation()
   on_generation()

   on_fitness()
   on_parents()
   on_crossover()
   on_mutation()
   on_generation()

   on_fitness()
   on_parents()
   on_crossover()
   on_mutation()
   on_generation()

   on_stop()

.. _header-n896:

Adaptive Mutation
=================

In the regular genetic algorithm, the mutation works by selecting a
single fixed mutation rate for all solutions regardless of their fitness
values. So, regardless on whether this solution has high or low quality,
the same number of genes are mutated all the time.

The pitfalls of using a constant mutation rate for all solutions are
summarized in this paper `Libelli, S. Marsili, and P. Alba. "Adaptive
mutation in genetic algorithms." Soft computing 4.2 (2000):
76-80 <https://idp.springer.com/authorize/casa?redirect_uri=https://link.springer.com/content/pdf/10.1007/s005000000042.pdf&casa_token=IT4NfJUvslcAAAAA:VegHW6tm2fe3e0R9cRKjuGKkKWXJTQSfNMT6z0kGbMsAllyK1NrEY3cEWg8bj7AJWEQPaqWIJxmHNBHg>`__
as follows:

   The weak point of "classical" GAs is the total randomness of
   mutation, which is applied equally to all chromosomes, irrespective
   of their fitness. Thus a very good chromosome is equally likely to be
   disrupted by mutation as a bad one.

   On the other hand, bad chromosomes are less likely to produce good
   ones through crossover, because of their lack of building blocks,
   until they remain unchanged. They would benefit the most from
   mutation and could be used to spread throughout the parameter space
   to increase the search thoroughness. So there are two conflicting
   needs in determining the best probability of mutation.

   Usually, a reasonable compromise in the case of a constant mutation
   is to keep the probability low to avoid disruption of good
   chromosomes, but this would prevent a high mutation rate of
   low-fitness chromosomes. Thus a constant probability of mutation
   would probably miss both goals and result in a slow improvement of
   the population.

According to `Libelli, S. Marsili, and P.
Alba. <https://idp.springer.com/authorize/casa?redirect_uri=https://link.springer.com/content/pdf/10.1007/s005000000042.pdf&casa_token=IT4NfJUvslcAAAAA:VegHW6tm2fe3e0R9cRKjuGKkKWXJTQSfNMT6z0kGbMsAllyK1NrEY3cEWg8bj7AJWEQPaqWIJxmHNBHg>`__
work, the adaptive mutation solves the problems of constant mutation.

Adaptive mutation works as follows:

1. Calculate the average fitness value of the population (``f_avg``).

2. For each chromosome, calculate its fitness value (``f``).

3. If ``f<f_avg``, then this solution is regarded as a **low-quality**
   solution and thus the mutation rate should be kept high because this
   would increase the quality of this solution.

4. If ``f>f_avg``, then this solution is regarded as a **high-quality**
   solution and thus the mutation rate should be kept low to avoid
   disrupting this high quality solution.

In PyGAD, if ``f=f_avg``, then the solution is regarded of high quality.

The next figure summarizes the previous steps.

.. figure:: https://user-images.githubusercontent.com/16560492/103468973-e3c26600-4d2c-11eb-8af3-b3bb39b50540.jpg
   :alt: 

This strategy is applied in PyGAD.

.. _header-n918:

Use Adaptive Mutation in PyGAD
------------------------------

In PyGAD 2.10.0, adaptive mutation is supported. To use it, just follow
the following 2 simple steps:

1. In the constructor of the ``pygad.GA`` class, set
   ``mutation_type="adaptive"`` to specify that the type of mutation is
   adaptive.

2. Specify the mutation rates for the low and high quality solutions
   using one of these 3 parameters according to your preference:
   ``mutation_probability``, ``mutation_num_genes``, and
   ``mutation_percent_genes``. Please check the `documentation of each
   of these
   parameters <https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#init>`__
   for more information.

When adaptive mutation is used, then the value assigned to any of the 3
parameters can be of any of these data types:

1. ``list``

2. ``tuple``

3. ``numpy.ndarray``

Whatever the data type used, the length of the ``list``, ``tuple``, or
the ``numpy.ndarray`` must be exactly 2. That is there are just 2
values:

1. The first value is the mutation rate for the low-quality solutions.

2. The second value is the mutation rate for the high-quality solutions.

PyGAD expects that the first value is higher than the second value and
thus a warning is printed in case the first value is lower than the
second one.

Here are some examples to feed the mutation rates:

.. code:: python

   # mutation_probability
   mutation_probability = [0.25, 0.1]
   mutation_probability = (0.35, 0.17)
   mutation_probability = numpy.array([0.15, 0.05])

   # mutation_num_genes
   mutation_num_genes = [4, 2]
   mutation_num_genes = (3, 1)
   mutation_num_genes = numpy.array([7, 2])

   # mutation_percent_genes
   mutation_percent_genes = [25, 12]
   mutation_percent_genes = (15, 8)
   mutation_percent_genes = numpy.array([21, 13])

Assume that the average fitness is 12 and the fitness values of 2
solutions are 15 and 7. If the mutation probabilities are specified as
follows:

.. code:: python

   mutation_probability = [0.25, 0.1]

Then the mutation probability of the first solution is 0.1 because its
fitness is 15 which is higher than the average fitness 12. The mutation
probability of the second solution is 0.25 because its fitness is 7
which is lower than the average fitness 12.

Here is an example that uses adaptive mutation.

.. code:: python

   import pygad
   import numpy

   function_inputs = [4,-2,3.5,5,-11,-4.7] # Function inputs.
   desired_output = 44 # Function output.

   def fitness_func(solution, solution_idx):
       # The fitness function calulates the sum of products between each input and its corresponding weight.
       output = numpy.sum(solution*function_inputs)
       # The value 0.000001 is used to avoid the Inf value when the denominator numpy.abs(output - desired_output) is 0.0.
       fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
       return fitness

   # Creating an instance of the GA class inside the ga module. Some parameters are initialized within the constructor.
   ga_instance = pygad.GA(num_generations=200,
                          fitness_func=fitness_func,
                          num_parents_mating=10,
                          sol_per_pop=20,
                          num_genes=len(function_inputs),
                          mutation_type="adaptive",
                          mutation_num_genes=(3, 1))

   # Running the GA to optimize the parameters of the function.
   ga_instance.run()

   ga_instance.plot_result(title="PyGAD with Adaptive Mutation", linewidth=5)

.. _header-n947:

Examples
========

This section gives the complete code of some examples that use
``pygad``. Each subsection builds a different example.

.. _header-n949:

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
       last_fitness = ga_instance.best_solution()[1]

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
                          on_generation=callback_generation)

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

.. _header-n952:

Reproducing Images
------------------

This project reproduces a single image using PyGAD by evolving pixel
values. This project works with both color and gray images. Check this
project at `GitHub <https://github.com/ahmedfgad/GARI>`__:
https://github.com/ahmedfgad/GARI.

For more information about this project, read this tutorial titled
`Reproducing Images using a Genetic Algorithm with
Python <https://www.linkedin.com/pulse/reproducing-images-using-genetic-algorithm-python-ahmed-gad>`__
available at these links:

-  `Heartbeat <https://heartbeat.fritz.ai/reproducing-images-using-a-genetic-algorithm-with-python-91fc701ff84>`__:
   https://heartbeat.fritz.ai/reproducing-images-using-a-genetic-algorithm-with-python-91fc701ff84

-  `LinkedIn <https://www.linkedin.com/pulse/reproducing-images-using-genetic-algorithm-python-ahmed-gad>`__:
   https://www.linkedin.com/pulse/reproducing-images-using-genetic-algorithm-python-ahmed-gad

.. _header-n960:

Project Steps
~~~~~~~~~~~~~

The steps to follow in order to reproduce an image are as follows:

-  Read an image

-  Prepare the fitness function

-  Create an instance of the pygad.GA class with the appropriate
   parameters

-  Run PyGAD

-  Plot results

-  Calculate some statistics

The next sections discusses the code of each of these steps.

.. _header-n976:

Read an Image
~~~~~~~~~~~~~

There is an image named ``fruit.jpg`` in the `GARI
project <https://github.com/ahmedfgad/GARI>`__ which is read according
to the next code.

.. code:: python

   import imageio
   import numpy

   target_im = imageio.imread('fruit.jpg')
   target_im = numpy.asarray(target_im/255, dtype=numpy.float)

Here is the read image.

.. figure:: https://user-images.githubusercontent.com/16560492/36948808-f0ac882e-1fe8-11e8-8d07-1307e3477fd0.jpg
   :alt: 

Based on the chromosome representation used in the example, the pixel
values can be either in the 0-255, 0-1, or any other ranges.

Note that the range of pixel values affect other parameters like the
range from which the random values are selected during mutation and also
the range of the values used in the initial population. So, be
consistent.

.. _header-n983:

Prepare the Fitness Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The next code creates a function that will be used as a fitness function
for calculating the fitness value for each solution in the population.
This function must be a maximization function that accepts 2 parameters
representing a solution and its index. It returns a value representing
the fitness value.

.. code:: python

   import gari

   target_chromosome = gari.img2chromosome(target_im)

   def fitness_fun(solution, solution_idx):
       fitness = numpy.sum(numpy.abs(target_chromosome-solution))

       # Negating the fitness value to make it increasing rather than decreasing.
       fitness = numpy.sum(target_chromosome) - fitness
       return fitness

The fitness value is calculated using the sum of absolute difference
between genes values in the original and reproduced chromosomes. The
``gari.img2chromosome()`` function is called before the fitness function
to represent the image as a vector because the genetic algorithm can
work with 1D chromosomes.

The implementation of the ``gari`` module is available at the `GARI
GitHub
project <https://github.com/ahmedfgad/GARI/blob/master/gari.py>`__ and
its code is listed below.

.. code:: python

   import numpy
   import functools
   import operator

   def img2chromosome(img_arr):
       return numpy.reshape(a=img_arr, newshape=(functools.reduce(operator.mul, img_arr.shape)))

   def chromosome2img(vector, shape):
       if len(vector) != functools.reduce(operator.mul, shape):
           raise ValueError("A vector of length {vector_length} into an array of shape {shape}.".format(vector_length=len(vector), shape=shape))

       return numpy.reshape(a=vector, newshape=shape)

.. _header-n989:

Create an Instance of the ``pygad.GA`` Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is very important to use random mutation and set the
``mutation_by_replacement`` to ``True``. Based on the range of pixel
values, the values assigned to the ``init_range_low``,
``init_range_high``, ``random_mutation_min_val``, and
``random_mutation_max_val`` parameters should be changed.

If the image pixel values range from 0 to 255, then set
``init_range_low`` and ``random_mutation_min_val`` to 0 as they are but
change ``init_range_high`` and ``random_mutation_max_val`` to 255.

Feel free to change the other parameters or add other parameters. Please
check the `PyGAD's documentation <https://pygad.readthedocs.io>`__ for
the full list of parameters.

.. code:: python

   import pygad

   ga_instance = pygad.GA(num_generations=20000,
                          num_parents_mating=10,
                          fitness_func=fitness_fun,
                          sol_per_pop=20,
                          num_genes=target_im.size,
                          init_range_low=0.0,
                          init_range_high=1.0,
                          mutation_percent_genes=0.01,
                          mutation_type="random",
                          mutation_by_replacement=True,
                          random_mutation_min_val=0.0,
                          random_mutation_max_val=1.0)

.. _header-n994:

Run PyGAD
~~~~~~~~~

Simply, call the ``run()`` method to run PyGAD.

.. code:: python

   ga_instance.run()

.. _header-n997:

Plot Results
~~~~~~~~~~~~

After the ``run()`` method completes, the fitness values of all
generations can be viewed in a plot using the ``plot_result()`` method.

.. code:: python

   ga_instance.plot_result()

Here is the plot after 20,000 generations.

.. figure:: https://user-images.githubusercontent.com/16560492/82232124-77762c00-992e-11ea-9fc6-14a1cd7a04ff.png
   :alt: 

.. _header-n1002:

Calculate Some Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~

Here is some information about the best solution.

.. code:: python

   # Returning the details of the best solution.
   solution, solution_fitness, solution_idx = ga_instance.best_solution()
   print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
   print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

   if ga_instance.best_solution_generation != -1:
       print("Best fitness value reached after {best_solution_generation} generations.".format(best_solution_generation=ga_instance.best_solution_generation))

   result = gari.chromosome2img(solution, target_im.shape)
   matplotlib.pyplot.imshow(result)
   matplotlib.pyplot.title("PyGAD & GARI for Reproducing Images")
   matplotlib.pyplot.show()

.. _header-n1005:

Evolution by Generation
~~~~~~~~~~~~~~~~~~~~~~~

The solution reached after the 20,000 generations is shown below.

.. figure:: https://user-images.githubusercontent.com/16560492/82232405-e0f63a80-992e-11ea-984f-b6ed76465bd1.png
   :alt: 

After more generations, the result can be enhanced like what shown
below.

.. figure:: https://user-images.githubusercontent.com/16560492/82232345-cf149780-992e-11ea-8390-bf1a57a19de7.png
   :alt: 

The results can also be enhanced by changing the parameters passed to
the constructor of the ``pygad.GA`` class.

Here is how the image is evolved from generation 0 to generation
20,000s.

**Generation 0**

.. figure:: https://user-images.githubusercontent.com/16560492/36948589-b47276f0-1fe5-11e8-8efe-0cd1a225ea3a.png
   :alt: 

**Generation 1,000**

.. figure:: https://user-images.githubusercontent.com/16560492/36948823-16f490ee-1fe9-11e8-97db-3e8905ad5440.png
   :alt: 

**Generation 2,500**

.. figure:: https://user-images.githubusercontent.com/16560492/36948832-3f314b60-1fe9-11e8-8f4a-4d9a53b99f3d.png
   :alt: 

**Generation 4,500**

.. figure:: https://user-images.githubusercontent.com/16560492/36948837-53d1849a-1fe9-11e8-9b36-e9e9291e347b.png
   :alt: 

**Generation 7,000**

.. figure:: https://user-images.githubusercontent.com/16560492/36948852-66f1b176-1fe9-11e8-9f9b-460804e94004.png
   :alt: 

**Generation 8,000**

.. figure:: https://user-images.githubusercontent.com/16560492/36948865-7fbb5158-1fe9-11e8-8c04-8ac3c1f7b1b1.png
   :alt: 

**Generation 20,000**

.. figure:: https://user-images.githubusercontent.com/16560492/82232405-e0f63a80-992e-11ea-984f-b6ed76465bd1.png
   :alt:
