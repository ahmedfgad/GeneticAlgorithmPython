.. _pygadtorchga-module:

``pygad.torchga`` Module
========================

This section of the PyGAD's library documentation discusses the
**pygad.utils** module.

PyGAD supports different types of operators for selecting the parents,
applying the crossover, and mutation. More features will be added in the
future. To ask for a new feature, please check the `Ask for
Feature <https://pygad.readthedocs.io/en/latest/releases.html#ask-for-feature>`__
section.

The submodules in the ``pygad.utils`` module are:

1. ``crossover``: Has the ``Crossover`` class that implements the
   crossover operators.

2. ``mutation``: Has the ``Mutation`` class that implements the mutation
   operators.

3. ``parent_selection``: Has the ``ParentSelection`` class that
   implements the parent selection operators.

4. ``nsga2``: Has the ``NSGA2`` class that implements the Non-Dominated
   Sorting Genetic Algorithm II (NSGA-II).

Note that the ``pygad.GA`` class extends all of these classes. So, the
user can access any of the methods in such classes directly by the
instance/object of the ``pygad.GA`` class.

The next sections discuss each submodule.

.. _pygadutilscrossover-submodule:

``pygad.utils.crossover`` Submodule
===================================

The ``pygad.utils.crossover`` module has a class named ``Crossover``
with the supported crossover operations which are:

1. Single point: Implemented using the ``single_point_crossover()``
   method.

2. Two points: Implemented using the ``two_points_crossover()`` method.

3. Uniform: Implemented using the ``uniform_crossover()`` method.

4. Scattered: Implemented using the ``scattered_crossover()`` method.

All crossover methods accept this parameter:

1. ``parents``: The parents to mate for producing the offspring.

2. ``offspring_size``: The size of the offspring to produce.

.. _pygadutilsmutation-submodule:

``pygad.utils.mutation`` Submodule
==================================

The ``pygad.utils.mutation`` module has a class named ``Mutation`` with
the supported mutation operations which are:

1. Random: Implemented using the ``random_mutation()`` method.

2. Swap: Implemented using the ``swap_mutation()`` method.

3. Inversion: Implemented using the ``inversion_mutation()`` method.

4. Scramble: Implemented using the ``scramble_mutation()`` method.

5. Scramble: Implemented using the ``adaptive_mutation()`` method.

All mutation methods accept this parameter:

1. ``offspring``: The offspring to mutate.

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

3. If ``f<f_avg``, then this solution is regarded as a low-quality
   solution and thus the mutation rate should be kept high because this
   would increase the quality of this solution.

4. If ``f>f_avg``, then this solution is regarded as a high-quality
   solution and thus the mutation rate should be kept low to avoid
   disrupting this high quality solution.

In PyGAD, if ``f=f_avg``, then the solution is regarded of high quality.

The next figure summarizes the previous steps.

.. image:: https://user-images.githubusercontent.com/16560492/103468973-e3c26600-4d2c-11eb-8af3-b3bb39b50540.jpg
   :alt: 

This strategy is applied in PyGAD.

Use Adaptive Mutation in PyGAD
------------------------------

In `PyGAD
2.10.0 <https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-10-0>`__,
adaptive mutation is supported. To use it, just follow the following 2
simple steps:

1. In the constructor of the ``pygad.GA`` class, set
   ``mutation_type="adaptive"`` to specify that the type of mutation is
   adaptive.

2. Specify the mutation rates for the low and high quality solutions
   using one of these 3 parameters according to your preference:
   ``mutation_probability``, ``mutation_num_genes``, and
   ``mutation_percent_genes``. Please check the `documentation of each
   of these
   parameters <https://pygad.readthedocs.io/en/latest/pygad.html#init>`__
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

   def fitness_func(ga_instance, solution, solution_idx):
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

   ga_instance.plot_fitness(title="PyGAD with Adaptive Mutation", linewidth=5)

.. _pygadutilsparentselection-submodule:

``pygad.utils.parent_selection`` Submodule
==========================================

The ``pygad.utils.parent_selection`` module has a class named
``ParentSelection`` with the supported parent selection operations which
are:

1. Steady-state: Implemented using the ``steady_state_selection()``
   method.

2. Roulette wheel: Implemented using the ``roulette_wheel_selection()``
   method.

3. Stochastic universal: Implemented using the
   ``stochastic_universal_selection()``\ method.

4. Rank: Implemented using the ``rank_selection()`` method.

5. Random: Implemented using the ``random_selection()`` method.

6. Tournament: Implemented using the ``tournament_selection()`` method.

7. NSGA-II: Implemented using the ``nsga2_selection()`` method.

8. NSGA-II Tournament: Implemented using the
   ``tournament_nsga2_selection()`` method.

All parent selection methods accept these parameters:

1. ``fitness``: The fitness of the entire population.

2. ``num_parents``: The number of parents to select.

.. _pygadutilsnsga2-submodule:

``pygad.utils.nsga2`` Submodule
===============================

The ``pygad.utils.nsga2`` module has a class named ``NSGA2`` that
implements NSGA-II. The methods inside this class are:

1. ``non_dominated_sorting()``: Returns all the pareto fronts by
   applying non-dominated sorting over the solutions.

2. ``get_non_dominated_set()``: Returns the set of non-dominated
   solutions from the passed solutions.

3. ``crowding_distance()``: Calculates the crowding distance for all
   solutions in the current pareto front.

4. ``sort_solutions_nsga2()``: Sort the solutions. If the problem is
   single-objective, then the solutions are sorted by sorting the
   fitness values of the population. If it is multi-objective, then
   non-dominated sorting and crowding distance are applied to sort the
   solutions.

User-Defined Crossover, Mutation, and Parent Selection Operators
================================================================

Previously, the user can select the the type of the crossover, mutation,
and parent selection operators by assigning the name of the operator to
the following parameters of the ``pygad.GA`` class's constructor:

1. ``crossover_type``

2. ``mutation_type``

3. ``parent_selection_type``

This way, the user can only use the built-in functions for each of these
operators.

Starting from `PyGAD
2.16.0 <https://pygad.readthedocs.io/en/latest/releases.html#pygad-2-16-0>`__,
the user can create a custom crossover, mutation, and parent selection
operators and assign these functions to the above parameters. Thus, a
new operator can be plugged easily into the `PyGAD
Lifecycle <https://pygad.readthedocs.io/en/latest/pygad.html#life-cycle-of-pygad>`__.

This is a sample code that does not use any custom function.

.. code:: python

   import pygad
   import numpy

   equation_inputs = [4,-2,3.5]
   desired_output = 44

   def fitness_func(ga_instance, solution, solution_idx):
       output = numpy.sum(solution * equation_inputs)
       fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
       return fitness

   ga_instance = pygad.GA(num_generations=10,
                          sol_per_pop=5,
                          num_parents_mating=2,
                          num_genes=len(equation_inputs),
                          fitness_func=fitness_func)

   ga_instance.run()
   ga_instance.plot_fitness()

This section describes the expected input parameters and outputs. For
simplicity, all of these custom functions all accept the instance of the
``pygad.GA`` class as the last parameter.

User-Defined Crossover Operator
-------------------------------

The user-defined crossover function is a Python function that accepts 3
parameters:

1. The selected parents.

2. The size of the offspring as a tuple of 2 numbers: (the offspring
   size, number of genes).

3. The instance from the ``pygad.GA`` class. This instance helps to
   retrieve any property like ``population``, ``gene_type``,
   ``gene_space``, etc.

This function should return a NumPy array of shape equal to the value
passed to the second parameter.

The next code creates a template for the user-defined crossover
operator. You can use any names for the parameters. Note how a NumPy
array is returned.

.. code:: python

   def crossover_func(parents, offspring_size, ga_instance):
       offspring = ...
       ...
       return numpy.array(offspring)

As an example, the next code creates a single-point crossover function.
By randomly generating a random point (i.e. index of a gene), the
function simply uses 2 parents to produce an offspring by copying the
genes before the point from the first parent and the remaining from the
second parent.

.. code:: python

   def crossover_func(parents, offspring_size, ga_instance):
       offspring = []
       idx = 0
       while len(offspring) != offspring_size[0]:
           parent1 = parents[idx % parents.shape[0], :].copy()
           parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

           random_split_point = numpy.random.choice(range(offspring_size[1]))

           parent1[random_split_point:] = parent2[random_split_point:]

           offspring.append(parent1)

           idx += 1

       return numpy.array(offspring)

To use this user-defined function, simply assign its name to the
``crossover_type`` parameter in the constructor of the ``pygad.GA``
class. The next code gives an example. In this case, the custom function
will be called in each generation rather than calling the built-in
crossover functions defined in PyGAD.

.. code:: python

   ga_instance = pygad.GA(num_generations=10,
                          sol_per_pop=5,
                          num_parents_mating=2,
                          num_genes=len(equation_inputs),
                          fitness_func=fitness_func,
                          crossover_type=crossover_func)

User-Defined Mutation Operator
------------------------------

A user-defined mutation function/operator can be created the same way a
custom crossover operator/function is created. Simply, it is a Python
function that accepts 2 parameters:

1. The offspring to be mutated.

2. The instance from the ``pygad.GA`` class. This instance helps to
   retrieve any property like ``population``, ``gene_type``,
   ``gene_space``, etc.

The template for the user-defined mutation function is given in the next
code. According to the user preference, the function should make some
random changes to the genes.

.. code:: python

   def mutation_func(offspring, ga_instance):
       ...
       return offspring

The next code builds the random mutation where a single gene from each
chromosome is mutated by adding a random number between 0 and 1 to the
gene's value.

.. code:: python

   def mutation_func(offspring, ga_instance):

       for chromosome_idx in range(offspring.shape[0]):
           random_gene_idx = numpy.random.choice(range(offspring.shape[1]))

           offspring[chromosome_idx, random_gene_idx] += numpy.random.random()

       return offspring

Here is how this function is assigned to the ``mutation_type``
parameter.

.. code:: python

   ga_instance = pygad.GA(num_generations=10,
                          sol_per_pop=5,
                          num_parents_mating=2,
                          num_genes=len(equation_inputs),
                          fitness_func=fitness_func,
                          crossover_type=crossover_func,
                          mutation_type=mutation_func)

Note that there are other things to take into consideration like:

-  Making sure that each gene conforms to the data type(s) listed in the
   ``gene_type`` parameter.

-  If the ``gene_space`` parameter is used, then the new value for the
   gene should conform to the values/ranges listed.

-  Mutating a number of genes that conforms to the parameters
   ``mutation_percent_genes``, ``mutation_probability``, and
   ``mutation_num_genes``.

-  Whether mutation happens with or without replacement based on the
   ``mutation_by_replacement`` parameter.

-  The minimum and maximum values from which a random value is generated
   based on the ``random_mutation_min_val`` and
   ``random_mutation_max_val`` parameters.

-  Whether duplicates are allowed or not in the chromosome based on the
   ``allow_duplicate_genes`` parameter.

and more.

It all depends on your objective from building the mutation function.
You may neglect or consider some of the considerations according to your
objective.

User-Defined Parent Selection Operator
--------------------------------------

No much to mention about building a user-defined parent selection
function as things are similar to building a crossover or mutation
function. Just create a Python function that accepts 3 parameters:

1. The fitness values of the current population.

2. The number of parents needed.

3. The instance from the ``pygad.GA`` class. This instance helps to
   retrieve any property like ``population``, ``gene_type``,
   ``gene_space``, etc.

The function should return 2 outputs:

1. The selected parents as a NumPy array. Its shape is equal to (the
   number of selected parents, ``num_genes``). Note that the number of
   selected parents is equal to the value assigned to the second input
   parameter.

2. The indices of the selected parents inside the population. It is a 1D
   list with length equal to the number of selected parents.

The outputs must be of type ``numpy.ndarray``.

Here is a template for building a custom parent selection function.

.. code:: python

   def parent_selection_func(fitness, num_parents, ga_instance):
       ...
       return parents, fitness_sorted[:num_parents]

The next code builds the steady-state parent selection where the best
parents are selected. The number of parents is equal to the value in the
``num_parents`` parameter.

.. code:: python

   def parent_selection_func(fitness, num_parents, ga_instance):

       fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
       fitness_sorted.reverse()

       parents = numpy.empty((num_parents, ga_instance.population.shape[1]))

       for parent_num in range(num_parents):
           parents[parent_num, :] = ga_instance.population[fitness_sorted[parent_num], :].copy()

       return parents, numpy.array(fitness_sorted[:num_parents])

Finally, the defined function is assigned to the
``parent_selection_type`` parameter as in the next code.

.. code:: python

   ga_instance = pygad.GA(num_generations=10,
                          sol_per_pop=5,
                          num_parents_mating=2,
                          num_genes=len(equation_inputs),
                          fitness_func=fitness_func,
                          crossover_type=crossover_func,
                          mutation_type=mutation_func,
                          parent_selection_type=parent_selection_func)

Example
-------

By discussing how to customize the 3 operators, the next code uses the
previous 3 user-defined functions instead of the built-in functions.

.. code:: python

   import pygad
   import numpy

   equation_inputs = [4,-2,3.5]
   desired_output = 44

   def fitness_func(ga_instance, solution, solution_idx):
       output = numpy.sum(solution * equation_inputs)

       fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)

       return fitness

   def parent_selection_func(fitness, num_parents, ga_instance):

       fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
       fitness_sorted.reverse()

       parents = numpy.empty((num_parents, ga_instance.population.shape[1]))

       for parent_num in range(num_parents):
           parents[parent_num, :] = ga_instance.population[fitness_sorted[parent_num], :].copy()

       return parents, numpy.array(fitness_sorted[:num_parents])

   def crossover_func(parents, offspring_size, ga_instance):

       offspring = []
       idx = 0
       while len(offspring) != offspring_size[0]:
           parent1 = parents[idx % parents.shape[0], :].copy()
           parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

           random_split_point = numpy.random.choice(range(offspring_size[1]))

           parent1[random_split_point:] = parent2[random_split_point:]

           offspring.append(parent1)

           idx += 1

       return numpy.array(offspring)

   def mutation_func(offspring, ga_instance):

       for chromosome_idx in range(offspring.shape[0]):
           random_gene_idx = numpy.random.choice(range(offspring.shape[0]))

           offspring[chromosome_idx, random_gene_idx] += numpy.random.random()

       return offspring

   ga_instance = pygad.GA(num_generations=10,
                          sol_per_pop=5,
                          num_parents_mating=2,
                          num_genes=len(equation_inputs),
                          fitness_func=fitness_func,
                          crossover_type=crossover_func,
                          mutation_type=mutation_func,
                          parent_selection_type=parent_selection_func)

   ga_instance.run()
   ga_instance.plot_fitness()

This is the same example but using methods instead of functions.

.. code:: python

   import pygad
   import numpy

   equation_inputs = [4,-2,3.5]
   desired_output = 44

   class Test:
       def fitness_func(self, ga_instance, solution, solution_idx):
           output = numpy.sum(solution * equation_inputs)
       
           fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
       
           return fitness
       
       def parent_selection_func(self, fitness, num_parents, ga_instance):
       
           fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
           fitness_sorted.reverse()
       
           parents = numpy.empty((num_parents, ga_instance.population.shape[1]))
       
           for parent_num in range(num_parents):
               parents[parent_num, :] = ga_instance.population[fitness_sorted[parent_num], :].copy()
       
           return parents, numpy.array(fitness_sorted[:num_parents])
       
       def crossover_func(self, parents, offspring_size, ga_instance):

           offspring = []
           idx = 0
           while len(offspring) != offspring_size[0]:
               parent1 = parents[idx % parents.shape[0], :].copy()
               parent2 = parents[(idx + 1) % parents.shape[0], :].copy()
       
               random_split_point = numpy.random.choice(range(offspring_size[0]))
       
               parent1[random_split_point:] = parent2[random_split_point:]
       
               offspring.append(parent1)
       
               idx += 1
       
           return numpy.array(offspring)
       
       def mutation_func(self, offspring, ga_instance):

           for chromosome_idx in range(offspring.shape[0]):
               random_gene_idx = numpy.random.choice(range(offspring.shape[1]))
       
               offspring[chromosome_idx, random_gene_idx] += numpy.random.random()
       
           return offspring

   ga_instance = pygad.GA(num_generations=10,
                          sol_per_pop=5,
                          num_parents_mating=2,
                          num_genes=len(equation_inputs),
                          fitness_func=Test().fitness_func,
                          parent_selection_type=Test().parent_selection_func,
                          crossover_type=Test().crossover_func,
                          mutation_type=Test().mutation_func)

   ga_instance.run()
   ga_instance.plot_fitness()
