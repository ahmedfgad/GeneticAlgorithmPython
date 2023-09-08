.. _pygadvisualize-module:

``pygad.visualize`` Module
==========================

This section of the PyGAD's library documentation discusses the
**pygad.visualize** module. It offers the methods for results
visualization in PyGAD.

This section discusses the different options to visualize the results in
PyGAD through these methods:

1. ``plot_fitness()``: Create plots for the fitness.

2. ``plot_genes()``: Create plots for the genes.

3. ``plot_new_solution_rate()``: Create plots for the new solution rate.

In the following code, the ``save_solutions`` flag is set to ``True``
which means all solutions are saved in the ``solutions`` attribute. The
code runs for only 10 generations.

.. code:: python

   import pygad
   import numpy

   equation_inputs = [4, -2, 3.5, 8, -2, 3.5, 8]
   desired_output = 2671.1234

   def fitness_func(ga_instance, solution, solution_idx):
       output = numpy.sum(solution * equation_inputs)
       fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
       return fitness

   ga_instance = pygad.GA(num_generations=10,
                          sol_per_pop=10,
                          num_parents_mating=5,
                          num_genes=len(equation_inputs),
                          fitness_func=fitness_func,
                          gene_space=[range(1, 10), range(10, 20), range(15, 30), range(20, 40), range(25, 50), range(10, 30), range(20, 50)],
                          gene_type=int,
                          save_solutions=True)

   ga_instance.run()

Let's explore how to visualize the results by the above mentioned
methods.

Fitness
=======

.. _plotfitness:

``plot_fitness()``
------------------

The ``plot_fitness()`` method shows the fitness value for each
generation. It creates, shows, and returns a figure that summarizes how
the fitness value(s) evolve(s) by generation.

It works only after completing at least 1 generation. If no generation
is completed (at least 1), an exception is raised.

This method accepts the following parameters:

1. ``title``: Title of the figure.

2. ``xlabel``: X-axis label.

3. ``ylabel``: Y-axis label.

4. ``linewidth``: Line width of the plot. Defaults to ``3``.

5. ``font_size``: Font size for the labels and title. Defaults to
   ``14``.

6. ``plot_type``: Type of the plot which can be either ``"plot"``
   (default), ``"scatter"``, or ``"bar"``.

7. ``color``: Color of the plot which defaults to the greenish color
   ``"#64f20c"``.

8. ``label``: The label used for the legend in the figures of
   multi-objective problems. It is not used for single-objective
   problems. It defaults to ``None`` which means no labels used.

9. ``save_dir``: Directory to save the figure.

.. _plottypeplot:

``plot_type="plot"``
~~~~~~~~~~~~~~~~~~~~

The simplest way to call this method is as follows leaving the
``plot_type`` with its default value ``"plot"`` to create a continuous
line connecting the fitness values across all generations:

.. code:: python

   ga_instance.plot_fitness()
   # ga_instance.plot_fitness(plot_type="plot")

.. image:: https://user-images.githubusercontent.com/16560492/122472609-d02f5280-cf8e-11eb-88a7-f9366ff6e7c6.png
   :alt: 

.. _plottypescatter:

``plot_type="scatter"``
~~~~~~~~~~~~~~~~~~~~~~~

The ``plot_type`` can also be set to ``"scatter"`` to create a scatter
graph with each individual fitness represented as a dot. The size of
these dots can be changed using the ``linewidth`` parameter.

.. code:: python

   ga_instance.plot_fitness(plot_type="scatter")

.. image:: https://user-images.githubusercontent.com/16560492/122473159-75e2c180-cf8f-11eb-942d-31279b286dbd.png
   :alt: 

.. _plottypebar:

``plot_type="bar"``
~~~~~~~~~~~~~~~~~~~

The third value for the ``plot_type`` parameter is ``"bar"`` to create a
bar graph with each individual fitness represented as a bar.

.. code:: python

   ga_instance.plot_fitness(plot_type="bar")

.. image:: https://user-images.githubusercontent.com/16560492/122473340-b7736c80-cf8f-11eb-89c5-4f7db3b653cc.png
   :alt: 

New Solution Rate
=================

.. _plotnewsolutionrate:

``plot_new_solution_rate()``
----------------------------

The ``plot_new_solution_rate()`` method presents the number of new
solutions explored in each generation. This helps to figure out if the
genetic algorithm is able to find new solutions as an indication of more
possible evolution. If no new solutions are explored, this is an
indication that no further evolution is possible.

It works only after completing at least 1 generation. If no generation
is completed (at least 1), an exception is raised.

The ``plot_new_solution_rate()`` method accepts the same parameters as
in the ``plot_fitness()`` method (it also have 3 possible values for
``plot_type`` parameter). Here are all the parameters it accepts:

1. ``title``: Title of the figure.

2. ``xlabel``: X-axis label.

3. ``ylabel``: Y-axis label.

4. ``linewidth``: Line width of the plot. Defaults to ``3``.

5. ``font_size``: Font size for the labels and title. Defaults to
   ``14``.

6. ``plot_type``: Type of the plot which can be either ``"plot"``
   (default), ``"scatter"``, or ``"bar"``.

7. ``color``: Color of the plot which defaults to ``"#3870FF"``.

8. ``save_dir``: Directory to save the figure.

.. _plottypeplot-2:

``plot_type="plot"``
~~~~~~~~~~~~~~~~~~~~

The default value for the ``plot_type`` parameter is ``"plot"``.

.. code:: python

   ga_instance.plot_new_solution_rate()
   # ga_instance.plot_new_solution_rate(plot_type="plot")

The next figure shows that, for example, generation 6 has the least
number of new solutions which is 4. The number of new solutions in the
first generation is always equal to the number of solutions in the
population (i.e. the value assigned to the ``sol_per_pop`` parameter in
the constructor of the ``pygad.GA`` class) which is 10 in this example.

.. image:: https://user-images.githubusercontent.com/16560492/122475815-3322e880-cf93-11eb-9648-bf66f823234b.png
   :alt: 

.. _plottypescatter-2:

``plot_type="scatter"``
~~~~~~~~~~~~~~~~~~~~~~~

The previous graph can be represented as scattered points by setting
``plot_type="scatter"``.

.. code:: python

   ga_instance.plot_new_solution_rate(plot_type="scatter")

.. image:: https://user-images.githubusercontent.com/16560492/122476108-adec0380-cf93-11eb-80ac-7588bf90492f.png
   :alt: 

.. _plottypebar-2:

``plot_type="bar"``
~~~~~~~~~~~~~~~~~~~

By setting ``plot_type="scatter"``, each value is represented as a
vertical bar.

.. code:: python

   ga_instance.plot_new_solution_rate(plot_type="bar")

.. image:: https://user-images.githubusercontent.com/16560492/122476173-c2c89700-cf93-11eb-9e77-d39737cd3a96.png
   :alt: 

Genes
=====

.. _plotgenes:

``plot_genes()``
----------------

The ``plot_genes()`` method is the third option to visualize the PyGAD
results. The ``plot_genes()`` method creates, shows, and returns a
figure that describes each gene. It has different options to create the
figures which helps to:

1. Explore the gene value for each generation by creating a normal plot.

2. Create a histogram for each gene.

3. Create a boxplot.

It works only after completing at least 1 generation. If no generation
is completed, an exception is raised. If no generation is completed (at
least 1), an exception is raised.

This method accepts the following parameters:

1.  ``title``: Title of the figure.

2.  ``xlabel``: X-axis label.

3.  ``ylabel``: Y-axis label.

4.  ``linewidth``: Line width of the plot. Defaults to ``3``.

5.  ``font_size``: Font size for the labels and title. Defaults to
    ``14``.

6.  ``plot_type``: Type of the plot which can be either ``"plot"``
    (default), ``"scatter"``, or ``"bar"``.

7.  ``graph_type``: Type of the graph which can be either ``"plot"``
    (default), ``"boxplot"``, or ``"histogram"``.

8.  ``fill_color``: Fill color of the graph which defaults to
    ``"#3870FF"``. This has no effect if ``graph_type="plot"``.

9.  ``color``: Color of the plot which defaults to ``"#3870FF"``.

10. ``solutions``: Defaults to ``"all"`` which means use all solutions.
    If ``"best"`` then only the best solutions are used.

11. ``save_dir``: Directory to save the figure.

This method has 3 control variables:

1. ``graph_type="plot"``: Can be ``"plot"`` (default), ``"boxplot"``, or
   ``"histogram"``.

2. ``plot_type="plot"``: Identical to the ``plot_type`` parameter
   explored in the ``plot_fitness()`` and ``plot_new_solution_rate()``
   methods.

3. ``solutions="all"``: Can be ``"all"`` (default) or ``"best"``.

These 3 parameters controls the style of the output figure.

The ``graph_type`` parameter selects the type of the graph which helps
to explore the gene values as:

1. A normal plot.

2. A histogram.

3. A box and whisker plot.

The ``plot_type`` parameter works only when the type of the graph is set
to ``"plot"``.

The ``solutions`` parameter selects whether the genes come from all
solutions in the population or from just the best solutions.

An exception is raised if:

-  ``solutions="all"`` while ``save_solutions=False`` in the constructor
   of the ``pygad.GA`` class. .

-  ``solutions="best"`` while ``save_best_solutions=False`` in the
   constructor of the ``pygad.GA`` class. .

.. _graphtypeplot:

``graph_type="plot"``
~~~~~~~~~~~~~~~~~~~~~

When ``graph_type="plot"``, then the figure creates a normal graph where
the relationship between the gene values and the generation numbers is
represented as a continuous plot, scattered points, or bars.

.. _plottypeplot-3:

``plot_type="plot"``
^^^^^^^^^^^^^^^^^^^^

Because the default value for both ``graph_type`` and ``plot_type`` is
``"plot"``, then all of the lines below creates the same figure. This
figure is helpful to know whether a gene value lasts for more
generations as an indication of the best value for this gene. For
example, the value 16 for the gene with index 5 (at column 2 and row 2
of the next graph) lasted for 83 generations.

.. code:: python

   ga_instance.plot_genes()

   ga_instance.plot_genes(graph_type="plot")

   ga_instance.plot_genes(plot_type="plot")

   ga_instance.plot_genes(graph_type="plot", 
                          plot_type="plot")

.. image:: https://user-images.githubusercontent.com/16560492/122477158-4a62d580-cf95-11eb-8c93-9b6e74cb814c.png
   :alt: 

As the default value for the ``solutions`` parameter is ``"all"``, then
the following method calls generate the same plot.

.. code:: python

   ga_instance.plot_genes(solutions="all")

   ga_instance.plot_genes(graph_type="plot",
                          solutions="all")

   ga_instance.plot_genes(plot_type="plot",
                          solutions="all")

   ga_instance.plot_genes(graph_type="plot", 
                          plot_type="plot",
                          solutions="all")

.. _plottypescatter-3:

``plot_type="scatter"``
^^^^^^^^^^^^^^^^^^^^^^^

The following calls of the ``plot_genes()`` method create the same
scatter plot.

.. code:: python

   ga_instance.plot_genes(plot_type="scatter")

   ga_instance.plot_genes(graph_type="plot", 
                          plot_type="scatter", 
                          solutions='all')

.. image:: https://user-images.githubusercontent.com/16560492/122477273-73836600-cf95-11eb-828f-f357c7b0f815.png
   :alt: 

.. _plottypebar-3:

``plot_type="bar"``
^^^^^^^^^^^^^^^^^^^

.. code:: python

   ga_instance.plot_genes(plot_type="bar")

   ga_instance.plot_genes(graph_type="plot", 
                          plot_type="bar", 
                          solutions='all')

.. image:: https://user-images.githubusercontent.com/16560492/122477370-99106f80-cf95-11eb-8643-865b55e6b844.png
   :alt: 

.. _graphtypeboxplot:

``graph_type="boxplot"``
~~~~~~~~~~~~~~~~~~~~~~~~

By setting ``graph_type`` to ``"boxplot"``, then a box and whisker graph
is created. Now, the ``plot_type`` parameter has no effect.

The following 2 calls of the ``plot_genes()`` method create the same
figure as the default value for the ``solutions`` parameter is
``"all"``.

.. code:: python

   ga_instance.plot_genes(graph_type="boxplot")

   ga_instance.plot_genes(graph_type="boxplot", 
                          solutions='all')

.. image:: https://user-images.githubusercontent.com/16560492/122479260-beeb4380-cf98-11eb-8f08-23707929b12c.png
   :alt: 

.. _graphtypehistogram:

``graph_type="histogram"``
~~~~~~~~~~~~~~~~~~~~~~~~~~

For ``graph_type="boxplot"``, then a histogram is created for each gene.
Similar to ``graph_type="boxplot"``, the ``plot_type`` parameter has no
effect.

The following 2 calls of the ``plot_genes()`` method create the same
figure as the default value for the ``solutions`` parameter is
``"all"``.

.. code:: python

   ga_instance.plot_genes(graph_type="histogram")

   ga_instance.plot_genes(graph_type="histogram", 
                          solutions='all')

.. image:: https://user-images.githubusercontent.com/16560492/122477314-8007be80-cf95-11eb-9c95-da3f49204151.png
   :alt: 

All the previous figures can be created for only the best solutions by
setting ``solutions="best"``.
