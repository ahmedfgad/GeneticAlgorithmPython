# More About PyGAD

This section covers the more advanced features of the `pygad` module. Pick a topic:

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Multi-Objective Optimization
:link: multi_objective
:link-type: doc

Optimize several objectives at once using NSGA-II.
:::

:::{grid-item-card} Controlling Gene Values
:link: gene_values
:link-type: doc

Restrict gene values with `gene_space`, `gene_type`, constraints, `sample_size`, and duplicate prevention.
:::

:::{grid-item-card} Controlling Generations
:link: generations
:link-type: doc

Elitism, stopping criteria, random seed, saving and continuing, and population size.
:::

:::{grid-item-card} Fitness Calculation and Performance
:link: fitness_calculation
:link-type: doc

Parallel processing, batch fitness, reusing fitness, and non-deterministic problems.
:::

:::{grid-item-card} Logging and the Lifecycle Summary
:link: logging
:link-type: doc

Print a Keras-like summary and log the outputs.
:::

:::{grid-item-card} User-Defined Functions, Methods, and Classes
:link: custom_functions
:link-type: doc

Pass your own functions, methods, or classes for the fitness and callbacks.
:::

::::

:::{toctree}
:hidden:

multi_objective
gene_values
generations
fitness_calculation
logging
custom_functions
:::
