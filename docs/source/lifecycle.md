# Life Cycle of PyGAD

The next figure shows the main steps in the life cycle of a `pygad.GA` instance. The genetic algorithm evaluates its initial population, then repeats parent selection, crossover, mutation, population update, and fitness evaluation for each generation. It can reuse cached fitness values. PyGAD stops when all generations are done, a stopping criterion is met, or the function passed to `on_generation` returns the string `stop`.

:::{figure} images/ga_lifecycle.*
:alt: The PyGAD genetic algorithm life cycle
:width: 480px
:align: center

The main steps of the genetic algorithm in PyGAD.
:::

The next figure shows the same life cycle in more detail, including the callback functions that PyGAD calls at each stage.

:::{figure} images/pygad_lifecycle.*
:alt: The PyGAD life cycle with callback functions
:width: 480px
:align: center

The PyGAD life cycle in detail, including the callback functions called at each stage.
:::

## Plotting the Configured Lifecycle

Call `plot_lifecycle()` to create a chart adjusted to the operators, callbacks, gene settings, and stopping conditions of a GA instance. It can be called before or after `run()`.

```python
ga_instance.plot_lifecycle()
```

The chart shows the generation loop and exit paths. It includes population replacement and fitness evaluation before `on_generation`, and omits disabled crossover or mutation. Configured callbacks still appear at their execution points, including `on_crossover` and `on_mutation` when their corresponding operators are disabled. The detailed `Stop Early?` block lists the configured stopping criteria and the possible `"stop"` return from `on_generation`, when that callback is supplied.

Use `show_parameters=False` for a compact chart, or save the figure by passing `save_dir`. The filename extension selects the output format.

```python
ga_instance.plot_lifecycle(title="PyGAD - My Optimization Problem",
                           save_dir="lifecycle.svg",
                           show=False)
```

Drawing the chart does not run the GA or call user functions. See {ref}`plot_lifecycle() <plot-lifecycle>` for the parameters, a sample chart, and a runnable example. To print a text description, use {ref}`summary() <print-lifecycle-summary>`.

:::{python-examples}
plots/example_plot_lifecycle.py
:::

(reporting-progress)=
## Reporting Progress

Use `on_generation` to report progress once a generation has completed. There is no need to change the fitness function or the GA operators:

```python
def on_generation(ga_instance):
    print(f"Generation: {ga_instance.generations_completed}")
    solution, fitness, solution_idx = ga_instance.best_solution(
        pop_fitness=ga_instance.last_generation_fitness)
    print(f"Best fitness in the current population: {fitness}")


ga_instance = pygad.GA(..., on_generation=on_generation)
ga_instance.run()
```

Passing `last_generation_fitness` avoids calculating fitness again just to print the result. For detailed tracing, use the stage callbacks in the complete example below. `on_start` runs before initial fitness evaluation. Each generation calls `on_fitness`, selects parents, applies crossover and mutation, updates the population, evaluates that population, and then calls `on_generation`. `on_stop` receives the final population fitness when the run completes normally or stops early.

`on_fitness` observes the fitness used for parent selection; `on_generation` observes the updated population and its fitness. The diagram shows this ordering, including evaluation after the population update.

## Tracing Every Stage

The next code implements all the callback functions to trace the execution of the genetic algorithm. Each callback function prints its name.

```python
import pygad
import numpy

function_inputs = [4,-2,3.5,5,-11,-4.7]
desired_output = 44

def fitness_func(ga_instance, solution, solution_idx):
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
```

Based on the used 3 generations as assigned to the `num_generations` argument, here is the output.

```
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
```

To stop from `on_generation`, return `"stop"`; otherwise no return value is needed.

:::{python-examples}
pygad_lifecycle.py
:::
