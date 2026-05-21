# Use Functions and Methods to Build Fitness and Callbacks

In PyGAD 2.19.0, it is possible to pass user-defined functions or methods to the following parameters:

1. `fitness_func`
2. `on_start`
3. `on_fitness`
4. `on_parents`
5. `on_crossover`
6. `on_mutation`
7. `on_generation`
8. `on_stop`

This section gives 2 examples of how to build these handlers using:

1. Functions.
2. Methods.

## Assign Functions

This is a dummy example where the fitness function returns a random value. Note that the instance of the `pygad.GA` class is passed as the last parameter of all functions.

```python
import pygad
import numpy

def fitness_func(ga_instance, solution, solution_idx):
    return numpy.random.rand()

def on_start(ga_instance):
    print("on_start")

def on_fitness(ga_instance, last_gen_fitness):
    print("on_fitness")

def on_parents(ga_instance, last_gen_parents):
    print("on_parents")

def on_crossover(ga_instance, last_gen_offspring):
    print("on_crossover")

def on_mutation(ga_instance, last_gen_offspring):
    print("on_mutation")

def on_generation(ga_instance):
    print("on_generation\n")

def on_stop(ga_instance, last_gen_fitness):
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
```

## Assign Methods

The next example has all the methods defined inside the class `Test`. All of the methods accept an additional parameter representing the method's object of the class `Test`.

All methods accept `self` as the first parameter and the instance of the `pygad.GA` class as the last parameter.

```python
import pygad
import numpy

class Test:
    def fitness_func(self, ga_instance, solution, solution_idx):
        return numpy.random.rand()

    def on_start(self, ga_instance):
        print("on_start")

    def on_fitness(self, ga_instance, last_gen_fitness):
        print("on_fitness")

    def on_parents(self, ga_instance, last_gen_parents):
        print("on_parents")

    def on_crossover(self, ga_instance, last_gen_offspring):
        print("on_crossover")

    def on_mutation(self, ga_instance, last_gen_offspring):
        print("on_mutation")

    def on_generation(self, ga_instance):
        print("on_generation\n")

    def on_stop(self, ga_instance, last_gen_fitness):
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
```
