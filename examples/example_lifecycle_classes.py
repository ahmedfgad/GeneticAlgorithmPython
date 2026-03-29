import pygad
import numpy

"""
Use a method to build the lifecycle.
"""

class Fitness:
    def __call__(self, ga_instance, solution, solution_idx):
        fitness = numpy.sum(solution)
        return fitness

class Crossover:
    def __call__(self, parents, offspring_size, ga_instance):
        return numpy.random.rand(offspring_size[0], offspring_size[1])

class Mutation:
    def __call__(self, offspring, ga_instance):
        return offspring

class OnStart:
    def __call__(self, ga_instance):
        print("on_start")

class OnFitness:
    def __call__(self, ga_instance, fitness):
        print("on_fitness")

class OnCrossover:
    def __call__(self, ga_instance, offspring):
        print("on_crossover")

class OnMutation:
    def __call__(self, ga_instance, offspring):
        print("on_mutation")

class OnParents:
    def __call__(self, ga_instance, parents):
        print("on_parents")

class OnGeneration:
    def __call__(self, ga_instance):
        print("on_generation")

class OnStop:
    def __call__(self, ga_instance, fitness):
        print("on_stop")

num_generations = 10 # Number of generations.
num_parents_mating = 5 # Number of solutions to be selected as parents in the mating pool.

sol_per_pop = 10 # Number of solutions in the population.
num_genes = 5

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,

                       fitness_func=Fitness(),

                       crossover_type=Crossover(),
                       mutation_type=Mutation(),

                       on_start=OnStart(),
                       on_fitness=OnFitness(),
                       on_crossover=OnCrossover(),
                       on_mutation=OnMutation(),
                       on_parents=OnParents(),
                       on_generation=OnGeneration(),
                       on_stop=OnStop(),
                       
                       suppress_warnings=True)

# Running the GA to optimize the parameters of the function.
ga_instance.run()

ga_instance.plot_fitness()
