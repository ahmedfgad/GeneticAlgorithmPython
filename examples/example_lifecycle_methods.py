import pygad
import numpy

"""
Use a method to build the lifecycle.
"""

class GAOperations:
    def fitness_func(self, ga_instance, solution, solution_idx):
        fitness = numpy.sum(solution)
        return fitness

    def crossover(self, parents, offspring_size, ga_instance):
        return numpy.random.rand(offspring_size[0], offspring_size[1])

    def mutation(self, offspring, ga_instance):
        return offspring

class Lifecycle:
    def on_start(self, ga_instance):
        print("on_start")

    def on_fitness(self, ga_instance, fitness):
        print("on_fitness")

    def on_crossover(self, ga_instance, offspring):
        print("on_crossover")

    def on_mutation(self, ga_instance, offspring):
        print("on_mutation")

    def on_parents(self, ga_instance, parents):
        print("on_parents")

    def on_generation(self, ga_instance):
        print("on_generation")

    def on_stop(self, ga_instance, fitness):
        print("on_stop")

ga_obj = GAOperations()
lifecycle_obj = Lifecycle()

num_generations = 10 # Number of generations.
num_parents_mating = 5 # Number of solutions to be selected as parents in the mating pool.

sol_per_pop = 10 # Number of solutions in the population.
num_genes = 5

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,

                       fitness_func=ga_obj.fitness_func,

                       crossover_type=ga_obj.crossover,
                       mutation_type=ga_obj.mutation,

                       on_start=lifecycle_obj.on_start,
                       on_fitness=lifecycle_obj.on_fitness,
                       on_crossover=lifecycle_obj.on_crossover,
                       on_mutation=lifecycle_obj.on_mutation,
                       on_parents=lifecycle_obj.on_parents,
                       on_generation=lifecycle_obj.on_generation,
                       on_stop=lifecycle_obj.on_stop,
                       
                       suppress_warnings=True)

# Running the GA to optimize the parameters of the function.
ga_instance.run()

ga_instance.plot_fitness()
