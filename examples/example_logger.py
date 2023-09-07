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
