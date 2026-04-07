import pygad
import numpy
import random

# Global constants for testing
num_generations = 5
num_parents_mating = 4
sol_per_pop = 10
num_genes = 10
random_seed = 42

def fitness_func(ga_instance, solution, solution_idx):
    return numpy.sum(solution)

def fitness_func_multi(ga_instance, solution, solution_idx):
    return [numpy.sum(solution), numpy.sum(solution**2)]

def run_ga_with_params(parent_selection_type='sss', crossover_type='single_point', mutation_type='random', multi_objective=False):
    if multi_objective:
        f = fitness_func_multi
    else:
        f = fitness_func
        
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=f,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           parent_selection_type=parent_selection_type,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           random_seed=random_seed,
                           suppress_warnings=True)
    ga_instance.run()
    return ga_instance

def test_selection_operators():
    operators = ['sss', 'rws', 'sus', 'rank', 'random', 'tournament']
    for op in operators:
        ga = run_ga_with_params(parent_selection_type=op)
        # Verify parents were selected
        assert ga.last_generation_parents.shape == (num_parents_mating, num_genes)
        print(f"Selection operator '{op}' passed.")

def test_crossover_operators():
    operators = ['single_point', 'two_points', 'uniform', 'scattered']
    for op in operators:
        ga = run_ga_with_params(crossover_type=op)
        # Verify population shape
        assert ga.population.shape == (sol_per_pop, num_genes)
        print(f"Crossover operator '{op}' passed.")

def test_mutation_operators():
    operators = ['random', 'swap', 'inversion', 'scramble']
    for op in operators:
        ga = run_ga_with_params(mutation_type=op)
        # Verify population shape
        assert ga.population.shape == (sol_per_pop, num_genes)
        print(f"Mutation operator '{op}' passed.")

def test_multi_objective_selection():
    # NSGA-II is usually used for multi-objective
    ga = run_ga_with_params(parent_selection_type='nsga2', multi_objective=True)
    assert ga.last_generation_parents.shape == (num_parents_mating, num_genes)
    print("Multi-objective selection (nsga2) passed.")
    
    # Tournament NSGA-II
    ga = run_ga_with_params(parent_selection_type='tournament_nsga2', multi_objective=True)
    assert ga.last_generation_parents.shape == (num_parents_mating, num_genes)
    print("Multi-objective selection (tournament_nsga2) passed.")

if __name__ == "__main__":
    test_selection_operators()
    test_crossover_operators()
    test_mutation_operators()
    test_multi_objective_selection()
    print("\nAll operator tests passed!")
