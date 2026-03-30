import pygad
import random

num_generations = 100
sol_per_pop = 10
num_parents_mating = 5

def number_saved_solutions(keep_elitism=1, 
                           keep_parents=-1,
                           mutation_type="random",
                           mutation_percent_genes="default",
                           parent_selection_type='sss',
                           multi_objective=False,
                           fitness_batch_size=None,
                           save_solutions=False,
                           save_best_solutions=False):

    def fitness_func_no_batch_single(ga, solution, idx):
        return random.random()

    def fitness_func_no_batch_multi(ga_instance, solution, solution_idx):
        return [random.random(), random.random()]

    def fitness_func_batch_single(ga_instance, solution, solution_idx):
        f = []
        for sol in solution:
            f.append(random.random())
        return f

    def fitness_func_batch_multi(ga_instance, solution, solution_idx):
        f = []
        for sol in solution:
            f.append([random.random(), random.random()])
        return f

    if fitness_batch_size is None or (type(fitness_batch_size) in pygad.GA.supported_int_types and fitness_batch_size == 1):
        if multi_objective == True:
            fitness_func = fitness_func_no_batch_multi
        else:
            fitness_func = fitness_func_no_batch_single
    elif (type(fitness_batch_size) in pygad.GA.supported_int_types and fitness_batch_size > 1):
        if multi_objective == True:
            fitness_func = fitness_func_batch_multi
        else:
            fitness_func = fitness_func_batch_single

    ga_optimizer = pygad.GA(num_generations=num_generations,
                            sol_per_pop=sol_per_pop,
                            num_genes=6,
                            num_parents_mating=num_parents_mating,
                            fitness_func=fitness_func,
                            mutation_type=mutation_type,
                            parent_selection_type=parent_selection_type,
                            mutation_percent_genes=mutation_percent_genes,
                            keep_elitism=keep_elitism,
                            keep_parents=keep_parents,
                            suppress_warnings=True,
                            fitness_batch_size=fitness_batch_size,
                            save_best_solutions=save_best_solutions,
                            save_solutions=save_solutions)

    ga_optimizer.run()

    ga_optimizer.summary()

#### Single Objective
def test_save_solutions_default_keep():
    number_saved_solutions()


######## Batch Fitness
#### Single Objective
def test_save_solutions_no_keep_batch_1():
    number_saved_solutions(fitness_batch_size=1)


#### Multi Objective
def test_save_solutions_default_keep_multi_objective():
    number_saved_solutions(multi_objective=True)


#### Multi Objective
def test_save_solutions_no_keep_multi_objective_batch_1():
    number_saved_solutions(multi_objective=True,
                           fitness_batch_size=1)

#### Multi Objective NSGA-II Parent Selection
def test_save_solutions_default_keep_multi_objective_nsga2():
    number_saved_solutions(multi_objective=True,
    parent_selection_type='nsga2')

#### Multi Objective NSGA-II Parent Selection
def test_save_solutions_no_keep_multi_objective_nsga2_batch_1():
    number_saved_solutions(multi_objective=True,
    parent_selection_type='nsga2',
    fitness_batch_size=1)

if __name__ == "__main__":
    #### Single Objective
    print()
    test_save_solutions_default_keep()
    print()

    #### Multi-Objective
    print()
    test_save_solutions_default_keep_multi_objective()

    #### Multi-Objective NSGA-II Parent Selection
    print()
    test_save_solutions_default_keep_multi_objective_nsga2()

    ######## Batch Fitness Calculation
    #### Single Objective
    print()
    test_save_solutions_no_keep_batch_1()

    #### Multi-Objective
    print()
    test_save_solutions_no_keep_multi_objective_batch_1()

    #### Multi-Objective NSGA-II Parent Selection
    print()
    test_save_solutions_no_keep_multi_objective_nsga2_batch_1()
