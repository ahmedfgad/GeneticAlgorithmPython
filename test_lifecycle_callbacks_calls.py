import pygad
import random

num_generations = 100

def number_lifecycle_callback_functions_calls(stop_criteria=None,
                                              on_generation_stop=None,
                                              crossover_type="single_point",
                                              mutation_type="random"):
    actual_num_callbacks_calls = 0

    def fitness_func(ga_instanse, solution, solution_idx):
        return random.random()

    def on_start(ga_instance):
        nonlocal actual_num_callbacks_calls
        actual_num_callbacks_calls = actual_num_callbacks_calls + 1

    def on_fitness(ga_instance, population_fitness):
        nonlocal actual_num_callbacks_calls
        actual_num_callbacks_calls = actual_num_callbacks_calls + 1

    def on_parents(ga_instance, selected_parents):
        nonlocal actual_num_callbacks_calls
        actual_num_callbacks_calls = actual_num_callbacks_calls + 1

    def on_crossover(ga_instance, offspring_crossover):
        nonlocal actual_num_callbacks_calls
        actual_num_callbacks_calls = actual_num_callbacks_calls + 1

    def on_mutation(ga_instance, offspring_mutation):
        nonlocal actual_num_callbacks_calls
        actual_num_callbacks_calls = actual_num_callbacks_calls + 1

    def on_generation(ga_instance):
        nonlocal actual_num_callbacks_calls
        actual_num_callbacks_calls = actual_num_callbacks_calls + 1
        
        if on_generation_stop:
            if ga_instance.generations_completed == on_generation_stop:
                return "stop"

    def on_stop(ga_instance, last_population_fitness):
        nonlocal actual_num_callbacks_calls
        actual_num_callbacks_calls = actual_num_callbacks_calls + 1

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=5,
                           fitness_func=fitness_func,
                           sol_per_pop=10,
                           num_genes=5,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           on_start=on_start,
                           on_fitness=on_fitness,
                           on_parents=on_parents,
                           on_crossover=on_crossover,
                           on_mutation=on_mutation,
                           on_generation=on_generation,
                           on_stop=on_stop,
                           stop_criteria=stop_criteria,
                           suppress_warnings=True)

    ga_instance.run()

    # The total number is:
        # 1 [for on_start()] +
        # num_generations [for on_fitness()] +
        # num_generations [for on_parents()] +
        # num_generations [for on_crossover()] +
        # num_generations [for on_mutation()] +
        # num_generations [for on_generation()] +
        # 1 [for on_stop()]
        # = 1 + num_generations * 5 + 1

    # Use 'generations_completed' instead of 'num_generations' because the evolution may stops in the on_generation() callback.
    expected_num_callbacks_calls = 1 + ga_instance.generations_completed * 5 + 1

    print(f"Expected {expected_num_callbacks_calls}.")
    print(f"Actual {actual_num_callbacks_calls}.")
    return actual_num_callbacks_calls, expected_num_callbacks_calls

def number_lifecycle_callback_methods_calls(stop_criteria=None,
                                            on_generation_stop=None,
                                            crossover_type="single_point",
                                            mutation_type="random"):
    actual_num_callbacks_calls = 0

    class Callbacks:
        def fitness_func(self, ga_instanse, solution, solution_idx):
            return 1

        def on_start(self, ga_instance):
            nonlocal actual_num_callbacks_calls
            actual_num_callbacks_calls = actual_num_callbacks_calls + 1

        def on_fitness(self, ga_instance, population_fitness):
            nonlocal actual_num_callbacks_calls
            actual_num_callbacks_calls = actual_num_callbacks_calls + 1

        def on_parents(self, ga_instance, selected_parents):
            nonlocal actual_num_callbacks_calls
            actual_num_callbacks_calls = actual_num_callbacks_calls + 1

        def on_crossover(self, ga_instance, offspring_crossover):
            nonlocal actual_num_callbacks_calls
            actual_num_callbacks_calls = actual_num_callbacks_calls + 1

        def on_mutation(self, ga_instance, offspring_mutation):
            nonlocal actual_num_callbacks_calls
            actual_num_callbacks_calls = actual_num_callbacks_calls + 1

        def on_generation(self, ga_instance):
            nonlocal actual_num_callbacks_calls
            actual_num_callbacks_calls = actual_num_callbacks_calls + 1

            if on_generation_stop:
                if ga_instance.generations_completed == on_generation_stop:
                    return "stop"

        def on_stop(self, ga_instance, last_population_fitness):
            nonlocal actual_num_callbacks_calls
            actual_num_callbacks_calls = actual_num_callbacks_calls + 1

    Callbacks_obj = Callbacks()
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=5,
                           fitness_func=Callbacks_obj.fitness_func,
                           sol_per_pop=10,
                           num_genes=5,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           on_start=Callbacks_obj.on_start,
                           on_fitness=Callbacks_obj.on_fitness,
                           on_parents=Callbacks_obj.on_parents,
                           on_crossover=Callbacks_obj.on_crossover,
                           on_mutation=Callbacks_obj.on_mutation,
                           on_generation=Callbacks_obj.on_generation,
                           on_stop=Callbacks_obj.on_stop,
                           stop_criteria=stop_criteria,
                           suppress_warnings=True)

    ga_instance.run()

    # The total number is:
        # 1 [for on_start()] +
        # num_generations [for on_fitness()] +
        # num_generations [for on_parents()] +
        # num_generations [for on_crossover()] +
        # num_generations [for on_mutation()] +
        # num_generations [for on_generation()] +
        # 1 [for on_stop()]
        # = 1 + num_generations * 5 + 1

    # Use 'generations_completed' instead of 'num_generations' because the evolution may stops in the on_generation() callback.
    expected_num_callbacks_calls = 1 + ga_instance.generations_completed * 5 + 1

    print(f"Expected {expected_num_callbacks_calls}.")
    print(f"Actual {actual_num_callbacks_calls}.")
    return actual_num_callbacks_calls, expected_num_callbacks_calls

def test_number_lifecycle_callback_functions_calls():
    actual, expected = number_lifecycle_callback_functions_calls()
    
    assert actual == expected

def test_number_lifecycle_callback_functions_calls_stop_criteria():
    actual, expected = number_lifecycle_callback_functions_calls(on_generation_stop=30)

    assert actual == expected

def test_number_lifecycle_callback_methods_calls():
    actual, expected = number_lifecycle_callback_methods_calls()
    
    assert actual == expected

def test_number_lifecycle_callback_methods_calls_stop_criteria():
    actual, expected = number_lifecycle_callback_methods_calls(on_generation_stop=30)

    assert actual == expected

def test_number_lifecycle_callback_functions_calls_no_crossover():
    actual, expected = number_lifecycle_callback_functions_calls(crossover_type=None)

    assert actual == expected

def test_number_lifecycle_callback_functions_calls_no_mutation():
    actual, expected = number_lifecycle_callback_functions_calls(mutation_type=None)

    assert actual == expected

def test_number_lifecycle_callback_functions_calls_no_crossover_no_mutation():
    actual, expected = number_lifecycle_callback_functions_calls(crossover_type=None,
                                                                 mutation_type=None)

    assert actual == expected

def test_number_lifecycle_callback_methods_calls_no_crossover():
    actual, expected = number_lifecycle_callback_methods_calls(crossover_type=None)

    assert actual == expected

def test_number_lifecycle_callback_methods_calls_no_mutation():
    actual, expected = number_lifecycle_callback_methods_calls(mutation_type=None)

    assert actual == expected

def test_number_lifecycle_callback_methods_calls_no_crossover_no_mutation():
    actual, expected = number_lifecycle_callback_methods_calls(crossover_type=None,
                                                               mutation_type=None)

    assert actual == expected

if __name__ == "__main__":
    print()
    test_number_lifecycle_callback_functions_calls()
    print()

    test_number_lifecycle_callback_functions_calls_stop_criteria()
    print()

    test_number_lifecycle_callback_methods_calls()
    print()

    test_number_lifecycle_callback_methods_calls_stop_criteria()
    print()

    test_number_lifecycle_callback_functions_calls_no_crossover()
    print()

    test_number_lifecycle_callback_functions_calls_no_crossover()
    print()

    test_number_lifecycle_callback_functions_calls_no_mutation()
    print()

    test_number_lifecycle_callback_functions_calls_no_crossover_no_mutation()
    print()

    test_number_lifecycle_callback_methods_calls_no_crossover()
    print()

    test_number_lifecycle_callback_methods_calls_no_mutation()
    print()

    test_number_lifecycle_callback_methods_calls_no_crossover_no_mutation()
    print()
