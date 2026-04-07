import numpy
import pygad
import random

# Global constants for testing
num_generations = 100
num_parents_mating = 5
sol_per_pop = 10
num_genes = 3
random_seed = 42

def fitness_func(ga_instance, solution, solution_idx):
    """Single-objective fitness function."""
    return numpy.sum(solution**2)

def fitness_func_multi(ga_instance, solution, solution_idx):
    """Multi-objective fitness function."""
    return [numpy.sum(solution**2), numpy.sum(solution)]

def test_best_solution_consistency_single_objective():
    """
    Test best_solution() consistency for single-objective optimization.
    """
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           random_seed=random_seed,
                           suppress_warnings=True
                           )
    ga_instance.run()

    # Call with last_generation_fitness
    sol1, fitness1, idx1 = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
    
    # Call without pop_fitness
    sol2, fitness2, idx2 = ga_instance.best_solution()

    assert numpy.array_equal(sol1, sol2)
    assert fitness1 == fitness2
    assert idx1 == idx2
    print("test_best_solution_consistency_single_objective passed.")

def test_best_solution_consistency_multi_objective():
    """
    Test best_solution() consistency for multi-objective optimization.
    """
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func_multi,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           random_seed=random_seed,
                           parent_selection_type="nsga2",
                           suppress_warnings=True
                           )
    ga_instance.run()

    # Call with last_generation_fitness
    sol1, fitness1, idx1 = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
    
    # Call without pop_fitness
    sol2, fitness2, idx2 = ga_instance.best_solution()

    assert numpy.array_equal(sol1, sol2)
    assert numpy.array_equal(fitness1, fitness2)
    assert idx1 == idx2
    print("test_best_solution_consistency_multi_objective passed.")

def test_best_solution_before_run():
    """
    Test best_solution() consistency before run() is called.
    """
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           random_seed=random_seed,
                           suppress_warnings=True
                           )
    
    # Before run(), last_generation_fitness is None
    # We can still call best_solution(), it should call cal_pop_fitness()
    sol2, fitness2, idx2 = ga_instance.best_solution()
    
    # Now cal_pop_fitness() should match ga_instance.best_solution() output if we pass it
    pop_fitness = ga_instance.cal_pop_fitness()
    sol1, fitness1, idx1 = ga_instance.best_solution(pop_fitness=pop_fitness)

    assert numpy.array_equal(sol1, sol2)
    assert fitness1 == fitness2
    assert idx1 == idx2
    print("test_best_solution_before_run passed.")

def test_best_solution_with_save_solutions():
    """
    Test best_solution() consistency when save_solutions=True.
    This tests the caching mechanism in cal_pop_fitness().
    """
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           random_seed=random_seed,
                           save_solutions=True,
                           suppress_warnings=True
                           )
    ga_instance.run()

    # Call with last_generation_fitness
    sol1, fitness1, idx1 = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
    
    # Call without pop_fitness (this will trigger cal_pop_fitness which uses saved solutions)
    sol2, fitness2, idx2 = ga_instance.best_solution()

    assert numpy.array_equal(sol1, sol2)
    assert fitness1 == fitness2
    assert idx1 == idx2
    print("test_best_solution_with_save_solutions passed.")

def test_best_solution_with_save_best_solutions():
    """
    Test best_solution() consistency when save_best_solutions=True.
    """
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           random_seed=random_seed,
                           save_best_solutions=True,
                           suppress_warnings=True
                           )
    ga_instance.run()

    # Call with last_generation_fitness
    sol1, fitness1, idx1 = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
    
    # Call without pop_fitness
    sol2, fitness2, idx2 = ga_instance.best_solution()

    assert numpy.array_equal(sol1, sol2)
    assert fitness1 == fitness2
    assert idx1 == idx2
    print("test_best_solution_with_save_best_solutions passed.")

def test_best_solution_with_keep_elitism():
    """
    Test best_solution() consistency when keep_elitism > 0.
    """
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           random_seed=random_seed,
                           keep_elitism=2,
                           suppress_warnings=True
                           )
    ga_instance.run()

    # Call with last_generation_fitness
    sol1, fitness1, idx1 = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
    
    # Call without pop_fitness
    sol2, fitness2, idx2 = ga_instance.best_solution()

    assert numpy.array_equal(sol1, sol2)
    assert fitness1 == fitness2
    assert idx1 == idx2
    print("test_best_solution_with_keep_elitism passed.")

def test_best_solution_with_keep_parents():
    """
    Test best_solution() consistency when keep_parents > 0.
    Note: keep_parents is ignored if keep_elitism > 0 (default is 1).
    So this tests the case where keep_parents is passed but effectively ignored by population update,
    yet we check if best_solution() still works consistently.
    """
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           random_seed=random_seed,
                           keep_parents=2,
                           suppress_warnings=True
                           )
    ga_instance.run()

    # Call with last_generation_fitness
    sol1, fitness1, idx1 = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
    
    # Call without pop_fitness
    sol2, fitness2, idx2 = ga_instance.best_solution()

    assert numpy.array_equal(sol1, sol2)
    assert fitness1 == fitness2
    assert idx1 == idx2
    print("test_best_solution_with_keep_parents passed.")

def test_best_solution_with_keep_parents_elitism_0():
    """
    Test best_solution() consistency when keep_parents > 0 and keep_elitism = 0.
    This ensures the 'keep_parents' logic in cal_pop_fitness is exercised.
    """
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           random_seed=random_seed,
                           keep_elitism=0,
                           keep_parents=2,
                           suppress_warnings=True
                           )
    ga_instance.run()

    # Call with last_generation_fitness
    sol1, fitness1, idx1 = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
    
    # Call without pop_fitness
    sol2, fitness2, idx2 = ga_instance.best_solution()

    assert numpy.array_equal(sol1, sol2)
    assert fitness1 == fitness2
    assert idx1 == idx2
    print("test_best_solution_with_keep_parents_elitism_0 passed.")

def test_best_solution_pop_fitness_validation():
    """
    Test validation of the pop_fitness parameter in best_solution().
    
    Note: num_generations=1 is used for speed as evolution is not needed.
    sol_per_pop=5 is used to provide a small population for testing invalid lengths.
    """
    ga_instance = pygad.GA(num_generations=1,
                           num_parents_mating=1,
                           fitness_func=fitness_func,
                           sol_per_pop=5,
                           num_genes=3,
                           suppress_warnings=True
                           )
    
    # Test invalid type
    try:
        ga_instance.best_solution(pop_fitness="invalid")
    except ValueError as e:
        assert "expected to be list, tuple, or numpy.ndarray" in str(e)
        print("Validation: Invalid type caught.")

    # Test invalid length
    try:
        ga_instance.best_solution(pop_fitness=[1, 2, 3]) # Length 3, but sol_per_pop is 5
    except ValueError as e:
        assert "must match the length of the 'self.population' attribute" in str(e)
        print("Validation: Invalid length caught.")

def test_best_solution_single_objective_tie():
    """
    Test best_solution() when there is a tie in fitness values.
    It should return the first solution with the maximum fitness.

    Note: sol_per_pop=5 must match the length of the manual pop_fitness array below.
    num_generations=1 is sufficient for testing selection logic.
    """
    ga_instance = pygad.GA(num_generations=1,
                           num_parents_mating=1,
                           fitness_func=fitness_func,
                           sol_per_pop=5,
                           num_genes=3,
                           suppress_warnings=True
                           )
    
    # Mock fitness with a tie at index 1 and 3
    pop_fitness = numpy.array([10, 50, 20, 50, 5])
    
    sol, fitness, idx = ga_instance.best_solution(pop_fitness=pop_fitness)
    
    assert fitness == 50
    assert idx == 1 # First occurrence
    print("test_best_solution_single_objective_tie passed.")

def test_best_solution_with_parallel_processing():
    """
    Test best_solution() with parallel_processing enabled.

    Note: num_generations=5 is used to ensure the initial population and first generation 
    trigger parallel fitness calculation.
    """
    ga_instance = pygad.GA(num_generations=5,
                           num_parents_mating=2,
                           fitness_func=fitness_func,
                           sol_per_pop=10,
                           num_genes=3,
                           random_seed=random_seed,
                           parallel_processing=["thread", 2],
                           suppress_warnings=True
                           )
    # best_solution() should work and trigger cal_pop_fitness() internally
    sol, fitness, idx = ga_instance.best_solution()
    assert sol is not None
    assert fitness is not None
    print("test_best_solution_with_parallel_processing passed.")

def test_best_solution_with_fitness_batch_size():
    """
    Test best_solution() with fitness_batch_size > 1.

    Note: num_generations=5 and sol_per_pop=10 provide enough work for batch processing.
    """
    def fitness_func_batch(ga_instance, solutions, indices):
        return [numpy.sum(s**2) for s in solutions]

    ga_instance = pygad.GA(num_generations=5,
                           num_parents_mating=2,
                           fitness_func=fitness_func_batch,
                           sol_per_pop=10,
                           num_genes=3,
                           random_seed=random_seed,
                           fitness_batch_size=2,
                           suppress_warnings=True
                           )
    
    sol, fitness, idx = ga_instance.best_solution()
    assert sol is not None
    assert fitness is not None
    print("test_best_solution_with_fitness_batch_size passed.")

def test_best_solution_pop_fitness_types():
    """
    Test best_solution() with different types for the pop_fitness parameter.

    Note: sol_per_pop=3 must match the length of fitness_vals below.
    num_generations=1 is sufficient for this type-check test.
    """
    ga_instance = pygad.GA(num_generations=1,
                           num_parents_mating=1,
                           fitness_func=fitness_func,
                           sol_per_pop=3,
                           num_genes=3,
                           suppress_warnings=True
                           )
    
    fitness_vals = [1.0, 5.0, 2.0]
    
    # Test list
    _, _, idx_list = ga_instance.best_solution(pop_fitness=fitness_vals)
    # Test tuple
    _, _, idx_tuple = ga_instance.best_solution(pop_fitness=tuple(fitness_vals))
    # Test numpy array
    _, _, idx_ndarray = ga_instance.best_solution(pop_fitness=numpy.array(fitness_vals))
    
    assert idx_list == idx_tuple == idx_ndarray == 1
    print("test_best_solution_pop_fitness_types passed.")

if __name__ == "__main__":
    test_best_solution_consistency_single_objective()
    test_best_solution_consistency_multi_objective()
    test_best_solution_before_run()
    test_best_solution_with_save_solutions()
    test_best_solution_with_save_best_solutions()
    test_best_solution_with_keep_elitism()
    test_best_solution_with_keep_parents()
    test_best_solution_with_keep_parents_elitism_0()
    test_best_solution_pop_fitness_validation()
    test_best_solution_single_objective_tie()
    test_best_solution_with_parallel_processing()
    test_best_solution_with_fitness_batch_size()
    test_best_solution_pop_fitness_types()
    print("\nAll tests passed!")
