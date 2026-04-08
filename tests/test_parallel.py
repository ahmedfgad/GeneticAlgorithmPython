import pygad
import numpy
import time

# Global constants for testing
num_generations = 5
num_parents_mating = 4
sol_per_pop = 10
num_genes = 3
random_seed = 42

def fitness_func(ga_instance, solution, solution_idx):
    # Simulate some work
    # time.sleep(0.01)
    return numpy.sum(solution**2)

def fitness_func_batch(ga_instance, solutions, indices):
    return [numpy.sum(s**2) for s in solutions]

def test_parallel_thread():
    """Test parallel_processing with 'thread' mode."""
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           parallel_processing=["thread", 2],
                           random_seed=random_seed,
                           suppress_warnings=True
                           )
    ga_instance.run()
    assert ga_instance.run_completed
    print("test_parallel_thread passed.")

def test_parallel_process():
    """Test parallel_processing with 'process' mode."""
    # Note: 'process' mode might be tricky in some environments (e.g. Windows without if __name__ == '__main__':)
    # But for a CI environment it should be tested.
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           parallel_processing=["process", 2],
                           random_seed=random_seed,
                           suppress_warnings=True
                           )
    ga_instance.run()
    assert ga_instance.run_completed
    print("test_parallel_process passed.")

def test_parallel_thread_batch():
    """Test parallel_processing with 'thread' mode and batch fitness."""
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func_batch,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           parallel_processing=["thread", 2],
                           fitness_batch_size=2,
                           random_seed=random_seed,
                           suppress_warnings=True
                           )
    ga_instance.run()
    assert ga_instance.run_completed
    print("test_parallel_thread_batch passed.")

if __name__ == "__main__":
    # For 'process' mode to work on Windows/macOS, we need this guard
    test_parallel_thread()
    test_parallel_process()
    test_parallel_thread_batch()
    print("\nAll parallel tests passed!")
