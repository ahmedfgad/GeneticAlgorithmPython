"""Numerical and lifecycle regressions for parallel fitness evaluation."""

import concurrent.futures
import functools
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys

import numpy
import pytest
import pygad
import pygad.utils.parallel


MODES = [None, ["thread", 2], ["process", 2]]


def scalar_fitness(ga, solution, index):
    return float(numpy.sum(solution))


def batch_fitness(ga, solutions, indices):
    return numpy.sum(solutions, axis=1).tolist()


def multi_fitness(ga, solution, index):
    value = scalar_fitness(ga, solution, index)
    return [value, -value]


def multi_batch_fitness(ga, solutions, indices):
    values = numpy.sum(solutions, axis=1)
    return numpy.column_stack((values, -values))


def make_ga(mode=None, **kwargs):
    options = dict(num_generations=2, num_parents_mating=2,
                   sol_per_pop=5, num_genes=4, fitness_func=scalar_fitness,
                   keep_elitism=0, keep_parents=0, random_seed=42,
                   suppress_warnings=True, parallel_processing=mode)
    options.update(kwargs)
    return pygad.GA(**options)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch_size", [None, 2, 3])
@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("retention", ["none", "parent", "all_parents", "elite"])
def test_adaptive_evaluates_offspring_and_retained_fitness(mode, batch_size, multi, retention):
    fitness = ((multi_batch_fitness if multi else batch_fitness)
               if batch_size else (multi_fitness if multi else scalar_fitness))
    ga = make_ga(mode, fitness_func=fitness, fitness_batch_size=batch_size,
                 mutation_type="adaptive", mutation_probability=[0.3, 0.1],
                 keep_parents=(1 if retention == "parent" else
                               -1 if retention == "all_parents" else 0),
                 keep_elitism=1 if retention == "elite" else 0)
    ga.population[:] = numpy.arange(1, 6)[:, None]
    ga.last_generation_fitness = ga.cal_pop_fitness()
    # Use a parent ordering that differs from the best retained solution.
    ga.last_generation_parents = ga.population[:2].copy()
    ga.last_generation_parents_indices = numpy.array([0, 1])
    offspring = numpy.full((ga.num_offspring, 4), 7.)
    before = ga.num_fitness_evaluations
    average, actual = ga.adaptive_mutation_population_fitness(offspring)
    expected = numpy.array([[28., -28.]] * ga.num_offspring) if multi else numpy.full(ga.num_offspring, 28.)
    numpy.testing.assert_allclose(actual, expected)
    if retention == "none":
        expected_average = expected.mean(axis=0)
    elif retention == "all_parents":
        kept_fitness = ga.last_generation_fitness[ga.last_generation_parents_indices]
        expected_average = numpy.concatenate((kept_fitness, expected)).mean(axis=0)
    else:
        kept, indices = ga.steady_state_selection(ga.last_generation_fitness, 1)
        expected_average = numpy.concatenate((ga.last_generation_fitness[indices], expected)).mean(axis=0)
    numpy.testing.assert_allclose(average, expected_average)
    assert ga.num_fitness_evaluations - before == ga.num_offspring


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch_size", [None, 2])
def test_adaptive_offspring_have_no_population_index(mode, batch_size):
    def fitness(ga, solution, index):
        if batch_size:
            return [100. if index is None else 200.] * len(solution)
        return 100. if index is None else 200.

    ga = make_ga(mode, fitness_func=fitness, fitness_batch_size=batch_size,
                 mutation_type="adaptive", mutation_probability=[0.3, 0.1])
    ga.last_generation_fitness = ga.cal_pop_fitness()
    ga.run_select_parents(call_on_parents=False)
    _, actual = ga.adaptive_mutation_population_fitness(numpy.ones((ga.num_offspring, 4)))
    numpy.testing.assert_array_equal(actual, 100.)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch_size", [None, 2])
def test_adaptive_preserves_fractional_fitness_after_integer_population(mode, batch_size):
    def fitness(ga, solution, index):
        if batch_size:
            return [28.5 if index is None else 4] * len(solution)
        return 28.5 if index is None else 4
    ga = make_ga(mode, fitness_func=fitness, fitness_batch_size=batch_size,
                 mutation_type="adaptive", mutation_probability=[0.3, 0.1])
    ga.last_generation_fitness = ga.cal_pop_fitness()
    assert ga.last_generation_fitness.dtype.kind == "i"
    ga.run_select_parents(call_on_parents=False)
    average, actual = ga.adaptive_mutation_population_fitness(numpy.ones((ga.num_offspring, 4)))
    numpy.testing.assert_array_equal(actual, 28.5)
    assert average == 28.5


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cache", ["solutions", "best_solutions", "parents", "elites"])
def test_cache_hits_preserve_values_without_dispatch(mode, cache, monkeypatch):
    ga = make_ga(mode)
    cached = numpy.arange(5) + 123.
    if cache == "solutions":
        ga.save_solutions = True
        ga.solutions = ga.population.tolist()
        ga.solutions_fitness = cached.tolist()
    elif cache == "best_solutions":
        ga.save_best_solutions = True
        ga.best_solutions = ga.population.copy()
        ga.best_solutions_fitness = cached.tolist()
    else:
        ga.previous_generation_fitness = cached
        if cache == "parents":
            ga.keep_parents = -1
            ga.last_generation_parents = ga.population.copy()
            ga.last_generation_parents_indices = numpy.arange(5)
        else:
            ga.keep_elitism = 5
            ga.last_generation_elitism = ga.population.copy()
            ga.last_generation_elitism_indices = numpy.arange(5)
    def no_pool():
        raise AssertionError("Cached solutions must not create an executor")
    monkeypatch.setattr(ga, "_fitness_pool", no_pool)
    numpy.testing.assert_array_equal(ga.cal_pop_fitness(), cached)
    assert ga.num_fitness_evaluations == 0


@pytest.mark.parametrize("mode", MODES)
def test_partial_cache_and_partial_batches_keep_population_indices(mode):
    def fitness(ga, solutions, indices):
        return [float(index) for index in indices]
    ga = make_ga(mode, fitness_func=fitness, fitness_batch_size=2,
                 save_best_solutions=True)
    ga.best_solutions = [ga.population[1].tolist()]
    ga.best_solutions_fitness = [123.]
    numpy.testing.assert_array_equal(ga.cal_pop_fitness(), [0., 123., 2., 3., 4.])
    assert ga.num_fitness_evaluations == 4


@pytest.mark.parametrize("mode", MODES)
def test_retained_elite_leaves_a_short_final_fitness_batch(mode):
    def fitness(ga, solutions, indices):
        assert len(solutions) in (10, 9)
        assert len(solutions) == len(indices)
        return numpy.sum(solutions, axis=1)

    population = numpy.arange(40, dtype=float).reshape(20, 2)
    ga = make_ga(mode, initial_population=population, num_genes=2,
                 fitness_func=fitness, fitness_batch_size=10, keep_elitism=1)
    expected = numpy.sum(population, axis=1)
    # Give the retained elite a distinct cached value to detect reevaluation.
    expected[0] = 999.
    ga.previous_generation_fitness = expected.copy()
    ga.last_generation_elitism = ga.population[:1].copy()
    ga.last_generation_elitism_indices = numpy.array([0])

    numpy.testing.assert_array_equal(ga.cal_pop_fitness(), expected)
    assert ga.num_fitness_evaluations == 19


@pytest.mark.parametrize("mode", MODES)
def test_adaptive_run_counts_real_evaluations(mode):
    ga = make_ga(mode, mutation_type="adaptive", mutation_probability=[0.3, 0.1])
    ga.run()
    # Initial population plus offspring before and after each mutation.
    assert ga.num_fitness_evaluations == 5 + 2 * 5 * ga.num_generations


@pytest.mark.parametrize("mode", MODES)
def test_evaluation_budget_includes_adaptive_fitness(mode):
    ga = make_ga(mode, num_generations=20, stop_criteria="evaluations_15",
                 mutation_type="adaptive", mutation_probability=[0.3, 0.1])
    ga.run()
    assert ga.generations_completed == 1
    assert ga.num_fitness_evaluations == 15


def _record_thread_pools(monkeypatch):
    pools = []
    original = concurrent.futures.ThreadPoolExecutor
    class RecordingPool(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.closed = False
            pools.append(self)
        def shutdown(self, *args, **kwargs):
            self.closed = True
            return super().shutdown(*args, **kwargs)
    monkeypatch.setattr(pygad.utils.parallel.concurrent.futures, "ThreadPoolExecutor", RecordingPool)
    return pools


@pytest.mark.parametrize("stop", [False, True])
def test_one_pool_per_run_including_adaptive_mutation_and_early_stop(monkeypatch, stop):
    pools = _record_thread_pools(monkeypatch)
    ga = make_ga(["thread", 2], mutation_type="adaptive", mutation_probability=[0.3, 0.1],
                 on_generation=(lambda ga: "stop") if stop else None)
    for expected in [1, 2]:
        ga.run()
        assert len(pools) == expected
        assert all(pool.closed for pool in pools)
        assert ga._fitness_executor is None


def test_direct_evaluation_uses_temporary_pools(monkeypatch):
    pools = _record_thread_pools(monkeypatch)
    ga = make_ga(["thread", 2])
    ga.cal_pop_fitness()
    ga.cal_pop_fitness()
    assert len(pools) == 2 and all(pool.closed for pool in pools)


@pytest.mark.parametrize("failure", ["fitness", "callback", "validation"])
def test_pool_cleanup_on_failure(monkeypatch, failure):
    pools = _record_thread_pools(monkeypatch)
    def fitness(ga, solution, index):
        if failure == "fitness":
            raise ValueError("fitness failed")
        return "invalid" if failure == "validation" else 1.
    def callback(ga):
        raise ValueError("callback failed")
    ga = make_ga(["thread", 2], fitness_func=fitness,
                 on_generation=callback if failure == "callback" else None)
    with pytest.raises(ValueError):
        ga.run()
    assert len(pools) == 1 and pools[0].closed
    assert ga._fitness_executor is None


@pytest.mark.parametrize("mode", [["thread", 2], ["process", 2]])
def test_checkpoint_inside_run_excludes_executor_and_resumes(mode, tmp_path):
    checkpoint = str(tmp_path / "checkpoint")
    def save(ga):
        ga.save(checkpoint)
    ga = make_ga(mode, on_generation=save)
    ga.run()
    restored = pygad.load(checkpoint)
    assert "_fitness_executor" not in restored.__dict__
    assert "_fitness_run_active" not in restored.__dict__
    restored.on_generation = None
    restored.run()
    assert restored.generations_completed == 4


def test_process_checkpoint_with_script_defined_fitness(tmp_path):
    script = tmp_path / "resume.py"
    script.write_text('''import numpy
import pygad

def fitness(ga, solution, index):
    return [float(numpy.sum(solution)), float(numpy.sum(solution * solution))]

if __name__ == "__main__":
    ga = pygad.GA(num_generations=2, num_parents_mating=2, sol_per_pop=4,
                  num_genes=4, fitness_func=fitness, parallel_processing=["process", 2],
                  save_best_solutions=True, suppress_warnings=True)
    ga.run()
    ga.save("checkpoint")
    restored = pygad.load("checkpoint")
    restored.run()
    assert restored.generations_completed == 4
    assert len(restored.best_solutions) == len(restored.best_solutions_fitness)
''')
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(pygad.__file__).resolve().parent.parent)
    result = subprocess.run([sys.executable, str(script)], cwd=str(tmp_path),
                            env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("kind", ["closure", "partial", "instance", "method"])
def test_process_dynamic_callables_and_current_round_state(kind):
    class Fitness:
        def __call__(self, ga, solution, index):
            return scalar_fitness(ga, solution, index) + ga.offset
        def method(self, ga, solution, index):
            return self(ga, solution, index)
    def closure(ga, solution, index):
        return scalar_fitness(ga, solution, index) + ga.offset
    def with_extra(extra, ga, solution, index):
        return scalar_fitness(ga, solution, index) + ga.offset + extra
    fitness = {"closure": closure, "partial": functools.partial(with_extra, 3),
               "instance": Fitness(), "method": Fitness().method}[kind]
    def update(ga):
        ga.offset = 20.
    ga = make_ga(["process", 2], fitness_func=fitness, on_generation=update,
                 crossover_type=None, mutation_type=None)
    ga.population[:] = 1.
    ga.offset = 0.
    ga.run()
    numpy.testing.assert_array_equal(ga.last_generation_fitness, 27. if kind == "partial" else 24.)


def test_process_tasks_do_not_share_mutated_ga_state():
    def fitness(ga, solution, index):
        value = ga.offset
        ga.offset += 1
        return value
    ga = make_ga(["process", 2], sol_per_pop=24, fitness_func=fitness)
    ga.offset = 42
    numpy.testing.assert_array_equal(ga.cal_pop_fitness(), 42)
    assert ga.offset == 42


@pytest.mark.parametrize("mode", MODES)
def test_all_elites_need_no_adaptive_evaluations(mode, monkeypatch):
    ga = make_ga(mode, keep_elitism=5, mutation_type="adaptive",
                 mutation_probability=[0.3, 0.1])
    ga.last_generation_fitness = numpy.array([1., 2., 3., 4., 5.])
    def no_pool():
        raise AssertionError("There are no offspring to evaluate")
    monkeypatch.setattr(ga, "_fitness_pool", no_pool)
    average, values = ga.adaptive_mutation_population_fitness(numpy.empty((0, 4)))
    assert average == 3.
    assert values.size == 0
    assert ga.num_fitness_evaluations == 0


def test_process_pool_cleanup_after_worker_error(monkeypatch):
    pools = []
    original = concurrent.futures.ProcessPoolExecutor
    class RecordingPool(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.closed = False
            pools.append(self)
        def shutdown(self, *args, **kwargs):
            self.closed = True
            return super().shutdown(*args, **kwargs)
    monkeypatch.setattr(pygad.utils.parallel.concurrent.futures, "ProcessPoolExecutor", RecordingPool)
    def fail(ga, solution, index):
        raise ValueError("worker failed")
    ga = make_ga(["process", 2], fitness_func=fail)
    with pytest.raises(ValueError, match="worker failed"):
        ga.run()
    assert len(pools) == 1 and pools[0].closed
    assert ga._fitness_executor is None


def test_callback_can_change_worker_count_and_disable_parallelism(monkeypatch):
    pools = _record_thread_pools(monkeypatch)
    def change(ga):
        ga.parallel_processing = ["thread", 3] if ga.generations_completed == 1 else None
    ga = make_ga(["thread", 2], num_generations=3, on_generation=change)
    ga.run()
    assert len(pools) == 2 and all(pool.closed for pool in pools)


@pytest.mark.parametrize("context", ["spawn", "fork"])
def test_process_transport_with_available_start_methods(monkeypatch, context):
    if context not in multiprocessing.get_all_start_methods():
        pytest.skip("Start method is unavailable on this platform")
    original = concurrent.futures.ProcessPoolExecutor
    monkeypatch.setattr(pygad.utils.parallel.concurrent.futures, "ProcessPoolExecutor",
                        functools.partial(original, mp_context=multiprocessing.get_context(context)))
    multiplier = 3
    def fitness(ga, solution, index):
        return multiplier * scalar_fitness(ga, solution, index)
    ga = make_ga(["process", 2], fitness_func=fitness)
    expected = 3 * numpy.sum(ga.population, axis=1)
    numpy.testing.assert_allclose(ga.cal_pop_fitness(), expected)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("bad_result", ["scalar_batch", "short_batch", "invalid_value"])
def test_result_validation_is_identical_in_every_mode(mode, bad_result):
    def fitness(ga, solutions, indices):
        if bad_result == "scalar_batch":
            return 1.
        if bad_result == "short_batch":
            return []
        return ["invalid"] * len(solutions)
    ga = make_ga(mode, fitness_func=fitness, fitness_batch_size=2)
    with pytest.raises(TypeError if bad_result == "scalar_batch" else ValueError):
        ga.cal_pop_fitness()
