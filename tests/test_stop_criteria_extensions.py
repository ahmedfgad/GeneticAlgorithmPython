import time

import numpy
import pytest

import pygad


def _slow_fitness(ga, solution, sol_idx):
    # A small artificial delay so the "time" criterion can fire well
    # before all generations complete.
    time.sleep(0.005)
    return float(numpy.sum(solution))


def _fast_fitness(ga, solution, sol_idx):
    return float(numpy.sum(solution))


def test_max_evaluations_stops_run_after_budget_is_exhausted():
    # 50 solutions per population means the first generation alone
    # uses ~50 evaluations. With keep_elitism=1 and num_parents_mating=50,
    # subsequent generations evaluate only the new offspring, but the
    # total still climbs past 100 well before 20 generations complete.
    ga = pygad.GA(num_generations=20,
                  num_parents_mating=50,
                  fitness_func=_fast_fitness,
                  sol_per_pop=50,
                  num_genes=4,
                  stop_criteria="evaluations_100",
                  random_seed=1,
                  suppress_warnings=True)
    ga.run()
    assert ga.num_fitness_evaluations >= 100
    # The run should stop before completing all 20 generations.
    assert ga.generations_completed < 20


def test_max_evaluations_runs_full_when_budget_is_huge():
    # With a budget far larger than the total number of evaluations,
    # the criterion never fires and all generations complete.
    ga = pygad.GA(num_generations=5,
                  num_parents_mating=4,
                  fitness_func=_fast_fitness,
                  sol_per_pop=10,
                  num_genes=4,
                  stop_criteria="evaluations_10000",
                  random_seed=1,
                  suppress_warnings=True)
    ga.run()
    assert ga.generations_completed == 5


def test_max_time_stops_run_after_budget_is_exhausted():
    # The slow fitness function takes ~5 ms per call. With 30
    # solutions and 50 generations, the run would normally take
    # ~7.5 s. We cap at 0.2 s and expect the run to stop early.
    budget_seconds = 0.2
    ga = pygad.GA(num_generations=50,
                  num_parents_mating=30,
                  fitness_func=_slow_fitness,
                  sol_per_pop=30,
                  num_genes=4,
                  stop_criteria=f"time_{budget_seconds}",
                  random_seed=1,
                  suppress_warnings=True)
    start = time.monotonic()
    ga.run()
    elapsed = time.monotonic() - start
    assert ga.generations_completed < 50
    # The actual elapsed time can exceed the budget slightly because
    # the criterion is checked between generations. Allow a generous
    # 2-second ceiling so the test stays robust on slow CI runners.
    assert elapsed < 2.0


def test_max_time_runs_full_when_budget_is_huge():
    ga = pygad.GA(num_generations=3,
                  num_parents_mating=4,
                  fitness_func=_fast_fitness,
                  sol_per_pop=8,
                  num_genes=4,
                  stop_criteria="time_3600",
                  random_seed=1,
                  suppress_warnings=True)
    ga.run()
    assert ga.generations_completed == 3


def test_num_fitness_evaluations_resets_per_run_call():
    ga = pygad.GA(num_generations=3,
                  num_parents_mating=4,
                  fitness_func=_fast_fitness,
                  sol_per_pop=8,
                  num_genes=4,
                  random_seed=1,
                  suppress_warnings=True)
    ga.run()
    first_run_count = ga.num_fitness_evaluations
    assert first_run_count > 0
    ga.run()
    # Second call resets and counts only its own work; the count
    # should not be the cumulative total of the two runs.
    assert ga.num_fitness_evaluations < 2 * first_run_count


def test_unknown_stop_word_raises_value_error():
    with pytest.raises(ValueError, match="evaluations"):
        # Sanity check: the error message lists the new stop words.
        pygad.GA(num_generations=2,
                 num_parents_mating=2,
                 fitness_func=_fast_fitness,
                 sol_per_pop=4,
                 num_genes=2,
                 stop_criteria="bogus_5",
                 suppress_warnings=True)
