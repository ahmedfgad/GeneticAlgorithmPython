import warnings

import numpy
import pytest

import pygad

# Global constants for testing
num_generations = 10
num_parents_mating = 5
sol_per_pop = 10
num_genes = 3
random_seed = 42

# A substring unique to the keep_parents/keep_elitism conflict warning.
CONFLICT_WARNING_SUBSTRING = "takes precedence"


def fitness_func(ga_instance, solution, solution_idx):
    """Single-objective fitness function."""
    return numpy.sum(solution ** 2)


def make_ga(**kwargs):
    """Build a GA instance with the shared defaults, overriding via kwargs."""
    params = dict(num_generations=num_generations,
                  num_parents_mating=num_parents_mating,
                  fitness_func=fitness_func,
                  sol_per_pop=sol_per_pop,
                  num_genes=num_genes,
                  random_seed=random_seed)
    params.update(kwargs)
    return pygad.GA(**params)


def conflict_warnings(record):
    """Return the keep_parents/keep_elitism conflict warnings in a record list."""
    return [w for w in record if CONFLICT_WARNING_SUBSTRING in str(w.message)]


def test_default_no_conflict_warning():
    """
    With neither keep_parents nor keep_elitism set, keep_parents resolves to its
    historical default (-1) and no conflict warning is raised.
    """
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        ga_instance = make_ga()

    assert ga_instance.keep_parents == -1
    assert ga_instance.keep_parents_explicitly_set is False
    assert ga_instance.keep_elitism == 1
    # Default keep_elitism=1, so num_offspring = sol_per_pop - keep_elitism.
    assert ga_instance.num_offspring == sol_per_pop - 1
    assert conflict_warnings(record) == []
    print("test_default_no_conflict_warning passed.")


def test_conflict_warning_fires():
    """
    Setting keep_parents while keep_elitism is at its default (1) raises the
    conflict warning instead of silently ignoring keep_parents.
    """
    with pytest.warns(UserWarning, match=CONFLICT_WARNING_SUBSTRING):
        make_ga(keep_parents=2)
    print("test_conflict_warning_fires passed.")


def test_no_warning_when_keep_parents_intended():
    """
    With keep_elitism=0, keep_parents takes effect and no conflict warning fires.
    Offspring count must reflect the kept parents.
    """
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        ga_instance = make_ga(keep_elitism=0, keep_parents=2)

    assert ga_instance.keep_parents == 2
    assert ga_instance.keep_parents_explicitly_set is True
    assert ga_instance.num_offspring == sol_per_pop - 2
    assert conflict_warnings(record) == []
    print("test_no_warning_when_keep_parents_intended passed.")


def test_keep_parents_minus_one_preserved():
    """
    The distinct keep_parents=-1 behavior (keep ALL selected parents) is preserved
    when keep_elitism=0: offspring count = sol_per_pop - num_parents_mating.
    """
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        ga_instance = make_ga(keep_elitism=0, keep_parents=-1)

    assert ga_instance.keep_parents == -1
    assert ga_instance.num_offspring == sol_per_pop - num_parents_mating
    assert conflict_warnings(record) == []
    print("test_keep_parents_minus_one_preserved passed.")


def test_suppress_warnings_honored():
    """
    suppress_warnings=True silences the conflict warning.
    """
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        make_ga(keep_parents=2, suppress_warnings=True)

    assert conflict_warnings(record) == []
    print("test_suppress_warnings_honored passed.")


def test_explicit_minus_one_with_elitism_warns():
    """
    Explicitly setting keep_parents=-1 while keep_elitism>0 still warns, because the
    user expressed an intent that the precedence rule overrides.
    """
    with pytest.warns(UserWarning, match=CONFLICT_WARNING_SUBSTRING):
        make_ga(keep_parents=-1)
    print("test_explicit_minus_one_with_elitism_warns passed.")


if __name__ == "__main__":
    test_default_no_conflict_warning()
    test_conflict_warning_fires()
    test_no_warning_when_keep_parents_intended()
    test_keep_parents_minus_one_preserved()
    test_suppress_warnings_honored()
    test_explicit_minus_one_with_elitism_warns()
    print("\nAll tests passed!")
