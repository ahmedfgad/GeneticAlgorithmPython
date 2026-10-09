"""Fitness validation, saved history, and repeated-run regression tests."""

import cloudpickle
import matplotlib
import numpy
import pytest

import pygad

matplotlib.use('Agg')


def fitness_func(ga_instance, solution, solution_index):
    return int(solution[0])


def increasing_mutation(offspring, ga_instance):
    return offspring + 10


def make_ga(**kwargs):
    parameters = dict(num_generations=2, num_parents_mating=1,
                      initial_population=[[0], [1]], gene_type=int,
                      fitness_func=fitness_func, crossover_type=None,
                      mutation_type=increasing_mutation, keep_parents=0,
                      keep_elitism=0, suppress_warnings=True, random_seed=7)
    parameters.update(kwargs)
    return pygad.GA(**parameters)


@pytest.mark.parametrize('save_best', [False, True])
@pytest.mark.parametrize('save_all', [False, True])
def test_repeated_runs_preserve_snapshots_and_actual_generation_numbers(save_best, save_all):
    ga = make_ga(save_best_solutions=save_best, save_solutions=save_all)
    ga.run()
    first_fitness = numpy.asarray(ga.best_solutions_fitness).copy()
    ga.run()
    assert ga.generations_completed == 4
    assert ga.best_solution_generation == 4
    assert ga.best_solutions_generations == [0, 1, 2, 2, 3, 4]
    numpy.testing.assert_array_equal(ga.best_solutions_fitness[:3], first_fitness)
    assert len(ga.best_solutions_fitness) == 6
    assert len(ga.best_solutions) == (6 if save_best else 0)
    if save_all:
        assert ga.solutions_generations == [0, 1, 2, 2, 3, 4]
        assert len(ga.solutions) == len(ga.solutions_fitness) == 12
        for solution, fitness in zip(ga.solutions, ga.solutions_fitness):
            assert solution[0] == fitness
    if save_best:
        numpy.testing.assert_array_equal(ga.best_solutions[:, 0], ga.best_solutions_fitness)


def test_zero_generation_runs_keep_boundary_snapshots():
    ga = make_ga(num_generations=0, save_best_solutions=True, save_solutions=True)
    ga.run()
    ga.run()
    assert ga.generations_completed == ga.best_solution_generation == 0
    assert ga.best_solutions_generations == ga.solutions_generations == [0, 0]
    assert len(ga.best_solutions) == 2
    assert len(ga.solutions) == 4


def test_early_stop_and_resume_track_absolute_generations():
    def stop(ga):
        return 'stop'
    ga = make_ga(on_generation=stop, save_solutions=True)
    ga.run()
    ga.run()
    assert ga.generations_completed == ga.best_solution_generation == 2
    assert ga.best_solutions_generations == ga.solutions_generations == [0, 1, 1, 2]


@pytest.mark.parametrize('in_place', [False, True])
def test_on_fitness_updates_saved_best_and_population_fitness(in_place):
    calls = []
    def on_fitness(ga, fitness):
        calls.append(ga.generations_completed)
        if in_place:
            fitness[:] = [100, 0]
        else:
            return [100, 0]
    ga = make_ga(on_fitness=on_fitness, save_best_solutions=True, save_solutions=True)
    ga.run()
    assert calls == [0, 1]
    for snapshot in range(2):
        assert ga.best_solutions_fitness[snapshot] == 100
        assert ga.best_solutions[snapshot, 0] == ga.solutions[snapshot * 2][0]
        assert ga.solutions_fitness[snapshot * 2:snapshot * 2 + 2] == [100, 0]


@pytest.mark.parametrize('sequence,window,expected', [
    ([0, 1, 2, 3, 4, 5], 1, 5),
    ([1, 1, 1, 1, 1, 1], 1, 1),
    ([1, 1, 1, 1, 1, 1], 3, 3),
    ([0, 1, 1, 1, 1, 1], 2, 3),
    ([0, 1, 0, 1, 0, 1], 2, 5),
    ([0, 0, 1, 1, 1, 1], 2, 4),
])
@pytest.mark.parametrize('multi_objective', [False, True])
def test_saturation_counts_consecutive_unchanged_generations(sequence, window, expected, multi_objective):
    def scheduled_fitness(ga, solution, index):
        value = sequence[ga.generations_completed]
        return [value, -value] if multi_objective else value
    ga = make_ga(fitness_func=scheduled_fitness, num_generations=5,
                 stop_criteria=f'saturate_{window}')
    ga.run()
    assert ga.generations_completed == expected


def test_saturation_counter_resets_for_each_run():
    ga = make_ga(fitness_func=lambda ga, solution, index: 1,
                 num_generations=10, stop_criteria='saturate_3')
    ga.run()
    assert ga.generations_completed == 3
    ga.run()
    assert ga.generations_completed == 6
    assert ga.best_solution_generation == 0


@pytest.mark.parametrize('mode', [None, ['thread', 2], ['process', 2]])
@pytest.mark.parametrize('batch_size', [None, 2])
@pytest.mark.parametrize('invalid', [[], [[1], [2]], 'bad', [1, 'bad'], [1, None],
                                   numpy.nan, [1, numpy.nan], [1, numpy.inf], [[1], [2, 3]], object()])
def test_invalid_fitness_is_rejected_before_selection(mode, batch_size, invalid):
    def invalid_fitness(ga, solution, index):
        return [invalid] * len(solution) if batch_size else invalid
    ga = make_ga(fitness_func=invalid_fitness, parallel_processing=mode,
                 fitness_batch_size=batch_size)
    with pytest.raises(ValueError, match='fitness_func'):
        ga.cal_pop_fitness()
    assert getattr(ga, '_fitness_executor', None) is None


@pytest.mark.parametrize('batch_size', [None, 2])
def test_inconsistent_objective_counts_are_rejected(batch_size):
    def inconsistent_fitness(ga, solution, index):
        if batch_size:
            return [1, [1, 2]]
        return 1 if index == 0 else [1, 2]
    ga = make_ga(fitness_func=inconsistent_fitness, fitness_batch_size=batch_size)
    with pytest.raises(ValueError, match='same fitness shape'):
        ga.cal_pop_fitness()


@pytest.mark.parametrize('in_place', [False, True])
def test_invalid_callback_fitness_is_rejected(in_place):
    def invalid_callback(ga, fitness):
        if in_place:
            fitness[:] = numpy.nan
        else:
            return ['bad', 'bad']
    ga = make_ga(gene_type=float, fitness_func=lambda ga, solution, index: float(solution[0]),
                 on_fitness=invalid_callback)
    with pytest.raises(ValueError, match='on_fitness'):
        ga.run()


def test_scalar_infinity_is_supported_by_best_solution_and_saturation():
    ga = make_ga(fitness_func=lambda ga, solution, index: numpy.inf,
                 stop_criteria='saturate_2', num_generations=5)
    ga.run()
    assert ga.generations_completed == 2
    assert ga.best_solution()[1] == numpy.inf


def test_checkpoint_resume_matches_uninterrupted_repeated_runs():
    ga = make_ga(save_best_solutions=True, save_solutions=True, mutation_type='random')
    ga.run()
    resumed = cloudpickle.loads(cloudpickle.dumps(ga))
    assert not hasattr(resumed, '_saved_fitness_indexes')
    ga.run()
    resumed.run()
    numpy.testing.assert_array_equal(resumed.population, ga.population)
    numpy.testing.assert_array_equal(resumed.best_solutions, ga.best_solutions)
    assert resumed.best_solutions_generations == ga.best_solutions_generations
    assert resumed.solutions_generations == ga.solutions_generations
    assert resumed.best_solution_generation == ga.best_solution_generation


@pytest.mark.parametrize('runs', [1, 2])
@pytest.mark.parametrize('array_history', [False, True])
def test_older_checkpoints_restore_available_generation_information(runs, array_history):
    ga = make_ga(save_solutions=True)
    for _ in range(runs):
        ga.run()
    state = ga.__getstate__()
    if array_history:
        state['best_solutions_fitness'] = numpy.asarray(state['best_solutions_fitness'])
    for name in ['best_solutions_generations', 'solutions_generations', '_saved_population_sizes']:
        state.pop(name)
    restored = pygad.GA.__new__(pygad.GA)
    restored.__setstate__(state)
    if runs == 1:
        assert restored.best_solutions_generations == [0, 1, 2]
        assert restored.best_solution_generation == 2
    else:
        assert restored.best_solutions_generations == [None] * 6
        assert restored.best_solution_generation == -1
    restored.run()
    assert restored.best_solution_generation == restored.generations_completed


def test_clearing_public_histories_keeps_new_generation_numbers_correct():
    ga = make_ga(save_solutions=True, save_best_solutions=True)
    ga.run()
    ga.best_solutions = []
    ga.best_solutions_fitness = []
    ga.solutions = []
    ga.solutions_fitness = []
    ga.run()
    assert ga.best_solutions_generations == ga.solutions_generations == [2, 3, 4]
    assert ga.best_solution_generation == 4


def test_cache_precedence_first_match_and_public_history_edits():
    ga = make_ga(save_solutions=True, save_best_solutions=True)
    ga.solutions = [[0], [0], [1]]
    ga.solutions_fitness = [12, 99, 13]
    ga.best_solutions = [[0], [1]]
    ga.best_solutions_fitness = [100, 101]
    numpy.testing.assert_array_equal(ga.cal_pop_fitness(), [12, 13])
    ga.solutions_fitness[0] = 42
    ga.solutions[2][0] = 2
    numpy.testing.assert_array_equal(ga.cal_pop_fitness(), [42, 101])
    assert ga.num_fitness_evaluations == 0


def test_incremental_cache_only_indexes_new_saved_solutions():
    class CountingHistory(list):
        reads = 0
        def __getitem__(self, index):
            self.reads += 1
            return super().__getitem__(index)
    ga = make_ga(save_solutions=True)
    ga.solutions = CountingHistory([[index] for index in range(1000)])
    ga.solutions_fitness = list(range(1000))
    ga._fitness_run_active = True
    ga.cal_pop_fitness()
    assert ga.solutions.reads == 1000
    ga.cal_pop_fitness()
    assert ga.solutions.reads == 1000
    ga.solutions.append([1000])
    ga.solutions_fitness.append(1000)
    ga.cal_pop_fitness()
    assert ga.solutions.reads == 1001


@pytest.mark.parametrize('plot_type', ['plot', 'scatter', 'bar'])
def test_repeated_run_fitness_and_gene_plots_use_generation_numbers(plot_type, monkeypatch):
    from matplotlib import pyplot
    monkeypatch.setattr(pyplot, 'show', lambda: None)
    ga = make_ga(save_best_solutions=True)
    ga.run()
    ga.run()
    expected = [0, 1, 2, 2, 3, 4]
    for figure in [ga.plot_fitness(plot_type=plot_type),
                   ga.plot_genes(solutions='best', plot_type=plot_type)]:
        axis = figure.axes[0]
        if plot_type == 'plot':
            actual = axis.lines[0].get_xdata()
        elif plot_type == 'scatter':
            actual = axis.collections[0].get_offsets()[:, 0]
        else:
            actual = [bar.get_x() + bar.get_width() / 2 for bar in axis.patches]
        numpy.testing.assert_array_equal(actual, expected)
        pyplot.close(figure)


def test_repeated_run_population_plots_and_new_solution_rate(monkeypatch):
    from matplotlib import pyplot
    monkeypatch.setattr(pyplot, 'show', lambda: None)
    ga = make_ga(save_solutions=True)
    ga.run()
    ga.run()
    for figure in [ga.plot_fitness_band(), ga.plot_population_diversity()]:
        numpy.testing.assert_array_equal(figure.axes[0].lines[0].get_xdata(), [0, 1, 2, 2, 3, 4])
        pyplot.close(figure)
    figure = ga.plot_new_solution_rate()
    numpy.testing.assert_array_equal(figure.axes[0].lines[0].get_xdata(), [0, 1, 2, 3])
    numpy.testing.assert_array_equal(figure.axes[0].lines[0].get_ydata(), [2, 2, 1, 1])
    pyplot.close(figure)


def test_population_snapshot_sizes_survive_nsga3_growth_between_runs():
    ga = make_ga(save_solutions=True, fitness_func=lambda ga, solution, index: [int(solution[0])] * 3)
    ga.run()
    ga.parent_selection_type = 'nsga3'
    ga.nsga3_num_divisions = 2
    # Change the configured operator consistently, as constructor validation does.
    ga.select_parents = ga.nsga3_selection
    ga.run()
    assert ga._saved_population_sizes == [2, 2, 2, 6, 6, 6]
    assert [len(population) for population in ga._per_generation_solutions()] == [2, 2, 2, 6, 6, 6]
    assert [len(fitness) for fitness in ga._per_generation_fitness()] == [2, 2, 2, 6, 6, 6]


def test_repeated_multi_objective_history_does_not_overwrite_current_pareto_fronts(monkeypatch):
    from matplotlib import pyplot
    monkeypatch.setattr(pyplot, 'show', lambda: None)
    ga = make_ga(save_solutions=True, save_best_solutions=True,
                 fitness_func=lambda ga, solution, index: [int(solution[0]), -int(solution[0])])
    ga.run()
    ga.run()
    assert 0 <= ga.best_solution_generation <= ga.generations_completed
    assert len(ga.pareto_fronts[0]) == len(ga.population)
    assert all(int(row[0]) < len(ga.population) for row in ga.pareto_fronts[0])
    figure = ga.plot_non_dominated_hypervolume(reference_point=[-100, -100])
    numpy.testing.assert_array_equal(figure.axes[0].lines[0].get_xdata(), [0, 1, 2, 2, 3, 4])
    pyplot.close(figure)
    figure = ga.plot_pareto_front_evolution(every_k=2)
    assert figure.axes[0].get_legend_handles_labels()[1] == ['gen 0', 'gen 2', 'gen 4']
    pyplot.close(figure)


def test_report_after_repeated_runs_includes_history_plots(tmp_path):
    pytest.importorskip('reportlab')
    ga = make_ga(save_solutions=True, save_best_solutions=True)
    ga.run()
    ga.run()
    filename = ga.generate_report(str(tmp_path / 'repeated_runs'),
                                  include_plots=['plot_fitness', 'plot_genes',
                                                 'plot_new_solution_rate', 'plot_fitness_band'])
    assert (tmp_path / 'repeated_runs.pdf').stat().st_size > 1000
    assert filename.endswith('.pdf')
    assert ga.best_solution_generation == 4


def test_cache_keeps_large_integer_solution_keys_exact():
    large_integer = 2 ** 60
    ga = make_ga(initial_population=[[large_integer], [large_integer + 1]],
                 save_solutions=True)
    ga.solutions = [[large_integer], [large_integer + 1]]
    ga.solutions_fitness = [12, 13]
    numpy.testing.assert_array_equal(ga.cal_pop_fitness(), [12, 13])
    assert ga.num_fitness_evaluations == 0


def test_cache_observes_history_edits_made_by_callbacks():
    def edit_history(ga):
        if ga.generations_completed == 1:
            ga.solutions[0][0] = int(ga.population[0, 0])
            ga.solutions_fitness[0] = 999
    ga = make_ga(save_solutions=True, on_generation=edit_history,
                 crossover_type=None, mutation_type=None)
    ga.run()
    assert 999 in ga.last_generation_fitness


@pytest.mark.parametrize('fitness', [[[], []], [[[1]], [[2]]], [1, numpy.nan],
                                    [[1], [1, 2]], ['bad', 'bad'], numpy.array(1)])
def test_best_solution_validates_explicit_fitness(fitness):
    ga = make_ga()
    with pytest.raises(ValueError, match='pop_fitness'):
        ga.best_solution(pop_fitness=fitness)


def test_adaptive_fitness_uses_the_population_objective_count():
    def fitness(ga, solution, index):
        return [1, 2] if index is None else 1
    ga = make_ga(fitness_func=fitness, mutation_type='adaptive',
                 mutation_num_genes=[1, 1])
    ga.last_generation_fitness = ga.cal_pop_fitness()
    ga.run_select_parents(call_on_parents=False)
    with pytest.raises(ValueError, match='same fitness shape'):
        ga.adaptive_mutation_population_fitness(numpy.ones((2, 1)))
