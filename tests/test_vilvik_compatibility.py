"""Public PyGAD -> real Vilvik SDK contract, with only HTTP intercepted.

Install ``vilvik>=0.5.3 responses`` to run. No account or cloud job is created.
The dedicated compatibility workflow installs these dependencies explicitly.
"""
import copy
import json

import numpy as np
import pytest
import pygad

vilvik = pytest.importorskip("vilvik", reason="optional Vilvik integration")
responses = pytest.importorskip("responses", reason="HTTP interception for Vilvik tests")

BASE = "https://vilvik.invalid/api/v1"


def fitness(ga_instance, solution, solution_idx):
    return float(np.sum(solution))


def multi_fitness(ga_instance, solution, solution_idx):
    return [float(sum(solution)), -float(sum(value * value for value in solution))]


def generation_callback(ga_instance):
    ga_instance.compatibility_callback_calls = getattr(ga_instance, "compatibility_callback_calls", 0) + 1


def make_ga(**overrides):
    options = dict(num_generations=2, num_parents_mating=2, sol_per_pop=4,
                   num_genes=3, fitness_func=fitness, random_seed=21,
                   mutation_num_genes=1, suppress_warnings=True)
    options.update(overrides)
    return pygad.GA(**options)


def import_run(ga, **overrides):
    with responses.RequestsMock() as http:
        http.add(responses.POST, BASE + "/imports", status=201,
                 json={"id": "imported", "result_id": "result", "result_url": BASE + "/results/result"})
        record = ga.push_to_vilvik(api_key="vlk_test_compatibility", base_url=BASE, **overrides)
        assert record.id == "imported"
        assert record.result_id == "result"
        request = http.calls[0].request
        assert request.headers["Authorization"] == "Bearer vlk_test_compatibility"
        assert request.headers.get("Idempotency-Key")
        return json.loads(request.body)


def reconstruct(payload):
    """Execute exported source and constructor parameters as a consumer would."""
    params = copy.deepcopy(payload["ga_parameters"])
    code = payload["code"]
    for role in ("fitness_func", "on_start", "on_fitness", "on_parents",
                 "on_crossover", "on_mutation", "on_generation", "on_stop"):
        if role in code:
            namespace = {}
            exec(compile(code[role], "<exported-" + role + ">", "exec"), namespace)
            params[role] = namespace[code[role + "_entry"]]
    expression = params.pop("custom_gene_type", params["gene_type"])
    params["gene_type"] = eval(expression, {"__builtins__": {}, "float": float,
                                           "int": int, "numpy": np})
    params.pop("sol_per_pop")
    params.pop("num_genes")
    params["initial_population"] = payload["result"]["population"]
    params["num_generations"] = 1
    params["suppress_warnings"] = True
    return pygad.GA(**params)


@pytest.mark.parametrize("gene_type", [float, int, np.float32, np.int32,
                                     [float, 2], [int, [float, 2], np.float32]])
def test_completed_run_survives_real_sdk_export_and_continuation(gene_type):
    ga = make_ga(gene_type=gene_type)
    ga.run()
    population = ga.population.copy()
    payload = import_run(ga, name="compatibility", description="local run")
    json.dumps(payload, allow_nan=False)
    assert payload["name"] == "compatibility"
    assert payload["description"] == "local run"
    assert payload["origin"]["client"] == "pygad_wrapper"
    assert payload["origin"]["pygad_version"] == pygad.__version__
    assert payload["origin"]["sdk_version"] == vilvik.__version__
    result = payload["result"]
    best, best_fitness, best_index = ga.best_solution()
    np.testing.assert_allclose(result["best_solution"], np.asarray(best, dtype=float))
    assert result["best_solution_fitness"] == pytest.approx(float(best_fitness))
    assert result["best_solution_idx"] == best_index
    assert result["generations_completed"] == 2
    assert result["gene_type_single"] is ga.gene_type_single
    np.testing.assert_array_equal(ga.population, population)
    continued = reconstruct(payload)
    np.testing.assert_allclose(np.asarray(continued.population, dtype=float), np.asarray(population, dtype=float))
    assert continued.gene_type_single is ga.gene_type_single
    continued.run()
    assert continued.generations_completed == 1
    assert np.isfinite(continued.best_solution()[1])


@pytest.mark.parametrize("options,control", [
    ({"mutation_probability": 0.5}, "mutation_probability"),
    ({"mutation_percent_genes": 50, "mutation_num_genes": None}, "mutation_num_genes"),
    ({"mutation_type": "adaptive", "mutation_num_genes": [1, 2]}, "mutation_num_genes"),
])
def test_effective_mutation_controls_reconstruct(options, control):
    ga = make_ga(**options)
    ga.run()
    payload = import_run(ga)
    controls = {"mutation_probability", "mutation_percent_genes", "mutation_num_genes"}
    assert controls.intersection(payload["ga_parameters"]) == {control}
    reconstruct(payload).run()


@pytest.mark.parametrize("multi", [False, True])
def test_stop_criteria_and_objective_results_reconstruct(multi):
    options = dict(stop_criteria=["saturate_5", "reach_100000_100000" if multi else "reach_100000"])
    if multi:
        options.update(fitness_func=multi_fitness, parent_selection_type="nsga2")
    ga = make_ga(**options)
    ga.run()
    payload = import_run(ga)
    assert all(isinstance(stop, str) for stop in payload["ga_parameters"]["stop_criteria"])
    np.testing.assert_allclose(payload["result"]["last_generation_fitness"], ga.last_generation_fitness)
    np.testing.assert_allclose(payload["result"]["best_solution_fitness"], ga.best_solution()[1])
    reconstructed = reconstruct(payload)
    reconstructed.run()
    assert np.asarray(reconstructed.last_generation_fitness).shape == ((4, 2) if multi else (4,))


def test_callback_source_and_population_opt_out():
    ga = make_ga(on_generation=generation_callback)
    ga.run()
    payload = import_run(ga)
    continued = reconstruct(payload)
    continued.run()
    assert continued.compatibility_callback_calls == 1
    payload = import_run(ga, include_population=False)
    assert "population" not in payload["result"]
    assert "best_solution" in payload["result"]


def test_dry_run_requires_no_credentials_or_http(monkeypatch):
    monkeypatch.delenv("VILVIK_API_KEY", raising=False)
    with responses.RequestsMock():
        report = make_ga().push_to_vilvik(dry_run=True)
    assert report.ok()
    assert report.role("fitness_func").entry == "fitness"


def test_uncapturable_function_can_be_replaced_explicitly():
    ga = make_ga(fitness_func=lambda ga, solution, index: float(sum(solution)))
    ga.run()
    with responses.RequestsMock():
        with pytest.raises(vilvik.CaptureError):
            ga.push_to_vilvik(api_key="vlk_test_compatibility", base_url=BASE)
    payload = import_run(ga, fitness_source="def restored(ga, solution, index):\n    return float(sum(solution))\n",
                         fitness_entry="restored")
    reconstruct(payload).run()


@pytest.mark.parametrize("status,exception", [(401, vilvik.AuthenticationError), (422, vilvik.ValidationError)])
def test_sdk_errors_reach_pygad_caller(status, exception):
    ga = make_ga()
    ga.run()
    with responses.RequestsMock() as http:
        http.add(responses.POST, BASE + "/imports", status=status,
                 json={"error": {"code": "compatibility_error", "message": "rejected"}})
        with pytest.raises(exception) as error:
            ga.push_to_vilvik(client=vilvik.Client(api_key="vlk_test_compatibility", base_url=BASE, max_retries=0))
    assert error.value.status_code == status
