import sys
import types
import pytest
import pygad


def _ga():
    return pygad.GA(
        num_generations=1, num_parents_mating=2, sol_per_pop=4, num_genes=3,
        fitness_func=lambda ga_instance, solution, solution_idx: 0.0)


def test_push_to_vilvik_forwards_to_sdk(monkeypatch):
    calls = {}
    fake = types.ModuleType("vilvik")

    def fake_push(ga, **kw):
        calls["ga"] = ga
        calls["kw"] = kw
        return "RECORD"

    fake.push = fake_push
    monkeypatch.setitem(sys.modules, "vilvik", fake)

    ga = _ga()
    out = ga.push_to_vilvik(name="local run")
    assert out == "RECORD"
    assert calls["ga"] is ga
    assert calls["kw"]["name"] == "local run"
    assert calls["kw"]["origin_overrides"]["client"] == "pygad_wrapper"
    assert calls["kw"]["origin_overrides"]["pygad_version"] == pygad.__version__


def test_push_to_vilvik_missing_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "vilvik", None)  # forces `import vilvik` to raise ImportError
    ga = _ga()
    with pytest.raises(ImportError, match="pip install vilvik"):
        ga.push_to_vilvik()
