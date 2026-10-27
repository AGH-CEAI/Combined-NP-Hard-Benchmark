"""Tests of the top-level ``benchmark`` registry."""
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import benchmark as nb  # noqa: E402

EXPECTED = [
    "Bin-Packing-1D",
    "Bin-Packing-2D",
    "GCP",
    "Knapsack problem",
    "QAP",
    "SAT",
    "TSP",
    "Timetabling",
]


def test_list_problems():
    assert nb.list_problems() == EXPECTED


@pytest.mark.parametrize(
    "alias,folder",
    [("kp", "Knapsack problem"), ("Knapsack", "Knapsack problem"), ("bpp1d", "Bin-Packing-1D"),
     ("Bin Packing 2D", "Bin-Packing-2D"), ("ctt", "Timetabling"), ("tsp", "TSP"), ("Graph-Coloring", "GCP")],
)
def test_resolve(alias, folder):
    assert nb.resolve(alias) == folder


def test_unknown_problem():
    with pytest.raises(KeyError):
        nb.get_problem("vehicle routing")


@pytest.mark.parametrize("problem", EXPECTED)
def test_uniform_api(problem):
    mod = nb.get_problem(problem)
    for attr in ("Instance", "Solution", "load_instance", "get_features", "list_instances",
                 "list_families", "list_feature_instances", "feature_names", "SENSE", "ACRONYM"):
        assert hasattr(mod, attr), (problem, attr)
    for meth in ("goal_function", "violation", "is_feasible", "evaluate"):
        assert callable(getattr(mod.Solution, meth)), (problem, meth)
    names = nb.list_instances(problem)
    assert names and names == sorted(set(names))
    with_f = nb.list_instances(problem, with_features=True)
    without = nb.list_instances(problem, with_features=False)
    assert sorted(with_f + without) == names
    fam = nb.list_families(problem)[0]
    assert set(nb.list_instances(problem, family=fam)) <= set(names)
    # an instance with features loads and its features are a dict
    name = with_f[0]
    assert nb.load_instance(problem, name).name == name
    feats = nb.get_features(problem, name)
    assert isinstance(feats, dict) and list(feats) == nb.feature_names(problem)


def test_same_module_object():
    import tsp  # registered by get_problem
    assert tsp is nb.get_problem("TSP")
    from benchmark import knapsack
    assert knapsack is nb.get_problem("KP")


def test_goal_function_wrapper():
    tsp = nb.get_problem("TSP")
    sol = tsp.Solution(list(range(100)))
    by_name = nb.goal_function("TSP", "CLKeasy_001", sol)
    by_obj = nb.goal_function("TSP", tsp.load_instance("CLKeasy_001"), sol)
    assert by_name == by_obj > 0
    assert nb.goal_function("TSP", "CLKeasy_001", tsp.Solution([0, 0])) == math.inf
    assert nb.sense("KP") == "max" and nb.sense("TSP") == "min"


def test_summary():
    rows = {r["problem"]: r for r in nb.summary()}
    assert set(rows) == set(EXPECTED)
    assert rows["SAT"]["feature_rows"] == rows["SAT"]["with_features"] == 44675
    assert rows["QAP"]["instances"] == 802


def test_cli():
    out = subprocess.run(
        [sys.executable, str(ROOT / "benchmark.py"), "instances", "TSP", "--family", "random", "--limit", "2"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    assert out == ["random_001", "random_002"]
    out = subprocess.run(
        [sys.executable, str(ROOT / "benchmark.py"), "features", "QAP", "chr25a"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert '"algo_BMA"' in out


def test_examples_script_runs():
    subprocess.run([sys.executable, str(ROOT / "examples.py")], check=True, capture_output=True, cwd=ROOT)
