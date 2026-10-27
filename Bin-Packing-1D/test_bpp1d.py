import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpp1d as M  # noqa: E402


@pytest.fixture
def toy():
    return M.Instance(name="toy", capacity=10, weights=[6, 4, 5, 5])


# --- goal function --------------------------------------------------------- #
def test_goal_feasible(toy):
    s = M.Solution([0, 0, 1, 1])
    assert s.is_feasible(toy)
    assert s.goal_function(toy) == 2
    assert s.evaluate(toy) == {"objective": 2, "violation": 0, "feasible": True}


def test_goal_arbitrary_labels(toy):
    assert M.Solution(["a", "a", "b", "b"]).goal_function(toy) == 2


def test_goal_overload(toy):
    s = M.Solution([0, 0, 0, 1])  # bin 0 load 15
    assert not s.is_feasible(toy)
    assert s.violation(toy) == 5
    assert s.goal_function(toy) == math.inf
    assert s.goal_function(toy, penalty=10) == 2 + 10 * 5


def test_goal_missing_items(toy):
    assert M.Solution([0, 0, 1]).violation(toy) == 1
    assert M.Solution([0, 0, 1, None]).violation(toy) == 1


def test_from_bins(toy):
    s = M.Solution.from_bins([[0, 1], [2, 3]], toy.n)
    assert s.assignment == [0, 0, 1, 1]
    assert s.bins() == {0: [0, 1], 1: [2, 3]}


def test_ffd_toy(toy):
    s = M.first_fit_decreasing(toy)
    assert s.is_feasible(toy) and s.goal_function(toy) == 2 == toy.lower_bound()


# --- instances ------------------------------------------------------------- #
def test_families():
    fams = M.list_families()
    for f in ["Falkenauer U", "Falkenauer T", "Scholl 1", "Delorme", "Wascher", "randomly generated"]:
        assert f in fams


def test_list_instances_skips_resource_forks():
    names = M.list_instances()
    assert len(names) > 8000
    assert not any(n.startswith("._") or n.startswith(".") for n in names)
    assert len(M.list_instances(family="Scholl 3")) == 10


def test_load_instance_matches_features():
    inst = M.load_instance("Falkenauer_u120_03")
    f = M.get_features("Falkenauer_u120_03")
    assert inst.n == f["feature_n"] == 120
    assert inst.capacity == 150
    assert inst.family == "Falkenauer U"
    assert sum(inst.weights) / inst.n / inst.capacity == pytest.approx(f["feature_Mean"], rel=1e-6)
    assert max(inst.weights) / inst.capacity == pytest.approx(f["feature_Max"], rel=1e-6)


def test_random_sample_matches_features():
    rng = random.Random(0)
    names = M.list_instances(with_features=True)
    for name in rng.sample(names, 40):
        inst = M.load_instance(name)
        f = M.get_features(name)
        assert inst.n == f["feature_n"], name
        assert sum(inst.weights) / inst.n / inst.capacity == pytest.approx(f["feature_Mean"], rel=1e-4)


def test_ffd_real_instances():
    for name in ["Falkenauer_u120_03", "HARD2", "Waescher_TEST0005"]:
        inst = M.load_instance(name)
        s = M.first_fit_decreasing(inst)
        assert s.is_feasible(inst)
        assert s.goal_function(inst) >= inst.lower_bound()


# --- features -------------------------------------------------------------- #
def test_get_features_dict():
    f = M.get_features("Falkenauer_u120_03")
    assert isinstance(f, dict)
    assert set(f) == set(M.feature_names())
    assert f["Source"] == "Falkenauer U"
    assert all(isinstance(f[k], float) for k in f if k.startswith(("feature_", "algo_")))


def test_get_features_returns_copy():
    f = M.get_features("Falkenauer_u120_03")
    f["feature_n"] = -1
    assert M.get_features("Falkenauer_u120_03")["feature_n"] == 120


def test_unknown_instance():
    with pytest.raises(KeyError):
        M.get_features("does-not-exist")
    with pytest.raises(KeyError):
        M.load_instance("does-not-exist")


def test_feature_rows_without_files():
    missing = set(M.list_feature_instances()) - set(M.list_instances())
    assert missing and all(n.startswith("rgen") for n in missing)
