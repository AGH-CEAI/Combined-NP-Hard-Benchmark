import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import knapsack as M  # noqa: E402


@pytest.fixture
def toy():
    return M.Instance(name="toy", capacity=10, weights=[5, 4, 6], profits=[10, 7, 12])


def test_goal_feasible(toy):
    s = M.Solution([1, 1, 0])
    assert s.is_feasible(toy)
    assert s.goal_function(toy) == 17


def test_goal_infeasible_is_maximisation_penalty(toy):
    s = M.Solution([1, 0, 1])  # weight 11 > 10
    assert s.violation(toy) == 1
    assert s.goal_function(toy) == -math.inf
    assert s.goal_function(toy, penalty=3) == 22 - 3


def test_from_indices(toy):
    s = M.Solution.from_indices([0, 1], toy.n)
    assert s.x == [1, 1, 0] and s.selected() == [0, 1]


def test_greedy_toy(toy):
    s = M.greedy(toy)
    assert s.is_feasible(toy) and s.goal_function(toy) == 17  # optimum here
    assert M.SENSE == "max"


def test_families_and_counts():
    assert M.list_families() == ["Genetic_Instances", "Original_Instances"]
    assert len(M.list_instances()) == 5300
    assert len(M.list_instances(with_features=False)) == 0


def test_load_matches_features():
    rng = random.Random(3)
    for name in rng.sample(M.list_instances(), 30):
        inst = M.load_instance(name)
        f = M.get_features(name)
        # weight is the first column: CapacityFraction = C / sum(w)
        assert inst.capacity / sum(inst.weights) == pytest.approx(f["feature_CapacityFraction"], rel=1e-6)


def test_greedy_real():
    for name in ["EvenOddStrong096", "GACubc_26_9"]:
        inst = M.load_instance(name)
        s = M.greedy(inst)
        assert s.is_feasible(inst) and s.goal_function(inst) > 0


def test_features():
    f = M.get_features("AlmostStrong001")
    assert set(f) == set(M.feature_names())
    assert {"algo_Expknap", "algo_Minknap", "algo_Combo"} <= set(f)
    with pytest.raises(KeyError):
        M.get_features("nope")
