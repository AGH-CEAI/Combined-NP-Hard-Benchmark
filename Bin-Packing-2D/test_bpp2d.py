import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpp2d as M  # noqa: E402
from bpp2d import Placement as P  # noqa: E402


@pytest.fixture
def toy():
    # bin 10x10, items: two 5x10 columns and one 10x5 slab
    return M.Instance(name="toy", bin_width=10, bin_height=10, items=[(5, 10), (5, 10), (10, 5)])


# --- goal function --------------------------------------------------------- #
def test_goal_feasible(toy):
    s = M.Solution([P(0, 0, 0), P(0, 5, 0), P(1, 0, 0)])
    assert s.is_feasible(toy)
    assert s.goal_function(toy) == 2


def test_overlap_detected(toy):
    s = M.Solution([P(0, 0, 0), P(0, 4, 0), P(1, 0, 0)])  # 1x10 overlap
    assert s.violation(toy) == pytest.approx(10)
    assert s.goal_function(toy) == math.inf
    assert s.goal_function(toy, penalty=1) == 2 + 10


def test_touching_edges_do_not_overlap(toy):
    s = M.Solution([P(0, 0, 0), P(0, 5, 0), P(1, 0, 5)])
    assert s.is_feasible(toy)


def test_out_of_bounds(toy):
    s = M.Solution([P(0, 0, 0), P(0, 5, 0), P(1, 0, 7)])  # top 10x2 sticks out
    assert s.violation(toy) == pytest.approx(20)
    s = M.Solution([P(0, 0, 0), P(0, 5, 0), P(1, -1, -1)])  # L-shaped 14 outside
    assert s.violation(toy) == pytest.approx(10 * 5 - 9 * 4)


def test_rotation_flag(toy):
    # rotate the 10x5 slab to 5x10 and put it next to an upright item
    s = M.Solution([P(0, 0, 0), P(1, 0, 0), P(1, 5, 0, rotated=True)])
    assert not s.is_feasible(toy)
    toy.allow_rotation = True
    assert s.is_feasible(toy)
    assert s.goal_function(toy) == 2


def test_unplaced(toy):
    s = M.Solution([P(0, 0, 0), None, P(1, 0, 0)])
    assert s.violation(toy) == 1


def test_shelf_toy(toy):
    s = M.shelf_first_fit(toy)
    assert s.is_feasible(toy)
    assert s.goal_function(toy) == 2


# --- instances ------------------------------------------------------------- #
def test_families_and_counts():
    assert M.list_families() == ["2DCPackGen", "CLASS", "GA", "GAInd"]
    names = M.list_instances()
    assert len(names) == len(set(names)) > 1000
    assert len(M.list_instances(with_features=True)) == 420


def test_load_and_features():
    inst = M.load_instance("CLASS01_040_01")
    f = M.get_features("CLASS01_040_01")
    assert inst.bin_width == f["feature_W"] == 10
    assert inst.bin_height == 10
    assert inst.n == 40 and len(set(inst.item_type)) < 40  # demand > 1 expanded
    assert f["Source"] == "CLASS01"
    assert inst.family == "CLASS"


def test_shelf_heuristic_sample():
    rng = random.Random(1)
    for name in rng.sample(M.list_instances(), 25):
        inst = M.load_instance(name)
        s = M.shelf_first_fit(inst)
        assert s.is_feasible(inst), name
        assert s.goal_function(inst) >= inst.lower_bound()


def test_features_dict_and_errors():
    f = M.get_features("generated_01_1")
    assert set(f) == set(M.feature_names())
    assert any(k.startswith("algo_") for k in f)
    with pytest.raises(KeyError):
        M.get_features("nope")
    with pytest.raises(KeyError):
        M.load_instance("nope")
