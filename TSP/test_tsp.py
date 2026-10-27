import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tsp as M  # noqa: E402


@pytest.fixture
def square():
    return M.Instance(name="toy", coords=[(0, 0), (0, 3), (4, 3), (4, 0)])


def test_goal_tour(square):
    assert M.Solution([0, 1, 2, 3]).goal_function(square) == 14
    assert M.Solution([0, 2, 1, 3]).goal_function(square) == 5 + 4 + 5 + 4


def test_tsplib_rounding():
    inst = M.Instance(name="r", coords=[(0, 0), (1, 1)])
    assert inst.distance(0, 1) == 1  # nint(1.414)
    assert inst.distance(0, 1, rounding=False) == pytest.approx(math.sqrt(2))
    inst.edge_weight_type = "CEIL_2D"
    assert inst.distance(0, 1) == 2


def test_invalid_tour(square):
    s = M.Solution([0, 1, 1, 3])
    assert not s.is_feasible(square) and s.violation(square) == 1
    assert s.goal_function(square) == math.inf
    assert s.goal_function(square, penalty=100) == s.objective(square) + 100
    assert M.Solution([0, 1, 2, 9]).goal_function(square, penalty=1) == math.inf


def test_nearest_neighbour(square):
    s = M.nearest_neighbour(square)
    assert s.is_feasible(square) and s.goal_function(square) == 14


def test_families_and_names():
    fams = M.list_families()
    assert fams == sorted(
        ["CLKeasy", "CLKhard", "LKCCeasy", "LKCChard", "easyCLK-hardLKCC", "hardCLK-easyLKCC", "random"]
    )
    for fam in fams:
        names = M.list_instances(family=fam)
        assert names[0] == f"{fam}_001" and names[-1] == f"{fam}_190"
    assert len(M.list_instances(with_features=True)) == 950


def _centroid(inst):
    return (
        sum(x for x, _ in inst.coords) / inst.n,
        sum(y for _, y in inst.coords) / inst.n,
    )


def test_name_mapping_matches_centroid_features():
    for name in M.list_instances(with_features=True):
        if name == "random_001":  # upstream row is missing a value
            continue
        cx, cy = _centroid(M.load_instance(name))
        f = M.get_features(name)
        assert cx == pytest.approx(f["feature_Centroid_X"], abs=0.011), name
        assert cy == pytest.approx(f["feature_Centroid_Y"], abs=0.011), name


def test_real_instance():
    inst = M.load_instance("CLKhard_001")
    assert inst.n == 100 and inst.edge_weight_type == "EUC_2D"
    s = M.nearest_neighbour(inst)
    perm = list(range(100))
    random.Random(7).shuffle(perm)
    assert s.is_feasible(inst)
    assert s.goal_function(inst) < M.Solution(perm).goal_function(inst)


def test_features():
    f = M.get_features("CLKeasy_001")
    assert set(f) == set(M.feature_names())
    assert {"algo_CLK_Mean_Effort", "algo_LKCC_Mean_Effort"} <= set(f)
    with pytest.raises(KeyError):
        M.get_features("nope")
