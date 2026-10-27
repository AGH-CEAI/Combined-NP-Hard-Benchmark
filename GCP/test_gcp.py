import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gcp as M  # noqa: E402


@pytest.fixture
def triangle_plus():
    # triangle 0-1-2 plus pendant vertex 3 attached to 0
    return M.Instance(name="toy", n=4, edges=[(0, 1), (0, 2), (1, 2), (0, 3)])


# --- goal function --------------------------------------------------------- #
def test_goal_feasible(triangle_plus):
    s = M.Solution([0, 1, 2, 1])
    assert s.is_feasible(triangle_plus)
    assert s.goal_function(triangle_plus) == 3


def test_conflicts(triangle_plus):
    s = M.Solution([0, 0, 1, 0])
    assert s.conflicts(triangle_plus) == [(0, 1), (0, 3)]
    assert s.violation(triangle_plus) == 2
    assert s.goal_function(triangle_plus) == math.inf
    assert s.goal_function(triangle_plus, penalty=10) == 2 + 20


def test_uncoloured(triangle_plus):
    assert M.Solution([0, 1, 2, None]).violation(triangle_plus) == 1
    assert M.Solution([0, 1, 2]).violation(triangle_plus) == 1


def test_dsatur_toy(triangle_plus):
    s = M.dsatur(triangle_plus)
    assert s.is_feasible(triangle_plus) and s.goal_function(triangle_plus) == 3


def test_parse_dimacs_dedupes_and_drops_loops():
    n, edges = M.parse_dimacs("c x\np edge 3 4\ne 1 2\ne 2 1\ne 3 3\ne 2 3\n")
    assert n == 3 and edges == [(0, 1), (1, 2)]


# --- name -> file mapping -------------------------------------------------- #
@pytest.mark.parametrize(
    "name,file",
    [
        ("Bipartite-000-730Nodes-21393Edges-00000ExtraEdges", "1"),
        ("Cycle_Driven-000-310Nodes", "1001"),
        ("Geometric-001-741Nodes", "2002"),
        ("Girth_And_Degree_Inhibited-000-506Nodes", "3001"),
        ("IDD_Graph-002-519Nodes", "4003"),
        ("Weight_Biased-000-730Nodes", "5001"),
        ("DIMACS-6700", "6700"),
        ("round-robin-26", "round-robin-26"),
        ("car-f-92.col", "car-f-92.col"),
    ],
)
def test_file_name_for(name, file):
    assert M.file_name_for(name) == file


def test_mapping_consistent_with_density_feature():
    """The density of the mapped file must equal feature_Density."""
    rng = random.Random(2)
    names = M.list_instances(with_features=True)
    by_source = {}
    for n in names:
        by_source.setdefault(M.get_features(n)["Source"], []).append(n)
    for source, lst in by_source.items():
        for name in rng.sample(lst, min(5, len(lst))):
            inst = M.load_instance(name)
            assert inst.density() == pytest.approx(M.get_features(name)["feature_Density"], rel=1e-6), (
                source,
                name,
            )


def test_list_instances():
    names = M.list_instances()
    assert len(names) > 6800
    assert len(M.list_instances(family="DIMACS")) == 731
    assert M.list_families() == ["Instances"]
    assert "readMeInstances.pdf" not in names
    # Flat / Random feature rows have no graph file
    missing = set(M.list_feature_instances()) - set(names)
    assert {M.get_features(n)["Source"] for n in missing} == {"Flat", "Random"}


def test_dsatur_real():
    for name in ["DIMACS-6700", "car-f-92.col", "1a"]:
        inst = M.load_instance(name)
        s = M.dsatur(inst)
        assert s.is_feasible(inst)
        assert 1 <= s.goal_function(inst) <= inst.n


def test_features():
    f = M.get_features("DIMACS-6700")
    assert set(f) == set(M.feature_names())
    assert f["Source"] == "DIMACS"
    assert "algo_DSATUR" in f
    with pytest.raises(KeyError):
        M.get_features("nope")
    with pytest.raises(KeyError):
        M.load_instance("FlatK10_1")
