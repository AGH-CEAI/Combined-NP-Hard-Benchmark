import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sat as M  # noqa: E402


@pytest.fixture
def toy():
    # (x1 or not x2) and (x2) and (not x1 or x3)
    return M.Instance(name="toy", n_vars=3, clauses=[[1, -2], [2], [-1, 3]])


def test_goal_model(toy):
    s = M.Solution([True, True, True])
    assert s.is_model(toy) and s.goal_function(toy) == 0


def test_goal_counts_unsatisfied(toy):
    s = M.Solution([False, True, False])
    assert s.unsatisfied(toy) == [0]
    assert s.goal_function(toy) == 1
    assert M.Solution([False, False, False]).goal_function(toy) == 1


def test_incomplete_assignment(toy):
    s = M.Solution([True, None, True])
    assert not s.is_feasible(toy)
    assert s.goal_function(toy) == math.inf
    assert s.goal_function(toy, penalty=10) == 1 + 10


def test_literals_roundtrip(toy):
    s = M.Solution.from_literals([1, -2, 3, 0], 3)
    assert s.assignment == [True, False, True]
    assert s.to_literals() == [1, -2, 3]


def test_parse_satlib_end_marker():
    n, cl = M.parse_dimacs_cnf("c\np cnf 3 2\n 1 -2\n 3 0\n-1 0\n%\n0\n")
    assert n == 3 and cl == [[1, -2, 3], [-1]]


def test_families_and_names():
    fams = M.list_families()
    for f in ["uf20-91", "uuf50-218", "flat30-60", "AIM", "PHOLE"]:
        assert f in fams
    names = M.list_instances(family="uf20-91")
    assert len(names) == 1000 and all(n.startswith("uf20-91/") for n in names)


def test_every_feature_row_has_a_file():
    assert not set(M.list_feature_instances()) - set(M.list_instances())


def test_load_matches_features():
    rng = random.Random(5)
    for name in rng.sample(M.list_instances(with_features=True), 30):
        inst = M.load_instance(name)
        f = M.get_features(name)
        assert inst.n_vars == f["nvarsOrig"], name
        assert inst.n_clauses == f["nclausesOrig"], name


def test_walksat_solves_satisfiable():
    for name in M.list_instances(family="uf20-91")[:5]:
        inst = M.load_instance(name)
        s = M.walksat(inst, max_flips=20000, seed=1)
        assert s.is_model(inst) and s.goal_function(inst) == 0


def test_unsatisfiable_never_zero():
    inst = M.load_instance(M.list_instances(family="uuf50-218")[0])
    rng = random.Random(6)
    for _ in range(20):
        s = M.Solution([rng.random() < 0.5 for _ in range(inst.n_vars)])
        assert s.goal_function(inst) >= 1
    assert M.walksat(inst, max_flips=5000).goal_function(inst) >= 1


def test_features():
    name = "AIM/aim-100-1_6-no-1_5598.cnf"
    f = M.get_features(name)
    assert set(f) == set(M.feature_names())
    assert f["nvarsOrig"] == 100
    with pytest.raises(KeyError):
        M.get_features("nope.cnf")
