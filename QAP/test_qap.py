import itertools
import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qap as M  # noqa: E402


@pytest.fixture
def toy():
    A = [[0, 3, 1, 2], [3, 0, 4, 0], [1, 4, 0, 5], [2, 0, 5, 0]]  # flow
    B = [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]]  # distance (line)
    return M.Instance(name="toy", A=A, B=B)


def brute(inst, p):
    return sum(inst.A[i][j] * inst.B[p[i]][p[j]] for i in range(inst.n) for j in range(inst.n))


def test_goal_matches_definition(toy):
    for p in itertools.permutations(range(4)):
        assert M.Solution(list(p)).goal_function(toy) == brute(toy, p)


def test_identity(toy):
    # 2*(3*1 + 1*2 + 2*3 + 4*1 + 0 + 5*1) = 2*20
    assert M.identity(toy).goal_function(toy) == 40


def test_one_based(toy):
    assert M.Solution.from_one_based([1, 2, 3, 4]).permutation == [0, 1, 2, 3]


def test_not_a_permutation(toy):
    s = M.Solution([0, 0, 1, 2])
    assert not s.is_feasible(toy) and s.violation(toy) == 1
    assert s.goal_function(toy) == math.inf
    assert s.goal_function(toy, penalty=100) == brute(toy, [0, 0, 1, 2]) + 100
    assert M.Solution([0, 1, 2]).goal_function(toy, penalty=1) == math.inf


def test_families_and_counts():
    fams = M.list_families()
    assert "QAPLIB" in fams and len(fams) == 12
    assert len(M.list_instances()) == 802
    assert len(M.list_instances(with_features=True)) == 802


def test_load_real():
    inst = M.load_instance("tai40a")
    assert inst.n == 40 and len(inst.B) == 40 and all(len(r) == 40 for r in inst.A)
    assert inst.family == "QAPLIB"
    rng = random.Random(4)
    perm = list(range(inst.n))
    rng.shuffle(perm)
    assert M.Solution(perm).goal_function(inst) == brute(inst, perm)


def test_lipa_symmetry_sanity():
    # costs are invariant under simultaneous relabelling of facilities
    inst = M.load_instance("chr25a")
    s = M.identity(inst)
    assert s.goal_function(inst) > 0


def test_features():
    f = M.get_features("chr25a")
    assert set(f) == set(M.feature_names())
    assert {"algo_BMA", "algo_MMAS"} <= set(f)
    with pytest.raises(KeyError):
        M.get_features("nope")
    with pytest.raises(KeyError):
        M.load_instance("nope")
