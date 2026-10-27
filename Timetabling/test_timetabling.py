import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import timetabling as M  # noqa: E402

TOY = """Name: toy
Courses: 3
Rooms: 2
Days: 2
Periods_per_day: 3
Curricula: 1
Min_Max_Daily_Lectures: 2 2
UnavailabilityConstraints: 1
RoomConstraints: 1

COURSES:
c1 t1 2 2 20 0
c2 t2 2 1 15 1
c3 t1 1 1 5 0

ROOMS:
rA 30 1
rB 10 2

CURRICULA:
q1 2 c1 c2

UNAVAILABILITY_CONSTRAINTS:
c3 0 0

ROOM_CONSTRAINTS:
c3 rB

END.
"""

# Hand-computed reference solution:
#   day 0: c1@p0 rA, c3@p1 rA, c2@p2 rA      day 1: c1@p0 rB, c2@p1 rA
GOOD = """c1 rA 0 0
c1 rB 1 0
c2 rA 0 2
c2 rA 1 1
c3 rA 0 1
"""


@pytest.fixture
def toy():
    return M.parse_ectt(TOY)


def test_parse(toy):
    assert toy.name == "toy" and toy.days == 2 and toy.periods_per_day == 3
    assert toy.courses["c2"].double_lectures and not toy.courses["c1"].double_lectures
    assert toy.rooms["rB"] == M.Room(10, "2")
    assert toy.curricula == {"q1": ["c1", "c2"]}
    assert toy.unavailability == {("c3", 0, 0)}
    assert toy.unsuitable_rooms == {("c3", "rB")}
    assert (toy.min_daily_lectures, toy.max_daily_lectures) == (2, 2)
    assert toy.n_lectures == 5
    assert toy.conflict("c1", "c3")  # same teacher
    assert toy.conflict("c1", "c2")  # same curriculum
    assert not toy.conflict("c2", "c3")


def test_soft_costs_by_hand(toy):
    s = M.Solution.from_text(GOOD)
    assert s.hard_violations(toy) == {k: 0 for k in M.HARD}
    assert s.soft_costs(toy) == {
        "RoomCapacity": 10,  # c1 (20 students) in rB (10 seats)
        "MinWorkingDays": 0,
        "IsolatedLectures": 2,  # q1 on day 0 at periods 0 and 2
        "Windows": 1,  # q1 on day 0: one empty period in between
        "RoomStability": 1,  # c1 uses rA and rB
        "StudentMinMaxLoad": 0,
        "TravelDistance": 1,  # q1 day 1: rB (bldg 2) -> rA (bldg 1)
        "RoomSuitability": 0,
        "DoubleLectures": 0,
    }


@pytest.mark.parametrize(
    "formulation,expected",
    [("UD1", 10 + 2), ("UD2", 10 + 2 * 2 + 1), ("UD3", 10 + 4), ("UD4", 10 + 1), ("UD5", 10 + 2 + 2 + 2)],
)
def test_goal_per_formulation(toy, formulation, expected):
    s = M.Solution.from_text(GOOD)
    assert s.goal_function(toy, formulation=formulation) == expected


def test_default_is_itc2007(toy):
    assert M.DEFAULT_FORMULATION == "UD2"
    assert M.Solution.from_text(GOOD).goal_function(toy) == 15


def test_hard_violations(toy):
    bad = GOOD.replace("c3 rA 0 1", "c3 rB 0 0")  # teacher clash, unavailable, unsuitable
    s = M.Solution.from_text(bad)
    hard = s.hard_violations(toy)
    assert hard["Availability"] == 1
    assert hard["Conflicts"] == 1
    assert hard["RoomOccupancy"] == 0
    assert s.goal_function(toy) == math.inf
    assert s.goal_function(toy, penalty=1000) == s.objective(toy) + 2000
    # RoomSuitability is soft in UD3 and hard in UD4
    assert s.soft_costs(toy)["RoomSuitability"] == 1
    assert s.hard_violations(toy, "UD4")["RoomSuitability"] == 1


def test_lectures_and_room_occupancy(toy):
    s = M.Solution.from_text(GOOD + "c2 rA 0 0\n")  # extra c2 lecture, rA double-booked, clash
    hard = s.hard_violations(toy)
    assert hard["Lectures"] == 1 and hard["RoomOccupancy"] == 1 and hard["Conflicts"] == 1
    missing = M.Solution.from_text("\n".join(GOOD.splitlines()[:-1]))
    assert missing.hard_violations(toy)["Lectures"] == 1


def test_invalid_assignments(toy):
    s = M.Solution.from_text(GOOD + "zz rA 0 0\nc1 rZ 0 0\nc1 rA 5 0\n")
    assert s.hard_violations(toy)["Invalid"] == 3


def test_double_lectures(toy):
    grouped = GOOD.replace("c2 rA 0 2", "c2 rA 1 2")  # c2 at day 1 p1,p2 same room
    assert M.Solution.from_text(grouped).soft_costs(toy)["DoubleLectures"] == 0
    split = GOOD.replace("c2 rA 0 2", "c2 rB 1 2")  # adjacent but different rooms
    assert M.Solution.from_text(split).soft_costs(toy)["DoubleLectures"] == 2


def test_min_max_load(toy):
    toy.max_daily_lectures = 1
    assert M.Solution.from_text(GOOD).soft_costs(toy)["StudentMinMaxLoad"] == 2
    toy.max_daily_lectures, toy.min_daily_lectures = 5, 3
    assert M.Solution.from_text(GOOD).soft_costs(toy)["StudentMinMaxLoad"] == 2


def test_text_roundtrip(toy):
    s = M.Solution.from_text(GOOD)
    assert M.Solution.from_text(s.to_text()) == s
    assert M.Solution.from_tuples([("c1", "rA", 0, 0)]).assignments == [M.Assignment("c1", "rA", 0, 0)]


def test_greedy_toy(toy):
    s = M.greedy(toy)
    assert s.is_feasible(toy)
    assert s.goal_function(toy) < math.inf


# --- real instances -------------------------------------------------------- #
def test_families_and_naming():
    assert M.list_families() == ["random", "real"]
    names = M.list_instances()
    assert len(names) == len(set(names)) > 7000
    # every feature row resolves to a file (by file name or Name: header)
    assert not set(M.list_feature_instances()) - set(names)
    assert "EasyAcademy15" in names and "EA15" not in names
    assert M.instance_path("Ing0607-2").parent.name == "real"  # real/ preferred over random/


def test_load_itc2007_comp07():
    inst = M.load_instance("Ing0607-2")  # = comp07.ectt
    assert (len(inst.courses), len(inst.rooms), len(inst.curricula)) == (131, 20, 77)
    assert (inst.days, inst.periods_per_day) == (5, 5)
    assert len(inst.unavailability) == 667 and len(inst.unsuitable_rooms) == 308
    s = M.greedy(inst)
    hard = s.hard_violations(inst)
    assert hard["Conflicts"] == hard["RoomOccupancy"] == hard["Availability"] == hard["Invalid"] == 0
    assert s.objective(inst) >= 0


def test_features():
    f = M.get_features("pca_sample_instance_818_seed_10")
    assert set(f) == set(M.feature_names())
    assert f["Source"] == "RandPCA"
    assert any(k.startswith("algo_") for k in f)
    with pytest.raises(KeyError):
        M.get_features("nope")
    with pytest.raises(KeyError):
        M.load_instance("nope")
