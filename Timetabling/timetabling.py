"""Curriculum-Based Course Timetabling (CB-CTT, ECTT format).

Instance files (``<family>/<file>.ectt``) use the extended ECTT format of
Bonutti, De Cesco, Di Gaspero & Schaerf (Ann. Oper. Res. 194, 2012)::

    Name: <name>
    Courses: <#>   Rooms: <#>   Days: <#>   Periods_per_day: <#>
    Curricula: <#> Min_Max_Daily_Lectures: <min> <max>
    UnavailabilityConstraints: <#>  RoomConstraints: <#>

    COURSES:                     <course> <teacher> <#lectures> <min_days> <#students> <double 0/1>
    ROOMS:                       <room> <capacity> <building>
    CURRICULA:                   <curriculum> <#courses> <course> ...
    UNAVAILABILITY_CONSTRAINTS:  <course> <day> <period>
    ROOM_CONSTRAINTS:            <course> <room>     (room NOT suitable for course)
    END.

ITC-2007 ``.ctt`` files (no building / double-lecture / min-max fields) are
parsed too.  The instance name is the ``Name:`` header unless the file name
is what the feature CSV uses (e.g. ``real/EA15.ectt`` is ``EasyAcademy15``,
``random/pca_*.ectt`` keep their file name).  Files duplicated in both
``real/`` and ``random/`` resolve to ``real/``.

A :class:`Solution` is a list of :class:`Assignment` ``(course, room, day,
period)``, one per lecture -- the standard ``.sol`` format
(``Solution.from_text`` / ``to_text``).

Hard constraints: H1 Lectures (all lectures scheduled, distinct periods),
H2 Conflicts (same curriculum or teacher), H3 RoomOccupancy,
H4 Availability, plus invalid ids/time slots.  Soft constraints S1..S9 are
weighted per formulation (Bonutti et al., Table 1)::

               S1  S2  S3  S4  S5  S6  S7  S8  S9
        UD1     1   5   1   -   -   -   -   -   -
        UD2     1   5   2   -   1   -   -   -   -      (ITC-2007, default)
        UD3     1   -   -   4   -   2   -   3   -
        UD4     1   1   -   1   -   1   -   H   1
        UD5     1   5   1   2   -   2   2   -   -

S1 RoomCapacity, S2 MinWorkingDays, S3 IsolatedLectures, S4 Windows,
S5 RoomStability, S6 StudentMinMaxLoad (days without lectures are free),
S7 TravelDistance, S8 RoomSuitability, S9 DoubleLectures.

The goal function is the weighted soft cost (minimisation) of a solution
without hard violations.

Features come from ``timetabling-features.csv`` (MATILDA).
"""
from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

PROBLEM = "Timetabling"
ACRONYM = "CTT"
SENSE = "min"
PROBLEM_DIR = Path(__file__).resolve().parent
FEATURES_FILE = PROBLEM_DIR / "timetabling-features.csv"

SOFT = (
    "RoomCapacity",
    "MinWorkingDays",
    "IsolatedLectures",
    "Windows",
    "RoomStability",
    "StudentMinMaxLoad",
    "TravelDistance",
    "RoomSuitability",
    "DoubleLectures",
)
HARD = ("Lectures", "Conflicts", "RoomOccupancy", "Availability", "Invalid")

# weight, or "H" for hard; missing = not used
FORMULATIONS: Dict[str, Dict[str, object]] = {
    "UD1": {"RoomCapacity": 1, "MinWorkingDays": 5, "IsolatedLectures": 1},
    "UD2": {"RoomCapacity": 1, "MinWorkingDays": 5, "IsolatedLectures": 2, "RoomStability": 1},
    "UD3": {"RoomCapacity": 1, "Windows": 4, "StudentMinMaxLoad": 2, "RoomSuitability": 3},
    "UD4": {
        "RoomCapacity": 1,
        "MinWorkingDays": 1,
        "Windows": 1,
        "StudentMinMaxLoad": 1,
        "RoomSuitability": "H",
        "DoubleLectures": 1,
    },
    "UD5": {
        "RoomCapacity": 1,
        "MinWorkingDays": 5,
        "IsolatedLectures": 1,
        "Windows": 2,
        "StudentMinMaxLoad": 2,
        "TravelDistance": 2,
    },
}
DEFAULT_FORMULATION = "UD2"


# --------------------------------------------------------------------------- #
# Instance
# --------------------------------------------------------------------------- #
@dataclass
class Course:
    teacher: str
    lectures: int
    min_working_days: int
    students: int
    double_lectures: bool = False


@dataclass
class Room:
    capacity: int
    building: str = "0"


@dataclass
class Instance:
    name: str
    days: int
    periods_per_day: int
    courses: Dict[str, Course]
    rooms: Dict[str, Room]
    curricula: Dict[str, List[str]]
    unavailability: Set[Tuple[str, int, int]] = field(default_factory=set)
    unsuitable_rooms: Set[Tuple[str, str]] = field(default_factory=set)
    min_daily_lectures: int = 0
    max_daily_lectures: int = 10**9
    family: Optional[str] = None

    @property
    def n_lectures(self) -> int:
        return sum(c.lectures for c in self.courses.values())

    @property
    def n_periods(self) -> int:
        return self.days * self.periods_per_day

    def _course_curricula(self) -> Dict[str, frozenset]:
        # cached per object; keyed on the curricula dict identity/size so that
        # replacing or growing ``curricula`` invalidates it
        key = (id(self.curricula), len(self.curricula))
        cache = self.__dict__.get("_cc_cache")
        if cache is None or cache[0] != key:
            out: Dict[str, Set[str]] = defaultdict(set)
            for q, cs in self.curricula.items():
                for c in cs:
                    out[c].add(q)
            cache = (key, {c: frozenset(v) for c, v in out.items()})
            self.__dict__["_cc_cache"] = cache
        return cache[1]

    def conflict(self, c1: str, c2: str) -> bool:
        """True if two different courses may not share a period."""
        if c1 == c2:
            return False
        if self.courses[c1].teacher == self.courses[c2].teacher:
            return True
        cc = self._course_curricula()
        return not cc.get(c1, frozenset()).isdisjoint(cc.get(c2, frozenset()))


# --------------------------------------------------------------------------- #
# Solution
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Assignment:
    course: str
    room: str
    day: int
    period: int


@dataclass
class Solution:
    """One :class:`Assignment` per scheduled lecture."""

    assignments: List[Assignment] = field(default_factory=list)

    # ---- I/O -------------------------------------------------------------- #
    @classmethod
    def from_text(cls, text: str) -> "Solution":
        """Parse the standard solution format ``<course> <room> <day> <period>``."""
        out = []
        for line in text.splitlines():
            parts = line.split()
            if len(parts) >= 4:
                out.append(Assignment(parts[0], parts[1], int(parts[2]), int(parts[3])))
        return cls(out)

    @classmethod
    def from_tuples(cls, rows: Iterable[Tuple[str, str, int, int]]) -> "Solution":
        return cls([Assignment(c, r, int(d), int(p)) for c, r, d, p in rows])

    def to_text(self) -> str:
        return "".join(f"{a.course} {a.room} {a.day} {a.period}\n" for a in self.assignments)

    # ---- evaluation ------------------------------------------------------- #
    def _valid(self, instance: Instance) -> List[Assignment]:
        return [
            a
            for a in self.assignments
            if a.course in instance.courses
            and a.room in instance.rooms
            and 0 <= a.day < instance.days
            and 0 <= a.period < instance.periods_per_day
        ]

    def hard_violations(self, instance: Instance, formulation: str = DEFAULT_FORMULATION) -> Dict[str, int]:
        valid = self._valid(instance)
        res = {k: 0 for k in HARD}
        res["Invalid"] = len(self.assignments) - len(valid)

        per_course: Dict[str, int] = defaultdict(int)
        course_slot: Dict[Tuple[str, int, int], int] = defaultdict(int)
        room_slot: Dict[Tuple[str, int, int], int] = defaultdict(int)
        slot_courses: Dict[Tuple[int, int], Set[str]] = defaultdict(set)
        for a in valid:
            per_course[a.course] += 1
            course_slot[(a.course, a.day, a.period)] += 1
            room_slot[(a.room, a.day, a.period)] += 1
            slot_courses[(a.day, a.period)].add(a.course)
            if (a.course, a.day, a.period) in instance.unavailability:
                res["Availability"] += 1
        # H1: missing/extra lectures, and two lectures of one course in one period
        for c, course in instance.courses.items():
            res["Lectures"] += abs(course.lectures - per_course.get(c, 0))
        res["Lectures"] += sum(k - 1 for k in course_slot.values() if k > 1)
        # H2
        for courses in slot_courses.values():
            cs = sorted(courses)
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    if instance.conflict(cs[i], cs[j]):
                        res["Conflicts"] += 1
        # H3
        res["RoomOccupancy"] = sum(k - 1 for k in room_slot.values() if k > 1)
        # S8 is hard in UD4
        if FORMULATIONS[formulation].get("RoomSuitability") == "H":
            res["RoomSuitability"] = sum(
                1 for a in valid if (a.course, a.room) in instance.unsuitable_rooms
            )
        return res

    def soft_costs(self, instance: Instance) -> Dict[str, int]:
        """Unweighted violation counts of S1..S9."""
        valid = self._valid(instance)
        res = {k: 0 for k in SOFT}
        P = instance.periods_per_day

        by_course: Dict[str, List[Assignment]] = defaultdict(list)
        for a in valid:
            by_course[a.course].append(a)
            # S1
            res["RoomCapacity"] += max(0, instance.courses[a.course].students - instance.rooms[a.room].capacity)
            # S8
            if (a.course, a.room) in instance.unsuitable_rooms:
                res["RoomSuitability"] += 1

        for c, lst in by_course.items():
            course = instance.courses[c]
            # S2
            res["MinWorkingDays"] += max(0, course.min_working_days - len({a.day for a in lst}))
            # S5
            res["RoomStability"] += len({a.room for a in lst}) - 1
            # S9
            if course.double_lectures:
                per_day: Dict[int, List[Assignment]] = defaultdict(list)
                for a in lst:
                    per_day[a.day].append(a)
                for day_list in per_day.values():
                    if len(day_list) < 2:
                        continue
                    for a in day_list:
                        if not any(
                            b is not a and b.room == a.room and abs(b.period - a.period) == 1
                            for b in day_list
                        ):
                            res["DoubleLectures"] += 1

        for courses in instance.curricula.values():
            # count[day][period], rooms[day][period]
            count = [[0] * P for _ in range(instance.days)]
            rooms: List[List[List[str]]] = [[[] for _ in range(P)] for _ in range(instance.days)]
            for c in courses:
                for a in by_course.get(c, ()):
                    count[a.day][a.period] += 1
                    rooms[a.day][a.period].append(a.room)
            for d in range(instance.days):
                row = count[d]
                daily = sum(row)
                # S3
                for p in range(P):
                    if row[p] and not ((p > 0 and row[p - 1]) or (p + 1 < P and row[p + 1])):
                        res["IsolatedLectures"] += row[p]
                # S4
                busy = [p for p in range(P) if row[p]]
                res["Windows"] += sum(busy[k + 1] - busy[k] - 1 for k in range(len(busy) - 1))
                # S6
                if daily > instance.max_daily_lectures:
                    res["StudentMinMaxLoad"] += daily - instance.max_daily_lectures
                elif 0 < daily < instance.min_daily_lectures:
                    res["StudentMinMaxLoad"] += instance.min_daily_lectures - daily
                # S7
                for p in range(P - 1):
                    for r1 in rooms[d][p]:
                        for r2 in rooms[d][p + 1]:
                            if instance.rooms[r1].building != instance.rooms[r2].building:
                                res["TravelDistance"] += 1
        return res

    def objective(self, instance: Instance, formulation: str = DEFAULT_FORMULATION) -> float:
        """Weighted soft cost (ignores hard constraints)."""
        weights = FORMULATIONS[formulation]
        costs = self.soft_costs(instance)
        return float(sum(costs[k] * w for k, w in weights.items() if w != "H"))

    def violation(self, instance: Instance, formulation: str = DEFAULT_FORMULATION) -> float:
        return float(sum(self.hard_violations(instance, formulation).values()))

    def is_feasible(self, instance: Instance, formulation: str = DEFAULT_FORMULATION) -> bool:
        return self.violation(instance, formulation) == 0

    def goal_function(
        self,
        instance: Instance,
        penalty: Optional[float] = None,
        formulation: str = DEFAULT_FORMULATION,
    ) -> float:
        """Weighted soft cost; ``inf`` or ``cost + penalty * hard`` if infeasible."""
        obj = self.objective(instance, formulation)
        viol = self.violation(instance, formulation)
        if viol == 0:
            return obj
        return math.inf if penalty is None else obj + penalty * viol

    def evaluate(self, instance: Instance, formulation: str = DEFAULT_FORMULATION) -> Dict[str, object]:
        hard = self.hard_violations(instance, formulation)
        return {
            "objective": self.objective(instance, formulation),
            "violation": float(sum(hard.values())),
            "feasible": sum(hard.values()) == 0,
            "hard": hard,
            "soft": self.soft_costs(instance),
            "formulation": formulation,
        }


def greedy(instance: Instance) -> Solution:
    """Reference heuristic: most-constrained-first greedy construction.

    It respects H2-H4 whenever it can; lectures that do not fit are left out
    (so the result may violate H1 on hard instances).
    """
    courses = sorted(
        instance.courses,
        key=lambda c: (
            -sum(1 for x in instance.courses if instance.conflict(c, x)),
            -instance.courses[c].lectures,
        ),
    )
    slots = [(d, p) for d in range(instance.days) for p in range(instance.periods_per_day)]
    rooms_by_cap = sorted(instance.rooms, key=lambda r: instance.rooms[r].capacity)
    room_busy: Set[Tuple[str, int, int]] = set()
    slot_courses: Dict[Tuple[int, int], Set[str]] = defaultdict(set)
    out: List[Assignment] = []
    for c in courses:
        course = instance.courses[c]
        used_days: Set[int] = set()
        placed = 0
        # prefer new days first (helps MinWorkingDays)
        for pass_ in (0, 1):
            for d, p in slots:
                if placed == course.lectures:
                    break
                if pass_ == 0 and d in used_days:
                    continue
                if (c, d, p) in instance.unavailability or c in slot_courses[(d, p)]:
                    continue
                if any(instance.conflict(c, o) for o in slot_courses[(d, p)]):
                    continue
                free = [r for r in rooms_by_cap if (r, d, p) not in room_busy]
                if not free:
                    continue
                fit = [
                    r
                    for r in free
                    if instance.rooms[r].capacity >= course.students
                    and (c, r) not in instance.unsuitable_rooms
                ]
                r = fit[0] if fit else free[-1]
                out.append(Assignment(c, r, d, p))
                room_busy.add((r, d, p))
                slot_courses[(d, p)].add(c)
                used_days.add(d)
                placed += 1
    return Solution(out)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_ectt(text: str, name: Optional[str] = None, family: Optional[str] = None) -> Instance:
    header: Dict[str, str] = {}
    sections: Dict[str, List[List[str]]] = defaultdict(list)
    section = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.upper().startswith("END"):
            break
        if line.endswith(":") and line[:-1].replace("_", "").isalpha() and line[:-1].isupper():
            section = line[:-1]
            continue
        if section is None:
            if ":" in line:
                k, v = line.split(":", 1)
                header[k.strip()] = v.strip()
            continue
        sections[section].append(line.split())

    courses: Dict[str, Course] = {}
    for p in sections["COURSES"]:
        courses[p[0]] = Course(
            teacher=p[1],
            lectures=int(p[2]),
            min_working_days=int(p[3]),
            students=int(p[4]),
            double_lectures=len(p) > 5 and p[5] == "1",
        )
    rooms = {p[0]: Room(int(p[1]), p[2] if len(p) > 2 else "0") for p in sections["ROOMS"]}
    curricula = {p[0]: p[2 : 2 + int(p[1])] for p in sections["CURRICULA"]}
    unav = {(p[0], int(p[1]), int(p[2])) for p in sections["UNAVAILABILITY_CONSTRAINTS"]}
    unsuitable = {(p[0], p[1]) for p in sections["ROOM_CONSTRAINTS"]}
    mn, mx = 0, 10**9
    if "Min_Max_Daily_Lectures" in header:
        a, b = header["Min_Max_Daily_Lectures"].split()[:2]
        mn, mx = int(a), int(b)
    return Instance(
        name=name or header.get("Name", ""),
        days=int(header["Days"]),
        periods_per_day=int(header["Periods_per_day"]),
        courses=courses,
        rooms=rooms,
        curricula=curricula,
        unavailability=unav,
        unsuitable_rooms=unsuitable,
        min_daily_lectures=mn,
        max_daily_lectures=mx,
        family=family,
    )


# --------------------------------------------------------------------------- #
# Instance discovery / loading
# --------------------------------------------------------------------------- #
_FAMILY_PRIORITY = {"real": 0}


def _header_name(path: Path) -> Optional[str]:
    with open(path, errors="replace") as fh:
        first = fh.readline()
    return first.split(":", 1)[1].strip() if first.startswith("Name:") else None


@lru_cache(maxsize=None)
def _index() -> Dict[str, Path]:
    feats = _features()
    candidates: List[Tuple[int, str, Path]] = []
    for fam in sorted(os.listdir(PROBLEM_DIR)):
        fam_dir = PROBLEM_DIR / fam
        if fam.startswith(".") or not fam_dir.is_dir() or fam == "__pycache__":
            continue
        for root, dirs, files in os.walk(fam_dir):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            for f in sorted(files):
                if f.startswith(".") or not f.endswith((".ectt", ".ctt")):
                    continue
                candidates.append((_FAMILY_PRIORITY.get(fam, 1), fam, Path(root) / f))
    index: Dict[str, Path] = {}
    for _, _, path in sorted(candidates, key=lambda t: (t[0], t[1], str(t[2]))):
        stem = path.stem
        hname = _header_name(path)
        if stem in feats:
            name = stem
        elif hname in feats:
            name = hname
        else:
            name = hname or stem
        index.setdefault(name, path)
    return index


def _family_of(path: Path) -> str:
    return path.relative_to(PROBLEM_DIR).parts[0]


def list_families() -> List[str]:
    return sorted({_family_of(p) for p in _index().values()})


def list_instances(family: Optional[str] = None, with_features: Optional[bool] = None) -> List[str]:
    """Names of instances that have a file (see ``bpp1d.list_instances``)."""
    feats = set(_features()) if with_features is not None else set()
    out = []
    for name, path in _index().items():
        if family is not None and _family_of(path) != family:
            continue
        if with_features is not None and (name in feats) != with_features:
            continue
        out.append(name)
    return sorted(out)


def instance_path(name: str) -> Path:
    try:
        return _index()[name]
    except KeyError:
        raise KeyError(f"{PROBLEM}: no instance file for {name!r}") from None


def load_instance(name: str) -> Instance:
    path = instance_path(name)
    return parse_ectt(path.read_text(errors="replace"), name=name, family=_family_of(path))


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def _coerce(value: str):
    value = value.strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return value


@lru_cache(maxsize=None)
def _feature_table() -> Tuple[Tuple[str, ...], Dict[str, str]]:
    """``(columns, {instance: raw csv line})`` -- rows are parsed on demand
    to keep memory low (the SAT table alone has ~6 million cells)."""
    rows: Dict[str, str] = {}
    with open(FEATURES_FILE, newline="", encoding="utf-8-sig") as fh:
        header = tuple(h.strip() for h in next(csv.reader([fh.readline()])))
        for line in fh:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            key = next(csv.reader([line]))[0] if '"' in line else line.split(",", 1)[0]
            rows[key.strip()] = line
    return header[1:], rows


def _features() -> Dict[str, str]:
    return _feature_table()[1]


def _feature_row(name: str) -> Dict[str, object]:
    header, rows = _feature_table()
    values = next(csv.reader([rows[name]]))[1:]
    return {h: _coerce(v) for h, v in zip(header, values)}


def feature_names() -> List[str]:
    """Column names returned by :func:`get_features` (instance id excluded)."""
    return list(_feature_table()[0])


def list_feature_instances() -> List[str]:
    return sorted(_features())


def get_features(name: str) -> Dict[str, object]:
    """Feature row for ``name`` (``feature_*``, ``algo_*`` and ``Source``)."""
    try:
        return _feature_row(name)
    except KeyError:
        raise KeyError(f"{PROBLEM}: no features for instance {name!r}") from None
