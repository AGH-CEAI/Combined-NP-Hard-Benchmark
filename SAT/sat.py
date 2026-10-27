"""Boolean Satisfiability (SAT / MAX-SAT view).

Instance files are DIMACS CNF (``<family>/.../<name>.cnf``)::

    c comment
    p cnf <n_vars> <n_clauses>
    1 -5 3 0
    ...
    %             (optional SATLIB end marker)

The instance name is the path relative to this folder, exactly as in the
``file`` column of ``satzilla_features.csv`` (e.g. ``"uf20-91/uf20-0388_5124.cnf"``).
Families are the SATLIB benchmark groups
(https://www.cs.ubc.ca/~hoos/SATLIB/benchm.htm): ``uf*`` are satisfiable
uniform random 3-SAT, ``uuf*`` unsatisfiable, ``flat*`` flat graph colouring,
``CBS_*`` controlled backbone size, etc.

Goal: number of clauses left unsatisfied by a truth assignment
(minimisation; ``0`` means the assignment is a model).  Every complete
assignment is feasible; the violation only counts unassigned or missing
variables.

A :class:`Solution` stores ``assignment[v]`` (bool) for variable ``v + 1``.

Features are the SATzilla features in ``satzilla_features.csv``.
"""
from __future__ import annotations

import csv
import math
import os
import random
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

PROBLEM = "SAT"
ACRONYM = "SAT"
SENSE = "min"
PROBLEM_DIR = Path(__file__).resolve().parent
FEATURES_FILE = PROBLEM_DIR / "satzilla_features.csv"


@dataclass
class Instance:
    name: str
    n_vars: int
    clauses: List[List[int]]  # DIMACS literals (+v / -v, 1-based)
    family: Optional[str] = None

    @property
    def n_clauses(self) -> int:
        return len(self.clauses)


@dataclass
class Solution:
    """``assignment[v]`` is the truth value of DIMACS variable ``v + 1``."""

    assignment: List[Optional[bool]] = field(default_factory=list)

    @classmethod
    def from_literals(cls, literals: Iterable[int], n_vars: int) -> "Solution":
        """Build from a model as printed by SAT solvers, e.g. ``[1, -2, 3]``."""
        a: List[Optional[bool]] = [None] * n_vars
        for lit in literals:
            if lit != 0 and abs(lit) <= n_vars:
                a[abs(lit) - 1] = lit > 0
        return cls(a)

    def to_literals(self) -> List[int]:
        return [(v + 1) if val else -(v + 1) for v, val in enumerate(self.assignment) if val is not None]

    def unsatisfied(self, instance: Instance) -> List[int]:
        """Indices of clauses not satisfied (unassigned literals count as false)."""
        a = self.assignment
        k = len(a)
        out = []
        for ci, clause in enumerate(instance.clauses):
            for lit in clause:
                v = abs(lit) - 1
                if v < k and a[v] is not None and a[v] == (lit > 0):
                    break
            else:
                out.append(ci)
        return out

    def objective(self, instance: Instance) -> float:
        """Raw objective: number of unsatisfied clauses."""
        return float(len(self.unsatisfied(instance)))

    def violation(self, instance: Instance) -> float:
        missing = sum(1 for v in self.assignment[: instance.n_vars] if v is None)
        return float(missing + abs(len(self.assignment) - instance.n_vars))

    def is_feasible(self, instance: Instance) -> bool:
        return self.violation(instance) == 0

    def is_model(self, instance: Instance) -> bool:
        """True when the assignment satisfies every clause."""
        return self.objective(instance) == 0

    def goal_function(self, instance: Instance, penalty: Optional[float] = None) -> float:
        """Number of unsatisfied clauses; ``inf`` (or penalised) if incomplete."""
        return _penalised(self.objective(instance), self.violation(instance), penalty)

    def evaluate(self, instance: Instance) -> Dict[str, float]:
        return {
            "objective": self.objective(instance),
            "violation": self.violation(instance),
            "feasible": self.is_feasible(instance),
        }


def walksat(instance: Instance, max_flips: int = 100_000, p: float = 0.5, seed: int = 0) -> Solution:
    """Reference heuristic: a compact WalkSAT (Selman et al., 1994)."""
    rng = random.Random(seed)
    n = instance.n_vars
    a = [rng.random() < 0.5 for _ in range(n)]
    occ: List[List[int]] = [[] for _ in range(n)]
    for ci, cl in enumerate(instance.clauses):
        for lit in cl:
            occ[abs(lit) - 1].append(ci)
    true_count = [sum(1 for lit in cl if a[abs(lit) - 1] == (lit > 0)) for cl in instance.clauses]
    unsat = {ci for ci, c in enumerate(true_count) if c == 0}

    def break_count(v: int) -> int:
        # clauses that become unsatisfied when flipping v
        return sum(
            1
            for ci in occ[v]
            if true_count[ci] == 1
            and any(abs(l) - 1 == v and a[v] == (l > 0) for l in instance.clauses[ci])
        )

    for _ in range(max_flips):
        if not unsat:
            break
        ci = rng.choice(tuple(unsat))
        vars_ = [abs(l) - 1 for l in instance.clauses[ci]]
        if not vars_:
            break  # empty clause: unsatisfiable
        if rng.random() < p:
            v = rng.choice(vars_)
        else:
            v = min(vars_, key=break_count)
        a[v] = not a[v]
        for cj in occ[v]:
            sat_now = sum(1 for l in instance.clauses[cj] if a[abs(l) - 1] == (l > 0))
            true_count[cj] = sat_now
            if sat_now == 0:
                unsat.add(cj)
            else:
                unsat.discard(cj)
    return Solution(list(a))


# --------------------------------------------------------------------------- #
# Instance discovery / loading
# --------------------------------------------------------------------------- #
def _is_hidden(name: str) -> bool:
    return name.startswith(".")


@lru_cache(maxsize=None)
def _index() -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for entry in sorted(os.listdir(PROBLEM_DIR)):
        fam_dir = PROBLEM_DIR / entry
        if _is_hidden(entry) or not fam_dir.is_dir() or entry == "__pycache__":
            continue
        for root, dirs, files in os.walk(fam_dir):
            dirs[:] = sorted(d for d in dirs if not _is_hidden(d))
            for f in sorted(files):
                if f.endswith(".cnf") and not _is_hidden(f):
                    path = Path(root) / f
                    index[path.relative_to(PROBLEM_DIR).as_posix()] = path
    return index


def _family_of(name: str) -> str:
    return name.split("/", 1)[0]


def list_families() -> List[str]:
    return sorted({_family_of(n) for n in _index()})


def list_instances(family: Optional[str] = None, with_features: Optional[bool] = None) -> List[str]:
    """Relative paths of all ``.cnf`` files (see ``bpp1d.list_instances``)."""
    feats = set(_features()) if with_features is not None else set()
    out = []
    for name in _index():
        if family is not None and _family_of(name) != family:
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


def parse_dimacs_cnf(text: str):
    n_vars = 0
    clauses: List[List[int]] = []
    cur: List[int] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s[0] == "c":
            continue
        if s[0] == "%":
            break
        if s[0] == "p":
            n_vars = int(s.split()[2])
            continue
        for tok in s.split():
            lit = int(tok)
            if lit == 0:
                clauses.append(cur)
                cur = []
            else:
                cur.append(lit)
    if cur:
        clauses.append(cur)
    return n_vars, clauses


def load_instance(name: str) -> Instance:
    path = instance_path(name)
    n_vars, clauses = parse_dimacs_cnf(path.read_text(errors="replace"))
    return Instance(name=name, n_vars=n_vars, clauses=clauses, family=_family_of(name))


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
    """SATzilla feature row for ``name`` (a relative ``.cnf`` path)."""
    try:
        return _feature_row(name)
    except KeyError:
        raise KeyError(f"{PROBLEM}: no features for instance {name!r}") from None


def _penalised(obj: float, viol: float, penalty: Optional[float]) -> float:
    if viol <= 1e-9:
        return obj
    if penalty is None:
        return math.inf if SENSE == "min" else -math.inf
    return obj + penalty * viol if SENSE == "min" else obj - penalty * viol
