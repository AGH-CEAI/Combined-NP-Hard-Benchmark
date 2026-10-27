"""One-dimensional Bin Packing Problem (BPP-1D).

Instance files (``<family>/<name>.txt``)::

    n            number of items
    C            bin capacity
    w_1 ... w_n  item weights, one per line

Goal: pack every item into bins of capacity ``C`` using as few bins as
possible (minimisation).

A :class:`Solution` assigns every item (in file order, 0-based) to a bin
label.  Bin labels are arbitrary hashable values; the goal function is the
number of distinct labels used.  A solution is infeasible when the length of
the assignment does not match the number of items or when a bin is
over-filled; the violation is the total overload plus the number of missing
or extra items.

Features come from ``1d-bin-features.csv`` (MATILDA).
"""
from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

PROBLEM = "Bin-Packing-1D"
ACRONYM = "BPP1D"
SENSE = "min"
PROBLEM_DIR = Path(__file__).resolve().parent
FEATURES_FILE = PROBLEM_DIR / "1d-bin-features.csv"


# --------------------------------------------------------------------------- #
# Instance / solution
# --------------------------------------------------------------------------- #
@dataclass
class Instance:
    name: str
    capacity: float
    weights: List[float]
    family: Optional[str] = None

    @property
    def n(self) -> int:
        return len(self.weights)

    def lower_bound(self) -> int:
        """Continuous lower bound ``ceil(sum(w) / C)``."""
        return math.ceil(sum(self.weights) / self.capacity - 1e-9)


@dataclass
class Solution:
    """``assignment[i]`` is the bin label of item ``i``."""

    assignment: List[Hashable] = field(default_factory=list)

    @classmethod
    def from_bins(cls, bins: Sequence[Sequence[int]], n: int) -> "Solution":
        """Build a solution from a list of bins, each a list of item indices."""
        assignment: List[Hashable] = [None] * n
        for b, items in enumerate(bins):
            for i in items:
                assignment[i] = b
        return cls(assignment)

    def bins(self) -> Dict[Hashable, List[int]]:
        out: Dict[Hashable, List[int]] = {}
        for i, b in enumerate(self.assignment):
            if b is not None:
                out.setdefault(b, []).append(i)
        return out

    def loads(self, instance: Instance) -> Dict[Hashable, float]:
        loads: Dict[Hashable, float] = {}
        for i, b in enumerate(self.assignment[: instance.n]):
            if b is not None:
                loads[b] = loads.get(b, 0.0) + instance.weights[i]
        return loads

    def objective(self, instance: Instance) -> float:
        """Raw objective: number of bins used (ignores feasibility)."""
        return float(len({b for b in self.assignment[: instance.n] if b is not None}))

    def violation(self, instance: Instance) -> float:
        unassigned = sum(1 for b in self.assignment[: instance.n] if b is None)
        size_err = abs(len(self.assignment) - instance.n) + unassigned
        overload = sum(
            max(0.0, load - instance.capacity) for load in self.loads(instance).values()
        )
        return float(size_err) + overload

    def is_feasible(self, instance: Instance) -> bool:
        return self.violation(instance) <= 1e-9

    def goal_function(self, instance: Instance, penalty: Optional[float] = None) -> float:
        """Number of bins used.

        Infeasible solutions return ``inf`` or, when ``penalty`` is given,
        ``bins + penalty * violation``.
        """
        return _penalised(self.objective(instance), self.violation(instance), penalty)

    def evaluate(self, instance: Instance) -> Dict[str, float]:
        return {
            "objective": self.objective(instance),
            "violation": self.violation(instance),
            "feasible": self.is_feasible(instance),
        }


def first_fit_decreasing(instance: Instance) -> Solution:
    """Reference heuristic (FFD); handy for examples and tests."""
    order = sorted(range(instance.n), key=lambda i: -instance.weights[i])
    loads: List[float] = []
    assignment: List[Hashable] = [None] * instance.n
    for i in order:
        w = instance.weights[i]
        for b, load in enumerate(loads):
            if load + w <= instance.capacity + 1e-9:
                loads[b] += w
                assignment[i] = b
                break
        else:
            loads.append(w)
            assignment[i] = len(loads) - 1
    return Solution(assignment)


# --------------------------------------------------------------------------- #
# Instance discovery / loading
# --------------------------------------------------------------------------- #
def _is_hidden(name: str) -> bool:
    return name.startswith(".")  # includes macOS "._*" resource forks


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
                if f.endswith(".txt") and not _is_hidden(f):
                    index.setdefault(f[:-4], Path(root) / f)
    return index


def _family_of(path: Path) -> str:
    return path.relative_to(PROBLEM_DIR).parts[0]


def list_families() -> List[str]:
    return sorted({_family_of(p) for p in _index().values()})


def list_instances(family: Optional[str] = None, with_features: Optional[bool] = None) -> List[str]:
    """Names of instances that have a file.

    ``family`` restricts to one family folder.  ``with_features=True`` keeps
    only instances that also have a feature row, ``False`` only those without.
    """
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
    tokens = path.read_text().split()
    n = int(float(tokens[0]))
    capacity = float(tokens[1])
    weights = [float(t) for t in tokens[2 : 2 + n]]
    if len(weights) != n:
        raise ValueError(f"{path}: expected {n} weights, found {len(weights)}")
    return Instance(name=name, capacity=capacity, weights=weights, family=_family_of(path))


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
    """Names of all instances with a feature row (some may lack a file)."""
    return sorted(_features())


def get_features(name: str) -> Dict[str, object]:
    """Feature row for ``name`` as ``{column: value}``.

    Contains ``feature_*`` columns, ``algo_*`` performance columns and the
    ``Source`` column; numeric values are floats.
    """
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
