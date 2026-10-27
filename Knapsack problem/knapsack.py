"""0-1 Knapsack Problem (KP).

Instance files (``<family>/<name>``, no extension)::

    <name>
    <generator / type>
    C                capacity
    w_1  p_1         weight and profit of item 1 (tab or space separated)
    ...

(The column order was checked against ``feature_CapacityFraction``
``= C / sum(w)`` in the feature CSV.)

Goal: choose items of total weight at most ``C`` with maximum total profit
(maximisation).

A :class:`Solution` is a 0/1 vector ``x`` over the items in file order.  The
violation is the capacity excess ``max(0, sum(w x) - C)`` (plus a length
mismatch, if any).  Use :meth:`Solution.from_indices` to build one from the
set of chosen item indices.

Features come from ``knapsack_features.csv`` (MATILDA).
"""
from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

PROBLEM = "Knapsack problem"
ACRONYM = "KP"
SENSE = "max"
PROBLEM_DIR = Path(__file__).resolve().parent
FEATURES_FILE = PROBLEM_DIR / "knapsack_features.csv"


@dataclass
class Instance:
    name: str
    capacity: float
    weights: List[float]
    profits: List[float]
    kind: Optional[str] = None  # second line of the file
    family: Optional[str] = None

    @property
    def n(self) -> int:
        return len(self.weights)


@dataclass
class Solution:
    """``x[i]`` is 1 if item ``i`` is packed, else 0."""

    x: List[int] = field(default_factory=list)

    @classmethod
    def from_indices(cls, indices: Iterable[int], n: int) -> "Solution":
        x = [0] * n
        for i in indices:
            x[i] = 1
        return cls(x)

    def selected(self) -> List[int]:
        return [i for i, v in enumerate(self.x) if v]

    def weight(self, instance: Instance) -> float:
        return sum(instance.weights[i] for i, v in enumerate(self.x[: instance.n]) if v)

    def objective(self, instance: Instance) -> float:
        """Raw objective: total profit of packed items."""
        return float(sum(instance.profits[i] for i, v in enumerate(self.x[: instance.n]) if v))

    def violation(self, instance: Instance) -> float:
        return max(0.0, self.weight(instance) - instance.capacity) + abs(len(self.x) - instance.n)

    def is_feasible(self, instance: Instance) -> bool:
        return self.violation(instance) <= 1e-9

    def goal_function(self, instance: Instance, penalty: Optional[float] = None) -> float:
        """Total profit; ``-inf`` (or ``profit - penalty * excess``) when infeasible."""
        return _penalised(self.objective(instance), self.violation(instance), penalty)

    def evaluate(self, instance: Instance) -> Dict[str, float]:
        return {
            "objective": self.objective(instance),
            "violation": self.violation(instance),
            "feasible": self.is_feasible(instance),
        }


def greedy(instance: Instance) -> Solution:
    """Reference heuristic: greedy by profit/weight ratio, then best single item."""
    order = sorted(
        range(instance.n),
        key=lambda i: -(instance.profits[i] / instance.weights[i]) if instance.weights[i] else -math.inf,
    )
    x = [0] * instance.n
    load = 0.0
    profit = 0.0
    for i in order:
        if load + instance.weights[i] <= instance.capacity:
            x[i] = 1
            load += instance.weights[i]
            profit += instance.profits[i]
    fits = [i for i in range(instance.n) if instance.weights[i] <= instance.capacity]
    if fits:
        best = max(fits, key=lambda i: instance.profits[i])
        if instance.profits[best] > profit:
            x = [0] * instance.n
            x[best] = 1
    return Solution(x)


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
                if not _is_hidden(f):
                    index.setdefault(f, Path(root) / f)
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
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    lines = [ln for ln in lines if ln]
    capacity = float(lines[2])
    weights, profits = [], []
    for ln in lines[3:]:
        w, p = ln.split()[:2]
        weights.append(float(w))
        profits.append(float(p))
    return Instance(
        name=name,
        capacity=capacity,
        weights=weights,
        profits=profits,
        kind=lines[1],
        family=_family_of(path),
    )


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
    """Feature row for ``name`` (``feature_*``, ``algo_*`` and ``source``)."""
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
