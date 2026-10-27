"""Symmetric Travelling Salesman Problem (TSP).

Instance files are TSPLIB (``<family>/<nnn>.tsp``, ``EDGE_WEIGHT_TYPE:
EUC_2D``, 100 cities each).  Instance names follow the feature CSV:
``<family>_<k>`` with ``k`` = 001..190.  Some folders number their files
000..189 and others 001..190; ``<family>_<k>`` always means the *k-th* file
of the folder (verified against the centroid features for every CSV row).
The families ``easyCLK-hardLKCC`` and ``hardCLK-easyLKCC`` have no feature
rows but are named the same way.

Goal: shortest closed tour visiting every city once (minimisation).
Distances follow TSPLIB: ``EUC_2D`` -> ``nint(euclidean)``, ``CEIL_2D`` ->
``ceil``, ``ATT`` -> pseudo-Euclidean; pass ``rounding=False`` to
``Instance.distance`` / ``Solution.goal_function`` for exact distances.

A :class:`Solution` is a permutation of city indices ``0..n-1`` (the order
of the ``NODE_COORD_SECTION``).  Non-permutations are infeasible.

Features come from ``tsp_features.csv`` (MATILDA).  Note: the upstream row
``random_001`` is missing one value, so its columns are shifted.
"""
from __future__ import annotations

import csv
import math
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROBLEM = "TSP"
ACRONYM = "TSP"
SENSE = "min"
PROBLEM_DIR = Path(__file__).resolve().parent
FEATURES_FILE = PROBLEM_DIR / "tsp_features.csv"

_NUM_RE = re.compile(r"^(\d+)\.tsp$")


@dataclass
class Instance:
    name: str
    coords: List[Tuple[float, float]]
    edge_weight_type: str = "EUC_2D"
    family: Optional[str] = None

    @property
    def n(self) -> int:
        return len(self.coords)

    def distance(self, i: int, j: int, rounding: bool = True) -> float:
        (x1, y1), (x2, y2) = self.coords[i], self.coords[j]
        dx, dy = x1 - x2, y1 - y2
        d = math.sqrt(dx * dx + dy * dy)
        if not rounding:
            return d
        t = self.edge_weight_type
        if t == "EUC_2D":
            return float(int(d + 0.5))
        if t == "CEIL_2D":
            return float(math.ceil(d))
        if t == "ATT":
            r = math.sqrt((dx * dx + dy * dy) / 10.0)
            ti = int(r + 0.5)
            return float(ti + 1 if ti < r else ti)
        return d


@dataclass
class Solution:
    """``tour`` lists city indices in visiting order (the tour is closed)."""

    tour: List[int] = field(default_factory=list)

    def objective(self, instance: Instance, rounding: bool = True) -> float:
        """Raw objective: closed-tour length."""
        t = self.tour
        if not t:
            return 0.0
        return sum(
            instance.distance(t[k], t[(k + 1) % len(t)], rounding) for k in range(len(t))
        )

    def violation(self, instance: Instance) -> float:
        n = instance.n
        valid = {c for c in self.tour if 0 <= c < n}
        return float((n - len(valid)) + abs(len(self.tour) - n))

    def is_feasible(self, instance: Instance) -> bool:
        return self.violation(instance) == 0

    def goal_function(
        self, instance: Instance, penalty: Optional[float] = None, rounding: bool = True
    ) -> float:
        """Tour length; ``inf`` (or penalised) when not a permutation."""
        viol = self.violation(instance)
        if viol == 0:
            return self.objective(instance, rounding)
        if penalty is None or any(not (0 <= c < instance.n) for c in self.tour):
            return math.inf
        return self.objective(instance, rounding) + penalty * viol

    def evaluate(self, instance: Instance) -> Dict[str, float]:
        return {
            "objective": self.objective(instance) if self.is_feasible(instance) else math.inf,
            "violation": self.violation(instance),
            "feasible": self.is_feasible(instance),
        }


def nearest_neighbour(instance: Instance, start: int = 0) -> Solution:
    """Reference heuristic: nearest-neighbour tour."""
    unvisited = set(range(instance.n))
    unvisited.discard(start)
    tour = [start]
    while unvisited:
        last = tour[-1]
        nxt = min(unvisited, key=lambda j: (instance.distance(last, j, False), j))
        tour.append(nxt)
        unvisited.remove(nxt)
    return Solution(tour)


# --------------------------------------------------------------------------- #
# Instance discovery / loading
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _index() -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for fam in sorted(os.listdir(PROBLEM_DIR)):
        fam_dir = PROBLEM_DIR / fam
        if fam.startswith(".") or not fam_dir.is_dir() or fam == "__pycache__":
            continue
        numbered = []
        for f in os.listdir(fam_dir):
            m = _NUM_RE.match(f)
            if m:
                numbered.append((int(m.group(1)), f))
        if not numbered:
            continue
        numbered.sort()
        shift = 1 if numbered[0][0] == 0 else 0
        for num, f in numbered:
            index[f"{fam}_{num + shift:03d}"] = fam_dir / f
    return index


def _family_of(name: str) -> str:
    return name.rsplit("_", 1)[0]


def list_families() -> List[str]:
    return sorted({_family_of(n) for n in _index()})


def list_instances(family: Optional[str] = None, with_features: Optional[bool] = None) -> List[str]:
    """Names of instances that have a file (see ``bpp1d.list_instances``)."""
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


def parse_tsplib(text: str) -> Tuple[List[Tuple[float, float]], str]:
    coords: List[Tuple[float, float]] = []
    ewt = "EUC_2D"
    in_coords = False
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("EOF"):
            break
        if in_coords:
            parts = s.split()
            if len(parts) >= 3 and re.match(r"^-?\d", parts[0]):
                coords.append((float(parts[1]), float(parts[2])))
                continue
            in_coords = False
        if ":" in s:
            key, val = (p.strip() for p in s.split(":", 1))
            if key == "EDGE_WEIGHT_TYPE":
                ewt = val
        if s.startswith("NODE_COORD_SECTION"):
            in_coords = True
    return coords, ewt


def load_instance(name: str) -> Instance:
    path = instance_path(name)
    coords, ewt = parse_tsplib(path.read_text())
    return Instance(name=name, coords=coords, edge_weight_type=ewt, family=_family_of(name))


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
    """Feature row for ``name`` (``feature_*``, ``algo_*`` and ``Sources``)."""
    try:
        return _feature_row(name)
    except KeyError:
        raise KeyError(f"{PROBLEM}: no features for instance {name!r}") from None
