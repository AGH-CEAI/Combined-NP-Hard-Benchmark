"""Graph Colouring Problem (GCP).

Instance files live in ``Instances/`` in DIMACS format::

    c comment
    p edge <n> <m>
    e <u> <v>          (vertices numbered 1..n)

Most files are numbered (``Instances/1`` ... ``Instances/6743``) while the
feature CSV uses descriptive names.  The mapping, verified against the
``feature_Density`` column for every row, is::

    Bipartite-<k>-...                   -> Instances/<k + 1>
    Cycle_Driven-<k>-...                -> Instances/<k + 1001>
    Geometric-<k>-...                   -> Instances/<k + 2001>
    Girth_And_Degree_Inhibited-<k>-...  -> Instances/<k + 3001>
    IDD_Graph-<k>-...                   -> Instances/<k + 4001>
    Weight_Biased-<k>-...               -> Instances/<k + 5001>
    DIMACS-<N>                          -> Instances/<N>
    anything else (e.g. ``round-robin-26``, ``1a``, ``car-f-92.col``)
                                        -> Instances/<name>

Feature rows of the ``Flat`` and ``Random`` sources have no instance file.
Files not referenced by the CSV are listed under their file name.

Goal: colour the vertices so that adjacent vertices get different colours,
using as few colours as possible (minimisation).

A :class:`Solution` stores ``colors[v]`` for every vertex ``v`` (0-based,
i.e. DIMACS vertex ``v + 1``).  The violation is the number of
monochromatic edges plus uncoloured / missing vertices.

Features come from ``gcp-features.csv`` (MATILDA).
"""
from __future__ import annotations

import csv
import math
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Hashable, List, Optional, Tuple

PROBLEM = "GCP"
ACRONYM = "GCP"
SENSE = "min"
PROBLEM_DIR = Path(__file__).resolve().parent
INSTANCES_DIR = PROBLEM_DIR / "Instances"
FEATURES_FILE = PROBLEM_DIR / "gcp-features.csv"

_OFFSETS = {
    "Bipartite": 1,
    "Cycle_Driven": 1001,
    "Geometric": 2001,
    "Girth_And_Degree_Inhibited": 3001,
    "IDD_Graph": 4001,
    "Weight_Biased": 5001,
}
_GEN_RE = re.compile(r"^(" + "|".join(_OFFSETS) + r")-(\d+)-")
_DIMACS_RE = re.compile(r"^DIMACS-(\d+)$")


@dataclass
class Instance:
    name: str
    n: int
    edges: List[Tuple[int, int]]  # 0-based, u < v, unique
    family: Optional[str] = None

    @property
    def m(self) -> int:
        return len(self.edges)

    def density(self) -> float:
        return 2.0 * self.m / (self.n * (self.n - 1)) if self.n > 1 else 0.0

    def adjacency(self) -> List[List[int]]:
        adj: List[List[int]] = [[] for _ in range(self.n)]
        for u, v in self.edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj


@dataclass
class Solution:
    """``colors[v]`` is the colour (any hashable) of vertex ``v`` (0-based)."""

    colors: List[Optional[Hashable]] = field(default_factory=list)

    def objective(self, instance: Instance) -> float:
        """Raw objective: number of distinct colours used."""
        return float(len({c for c in self.colors[: instance.n] if c is not None}))

    def conflicts(self, instance: Instance) -> List[Tuple[int, int]]:
        cols = self.colors
        k = len(cols)
        return [
            (u, v)
            for u, v in instance.edges
            if u < k and v < k and cols[u] is not None and cols[u] == cols[v]
        ]

    def violation(self, instance: Instance) -> float:
        missing = sum(1 for c in self.colors[: instance.n] if c is None)
        size_err = abs(len(self.colors) - instance.n)
        return float(len(self.conflicts(instance)) + missing + size_err)

    def is_feasible(self, instance: Instance) -> bool:
        return self.violation(instance) == 0

    def goal_function(self, instance: Instance, penalty: Optional[float] = None) -> float:
        """Number of colours; ``inf`` (or penalised) when infeasible."""
        return _penalised(self.objective(instance), self.violation(instance), penalty)

    def evaluate(self, instance: Instance) -> Dict[str, float]:
        return {
            "objective": self.objective(instance),
            "violation": self.violation(instance),
            "feasible": self.is_feasible(instance),
        }


def dsatur(instance: Instance) -> Solution:
    """Reference heuristic: DSATUR (Brélaz, 1979)."""
    adj = instance.adjacency()
    colors: List[Optional[int]] = [None] * instance.n
    sat: List[set] = [set() for _ in range(instance.n)]
    deg = [len(a) for a in adj]
    for _ in range(instance.n):
        v = max(
            (u for u in range(instance.n) if colors[u] is None),
            key=lambda u: (len(sat[u]), deg[u], -u),
        )
        c = 0
        while c in sat[v]:
            c += 1
        colors[v] = c
        for u in adj[v]:
            sat[u].add(c)
    return Solution(colors)


# --------------------------------------------------------------------------- #
# Instance discovery / loading
# --------------------------------------------------------------------------- #
def file_name_for(name: str) -> str:
    """File name in ``Instances/`` for a CSV instance name."""
    m = _GEN_RE.match(name)
    if m:
        return str(_OFFSETS[m.group(1)] + int(m.group(2)))
    m = _DIMACS_RE.match(name)
    if m:
        return m.group(1)
    return name


@lru_cache(maxsize=None)
def _index() -> Dict[str, Path]:
    files = {
        f
        for f in os.listdir(INSTANCES_DIR)
        if not f.startswith(".") and not f.lower().endswith(".pdf")
        and (INSTANCES_DIR / f).is_file()
    }
    index: Dict[str, Path] = {}
    used = set()
    for name in _features():
        f = file_name_for(name)
        if f in files:
            index[name] = INSTANCES_DIR / f
            used.add(f)
    for f in sorted(files - used):
        index.setdefault(f, INSTANCES_DIR / f)
    return index


def list_families() -> List[str]:
    return ["Instances"]


def list_instances(family: Optional[str] = None, with_features: Optional[bool] = None) -> List[str]:
    """Names of instances that have a file.

    ``family`` may be ``"Instances"`` (the only folder) or a ``Source`` value
    from the feature CSV such as ``"DIMACS"`` or ``"Bipartite"``.
    """
    feats = _features()
    out = []
    for name in _index():
        has = name in feats
        if with_features is not None and has != with_features:
            continue
        if family is not None and family != "Instances":
            if not has or _feature_row(name).get("Source") != family:
                continue
        out.append(name)
    return sorted(out)


def instance_path(name: str) -> Path:
    try:
        return _index()[name]
    except KeyError:
        raise KeyError(f"{PROBLEM}: no instance file for {name!r}") from None


def parse_dimacs(text: str) -> Tuple[int, List[Tuple[int, int]]]:
    n = 0
    edges = set()
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "p":
            n = int(parts[2])
        elif parts[0] == "e":
            u, v = int(parts[1]) - 1, int(parts[2]) - 1
            if u != v:
                edges.add((min(u, v), max(u, v)))
    return n, sorted(edges)


def load_instance(name: str) -> Instance:
    path = instance_path(name)
    n, edges = parse_dimacs(path.read_text(errors="replace"))
    family = _feature_row(name).get("Source") if name in _features() else None
    return Instance(name=name, n=n, edges=edges, family=family)


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


def _penalised(obj: float, viol: float, penalty: Optional[float]) -> float:
    if viol <= 1e-9:
        return obj
    if penalty is None:
        return math.inf if SENSE == "min" else -math.inf
    return obj + penalty * viol if SENSE == "min" else obj - penalty * viol
