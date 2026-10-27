"""Quadratic Assignment Problem (QAP).

Instance files (``<family>/<name>.dat``, QAPLIB format)::

    n
    A   (n x n matrix)
    B   (n x n matrix)

Goal: find a permutation ``p`` of ``{0..n-1}`` minimising the QAPLIB cost

    cost(p) = sum_{i, j} A[i][j] * B[p[i]][p[j]]

(minimisation).  ``p[i]`` is the location assigned to facility ``i``.  This is
exactly the QAPLIB convention, so permutations published in ``.sln`` files
(1-based) can be checked with ``Solution.from_one_based``.

A :class:`Solution` that is not a permutation is infeasible; the violation
counts missing locations plus a length mismatch.

Features come from ``qap_features.csv`` (MATILDA).
"""
from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROBLEM = "QAP"
ACRONYM = "QAP"
SENSE = "min"
PROBLEM_DIR = Path(__file__).resolve().parent
FEATURES_FILE = PROBLEM_DIR / "qap_features.csv"


@dataclass
class Instance:
    name: str
    A: List[List[float]]
    B: List[List[float]]
    family: Optional[str] = None

    @property
    def n(self) -> int:
        return len(self.A)


@dataclass
class Solution:
    """``permutation[i]`` is the (0-based) location of facility ``i``."""

    permutation: List[int] = field(default_factory=list)

    @classmethod
    def from_one_based(cls, perm: Sequence[int]) -> "Solution":
        return cls([int(p) - 1 for p in perm])

    def objective(self, instance: Instance) -> float:
        """Raw QAPLIB cost (requires a valid permutation)."""
        p = self.permutation
        A, B = instance.A, instance.B
        total = 0.0
        for i in range(instance.n):
            Ai = A[i]
            Bpi = B[p[i]]
            total += sum(Ai[j] * Bpi[p[j]] for j in range(instance.n))
        return total

    def violation(self, instance: Instance) -> float:
        n = instance.n
        valid = {int(p) for p in self.permutation if int(p) == p and 0 <= p < n}
        return float((n - len(valid)) + abs(len(self.permutation) - n))

    def is_feasible(self, instance: Instance) -> bool:
        return self.violation(instance) == 0

    def goal_function(self, instance: Instance, penalty: Optional[float] = None) -> float:
        """QAPLIB cost; ``inf`` when the vector is not a permutation.

        A non-permutation has no well-defined cost, so ``penalty`` is only
        applied when the vector still indexes valid locations.
        """
        viol = self.violation(instance)
        if viol == 0:
            return self.objective(instance)
        n = instance.n
        if penalty is None or len(self.permutation) != n or any(
            not (0 <= p < n) for p in self.permutation
        ):
            return math.inf
        return self.objective(instance) + penalty * viol

    def evaluate(self, instance: Instance) -> Dict[str, float]:
        feasible = self.is_feasible(instance)
        return {
            "objective": self.objective(instance) if feasible else math.inf,
            "violation": self.violation(instance),
            "feasible": feasible,
        }


def identity(instance: Instance) -> Solution:
    return Solution(list(range(instance.n)))


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
                if f.endswith(".dat") and not _is_hidden(f):
                    index.setdefault(f[:-4], Path(root) / f)
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


def parse_qaplib(text: str):
    tokens = text.split()
    n = int(tokens[0])
    vals = [float(t) for t in tokens[1 : 1 + 2 * n * n]]
    if len(vals) != 2 * n * n:
        raise ValueError(f"expected {2 * n * n} matrix entries, found {len(vals)}")
    A = [vals[i * n : (i + 1) * n] for i in range(n)]
    off = n * n
    B = [vals[off + i * n : off + (i + 1) * n] for i in range(n)]
    return A, B


def load_instance(name: str) -> Instance:
    path = instance_path(name)
    A, B = parse_qaplib(path.read_text())
    return Instance(name=name, A=A, B=B, family=_family_of(path))


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
    """Feature row for ``name`` (``feature_*`` and ``algo_*`` columns)."""
    try:
        return _feature_row(name)
    except KeyError:
        raise KeyError(f"{PROBLEM}: no features for instance {name!r}") from None
