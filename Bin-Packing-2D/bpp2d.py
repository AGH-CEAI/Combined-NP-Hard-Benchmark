"""Two-dimensional (orthogonal) Bin Packing Problem (BPP-2D).

Instance files are JSON (``<family>/.../json/<k>.json``)::

    {"Name": "CLASS01_020_01",
     "Objects": [{"Length": W, "Height": H, ...}],         # bin size
     "Items":   [{"Length": w, "Height": h, "Demand": d, ...}, ...]}

The instance name is the ``Name`` field (it is what the feature CSV uses);
file numbers repeat across folders.  Items with ``Demand > 1`` are expanded
into ``d`` identical copies, so ``Instance.items`` lists one rectangle per
copy and ``Instance.item_type[i]`` points back to the JSON item.

Goal: place all rectangles, without overlap and inside the bins, using as
few ``W x H`` bins as possible (minimisation).  By default items keep their
orientation (``Instance.allow_rotation = False``, the classical 2BP|O|F
setting of the Berkey-Wang / Martello-Vigo CLASS instances); set it to
``True`` to allow 90 degree rotations.

A :class:`Solution` holds one :class:`Placement` per expanded item: the bin
label and the bottom-left corner ``(x, y)``.  The violation adds the number
of unplaced items, forbidden rotations, the area sticking out of bins and
the pairwise overlap area.

Features come from ``2d-packing-features.csv`` (MATILDA).
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Hashable, List, Optional, Tuple

PROBLEM = "Bin-Packing-2D"
ACRONYM = "BPP2D"
SENSE = "min"
PROBLEM_DIR = Path(__file__).resolve().parent
FEATURES_FILE = PROBLEM_DIR / "2d-packing-features.csv"


@dataclass
class Instance:
    name: str
    bin_width: float
    bin_height: float
    items: List[Tuple[float, float]]  # (width, height) per expanded copy
    item_type: List[int] = field(default_factory=list)
    allow_rotation: bool = False
    family: Optional[str] = None

    @property
    def n(self) -> int:
        return len(self.items)

    def lower_bound(self) -> int:
        """Area lower bound ``ceil(sum(w*h) / (W*H))``."""
        area = sum(w * h for w, h in self.items)
        return math.ceil(area / (self.bin_width * self.bin_height) - 1e-9)


@dataclass
class Placement:
    bin: Hashable
    x: float
    y: float
    rotated: bool = False


@dataclass
class Solution:
    """``placements[i]`` places expanded item ``i`` (``None`` = unplaced)."""

    placements: List[Optional[Placement]] = field(default_factory=list)

    def _rects(self, instance: Instance):
        for i, p in enumerate(self.placements[: instance.n]):
            if p is None:
                continue
            w, h = instance.items[i]
            if p.rotated:
                w, h = h, w
            yield i, p, w, h

    def objective(self, instance: Instance) -> float:
        """Raw objective: number of bins used."""
        return float(len({p.bin for p in self.placements[: instance.n] if p is not None}))

    def violation(self, instance: Instance) -> float:
        W, H = instance.bin_width, instance.bin_height
        unplaced = sum(1 for p in self.placements[: instance.n] if p is None)
        viol = float(abs(len(self.placements) - instance.n) + unplaced)
        per_bin: Dict[Hashable, List[Tuple[float, float, float, float]]] = {}
        for _, p, w, h in self._rects(instance):
            if p.rotated and not instance.allow_rotation:
                viol += 1
            # area outside the bin
            inside_w = max(0.0, min(p.x + w, W) - max(p.x, 0.0))
            inside_h = max(0.0, min(p.y + h, H) - max(p.y, 0.0))
            viol += w * h - inside_w * inside_h
            per_bin.setdefault(p.bin, []).append((p.x, p.y, p.x + w, p.y + h))
        for rects in per_bin.values():
            for a in range(len(rects)):
                ax0, ay0, ax1, ay1 = rects[a]
                for b in range(a + 1, len(rects)):
                    bx0, by0, bx1, by1 = rects[b]
                    ow = min(ax1, bx1) - max(ax0, bx0)
                    oh = min(ay1, by1) - max(ay0, by0)
                    if ow > 0 and oh > 0:
                        viol += ow * oh
        return viol

    def is_feasible(self, instance: Instance) -> bool:
        return self.violation(instance) <= 1e-9

    def goal_function(self, instance: Instance, penalty: Optional[float] = None) -> float:
        """Number of bins used; ``inf`` (or penalised) when infeasible."""
        return _penalised(self.objective(instance), self.violation(instance), penalty)

    def evaluate(self, instance: Instance) -> Dict[str, float]:
        return {
            "objective": self.objective(instance),
            "violation": self.violation(instance),
            "feasible": self.is_feasible(instance),
        }


def shelf_first_fit(instance: Instance) -> Solution:
    """Reference heuristic: First-Fit Decreasing-Height shelf packing."""
    W, H = instance.bin_width, instance.bin_height
    order = sorted(range(instance.n), key=lambda i: -instance.items[i][1])
    # each bin: list of shelves [y, height, used_width]; plus top used height
    bins: List[List[List[float]]] = []
    tops: List[float] = []
    placements: List[Optional[Placement]] = [None] * instance.n
    for i in order:
        w, h = instance.items[i]
        if w > W or h > H:
            continue  # cannot be packed in its orientation
        placed = False
        for b, shelves in enumerate(bins):
            for s in shelves:
                if h <= s[1] and s[2] + w <= W:
                    placements[i] = Placement(b, s[2], s[0])
                    s[2] += w
                    placed = True
                    break
            if placed:
                break
            if tops[b] + h <= H:
                shelves.append([tops[b], h, w])
                placements[i] = Placement(b, 0.0, tops[b])
                tops[b] += h
                placed = True
                break
        if not placed:
            bins.append([[0.0, h, w]])
            tops.append(h)
            placements[i] = Placement(len(bins) - 1, 0.0, 0.0)
    return Solution(placements)


# --------------------------------------------------------------------------- #
# Instance discovery / loading
# --------------------------------------------------------------------------- #
_NAME_RE = re.compile(r'"Name"\s*:\s*"([^"]*)"')


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
                if not f.endswith(".json") or _is_hidden(f):
                    continue
                path = Path(root) / f
                with open(path, encoding="utf-8") as fh:
                    m = _NAME_RE.search(fh.read(512))
                if m is None:  # fall back to a full parse
                    with open(path, encoding="utf-8") as fh:
                        name = json.load(fh)["Name"]
                else:
                    name = m.group(1)
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


def load_instance(name: str, allow_rotation: bool = False) -> Instance:
    path = instance_path(name)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if len(data["Objects"]) != 1:
        raise ValueError(f"{path}: expected exactly one bin type")
    obj = data["Objects"][0]
    items: List[Tuple[float, float]] = []
    item_type: List[int] = []
    for t, it in enumerate(data["Items"]):
        for _ in range(int(it.get("Demand") or 1)):
            items.append((float(it["Length"]), float(it["Height"])))
            item_type.append(t)
    return Instance(
        name=data["Name"],
        bin_width=float(obj["Length"]),
        bin_height=float(obj["Height"]),
        items=items,
        item_type=item_type,
        allow_rotation=allow_rotation,
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
