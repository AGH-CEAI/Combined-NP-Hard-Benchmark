"""Single entry point to every problem in the Combined NP-Hard Benchmark.

Each problem folder contains a self-contained module (``bpp1d.py``,
``tsp.py``, ...) with the same API; this file finds and imports them so that
everything is reachable from the repository root::

    import benchmark as nb

    nb.list_problems()                       # ['Bin-Packing-1D', ..., 'Timetabling']
    nb.list_instances("TSP")                 # ['CLKeasy_001', ...]
    inst = nb.load_instance("TSP", "CLKeasy_001")
    nb.get_features("TSP", "CLKeasy_001")    # {'feature_Mean_StdDevDist': 98.95, ...}

    tsp = nb.get_problem("TSP")              # the module itself
    sol = tsp.Solution(list(range(inst.n)))
    sol.goal_function(inst)                  # tour length

    from benchmark import tsp, knapsack      # modules also importable by name

Problems can be referred to by folder name, module name or acronym, case
insensitively (``"Knapsack problem"``, ``"knapsack"``, ``"KP"``).

Command line::

    python benchmark.py problems
    python benchmark.py families  TSP
    python benchmark.py instances TSP [--family CLKeasy] [--with-features] [--limit 20]
    python benchmark.py features  TSP CLKeasy_001
    python benchmark.py summary
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent

# folder -> (module file stem, acronym, extra aliases)
_REGISTRY = {
    "Bin-Packing-1D": ("bpp1d", "BPP1D", ("bpp", "bin-packing", "binpacking1d", "1dbpp")),
    "Bin-Packing-2D": ("bpp2d", "BPP2D", ("binpacking2d", "2dbpp")),
    "GCP": ("gcp", "GCP", ("graph-coloring", "graph-colouring", "coloring", "colouring")),
    "Knapsack problem": ("knapsack", "KP", ("knapsack-problem", "0-1kp")),
    "QAP": ("qap", "QAP", ("quadratic-assignment",)),
    "SAT": ("sat", "SAT", ("satisfiability", "maxsat")),
    "TSP": ("tsp", "TSP", ("travelling-salesman", "traveling-salesman")),
    "Timetabling": ("timetabling", "CTT", ("cbctt", "course-timetabling", "ectt")),
}


def _norm(key: str) -> str:
    return "".join(ch for ch in key.lower() if ch.isalnum())


_ALIASES: Dict[str, str] = {}
for _folder, (_mod, _acr, _extra) in _REGISTRY.items():
    for _k in (_folder, _mod, _acr, *_extra):
        _ALIASES[_norm(_k)] = _folder

_LOADED: Dict[str, ModuleType] = {}


# --------------------------------------------------------------------------- #
# Problem access
# --------------------------------------------------------------------------- #
def list_problems() -> List[str]:
    """Names (folder names) of all problems available in this repository."""
    return [f for f, (mod, _, _) in _REGISTRY.items() if (ROOT / f / f"{mod}.py").is_file()]


def resolve(problem: str) -> str:
    """Canonical folder name for a problem name, module name or acronym."""
    try:
        return _ALIASES[_norm(problem)]
    except KeyError:
        raise KeyError(f"unknown problem {problem!r}; available: {list_problems()}") from None


def get_problem(problem: str) -> ModuleType:
    """Import and return the module of ``problem`` (cached).

    The module is registered in ``sys.modules`` under its own name
    (``"tsp"``, ``"bpp1d"``, ...), so ``import tsp`` afterwards returns the
    same object.  If an unrelated module of that name is already imported
    (say a third-party ``tsp`` package) it is left alone and ours is
    registered as ``"cnhb_<name>"`` instead.
    """
    folder = resolve(problem)
    if folder in _LOADED:
        return _LOADED[folder]
    mod_name = _REGISTRY[folder][0]
    existing = sys.modules.get(mod_name)
    path = ROOT / folder / f"{mod_name}.py"
    existing_file = getattr(existing, "__file__", None) if existing is not None else None
    if existing_file and Path(existing_file).resolve() == path:
        module = existing
    else:
        if existing is not None:
            mod_name = f"cnhb_{mod_name}"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
    _LOADED[folder] = module
    return module


def __getattr__(name: str) -> ModuleType:  # PEP 562: ``from benchmark import tsp``
    if _norm(name) in _ALIASES and not name.startswith("_"):
        return get_problem(name)
    raise AttributeError(f"module 'benchmark' has no attribute {name!r}")


# --------------------------------------------------------------------------- #
# Convenience wrappers
# --------------------------------------------------------------------------- #
def list_families(problem: str) -> List[str]:
    """Instance families (sub-folders) of ``problem``."""
    return get_problem(problem).list_families()


def list_instances(
    problem: str, family: Optional[str] = None, with_features: Optional[bool] = None
) -> List[str]:
    """Names of all instances of ``problem`` that have an instance file.

    ``family`` restricts to one family; ``with_features=True`` keeps only
    instances that also have a feature row (``False``: only those without).
    """
    return get_problem(problem).list_instances(family=family, with_features=with_features)


def list_feature_instances(problem: str) -> List[str]:
    """Names of all instances with a feature row (some have no file)."""
    return get_problem(problem).list_feature_instances()


def load_instance(problem: str, name: str):
    """Parse instance ``name`` of ``problem`` into its ``Instance`` object."""
    return get_problem(problem).load_instance(name)


def get_features(problem: str, name: str) -> Dict[str, Any]:
    """Feature row of instance ``name`` as a ``{column: value}`` dict."""
    return get_problem(problem).get_features(name)


def feature_names(problem: str) -> List[str]:
    return get_problem(problem).feature_names()


def goal_function(problem: str, instance, solution, **kwargs) -> float:
    """Evaluate ``solution`` on ``instance`` (an ``Instance`` or its name)."""
    mod = get_problem(problem)
    if isinstance(instance, str):
        instance = mod.load_instance(instance)
    return solution.goal_function(instance, **kwargs)


def sense(problem: str) -> str:
    """``"min"`` or ``"max"``."""
    return get_problem(problem).SENSE


def summary() -> List[Dict[str, Any]]:
    """One row per problem: acronym, sense, #families, #instances, #feature rows."""
    rows = []
    for folder in list_problems():
        mod = get_problem(folder)
        rows.append(
            {
                "problem": folder,
                "acronym": mod.ACRONYM,
                "module": mod.__name__,
                "sense": mod.SENSE,
                "families": len(mod.list_families()),
                "instances": len(mod.list_instances()),
                "with_features": len(mod.list_instances(with_features=True)),
                "feature_rows": len(mod.list_feature_instances()),
                "features": len(mod.feature_names()),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("problems", help="list problems")
    sub.add_parser("summary", help="instance / feature counts per problem")
    p = sub.add_parser("families", help="list instance families of a problem")
    p.add_argument("problem")
    p = sub.add_parser("instances", help="list instance names of a problem")
    p.add_argument("problem")
    p.add_argument("--family")
    p.add_argument("--with-features", action="store_true")
    p.add_argument("--without-features", action="store_true")
    p.add_argument("--limit", type=int)
    p = sub.add_parser("features", help="print the feature row of an instance as JSON")
    p.add_argument("problem")
    p.add_argument("instance")
    args = ap.parse_args(argv)

    if args.cmd == "problems":
        for name in list_problems():
            print(f"{name:18s} {get_problem(name).ACRONYM}")
    elif args.cmd == "summary":
        rows = summary()
        cols = list(rows[0])
        print("  ".join(f"{c:>13s}" if c != "problem" else f"{c:18s}" for c in cols))
        for r in rows:
            print("  ".join(f"{r[c]!s:>13s}" if c != "problem" else f"{r[c]:18s}" for c in cols))
    elif args.cmd == "families":
        print("\n".join(list_families(args.problem)))
    elif args.cmd == "instances":
        wf = True if args.with_features else (False if args.without_features else None)
        names = list_instances(args.problem, family=args.family, with_features=wf)
        print("\n".join(names[: args.limit] if args.limit else names))
    elif args.cmd == "features":
        feats = get_features(args.problem, args.instance)
        print(json.dumps({k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in feats.items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
