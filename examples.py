"""Usage examples for the Combined NP-Hard Benchmark Python API.

Run from the repository root:

    python examples.py
"""
import benchmark as nb


def section(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# --------------------------------------------------------------------------- #
section("1. Problems, families and instances")
# --------------------------------------------------------------------------- #
print("problems:", nb.list_problems())
for problem in nb.list_problems():
    names = nb.list_instances(problem)
    print(
        f"{problem:18s} {len(nb.list_families(problem)):3d} families  "
        f"{len(names):6d} instances  e.g. {names[0]!r}"
    )

print("\nTSP families:      ", nb.list_families("TSP"))
print("first 3 CLKhard:    ", nb.list_instances("TSP", family="CLKhard")[:3])
print("2D BPP w/ features: ", len(nb.list_instances("BPP2D", with_features=True)))
print("aliases work:       ", nb.resolve("kp"), "|", nb.resolve("bin packing 1d"))

# --------------------------------------------------------------------------- #
section("2. Features of an instance (dict keyed by column name)")
# --------------------------------------------------------------------------- #
feats = nb.get_features("TSP", "CLKeasy_001")
for k in list(feats)[:5]:
    print(f"  {k:35s} {feats[k]}")
print("  ...", len(feats), "columns; algorithm columns:", [k for k in feats if k.startswith("algo_")])

# --------------------------------------------------------------------------- #
section("3. Load an instance, build a solution, compute the goal function")
# --------------------------------------------------------------------------- #
# Every problem module exposes Instance, Solution and a simple reference heuristic.
demos = [
    ("Bin-Packing-1D", "Falkenauer_u120_03", "first_fit_decreasing"),
    ("Bin-Packing-2D", "CLASS01_040_01", "shelf_first_fit"),
    ("GCP", "DIMACS-6700", "dsatur"),
    ("Knapsack problem", "EvenOddStrong096", "greedy"),
    ("QAP", "tai40a", "identity"),
    ("SAT", "uf20-91/uf20-0388_5124.cnf", "walksat"),
    ("TSP", "CLKhard_001", "nearest_neighbour"),
    ("Timetabling", "Ing0607-2", "greedy"),
]
for problem, name, heuristic in demos:
    mod = nb.get_problem(problem)
    inst = mod.load_instance(name)
    sol = getattr(mod, heuristic)(inst)
    print(
        f"{problem:18s} {name:30s} {heuristic:22s} "
        f"goal={sol.goal_function(inst):<12g} ({mod.SENSE})  feasible={sol.is_feasible(inst)}"
    )

# --------------------------------------------------------------------------- #
section("4. Hand-made solutions and penalties")
# --------------------------------------------------------------------------- #
from benchmark import knapsack, tsp  # noqa: E402  (modules importable by name)

inst = tsp.load_instance("random_002")
tour = tsp.Solution(list(range(inst.n)))  # visit cities in file order
print("TSP identity tour length:", tour.goal_function(inst))
print("same, exact distances:   ", round(tour.goal_function(inst, rounding=False), 2))

kp = knapsack.load_instance("EvenOddStrong096")
everything = knapsack.Solution([1] * kp.n)  # violates the capacity
print("KP take-all feasible?   ", everything.is_feasible(kp))
print("KP take-all goal:       ", everything.goal_function(kp))  # -inf (maximisation)
print("KP take-all, penalty=1: ", everything.goal_function(kp, penalty=1.0))
print("KP evaluate():          ", everything.evaluate(kp))

# --------------------------------------------------------------------------- #
section("5. Course timetabling: formulations and solution files")
# --------------------------------------------------------------------------- #
tt = nb.get_problem("Timetabling")
inst = tt.load_instance("Ing0607-2")  # ITC-2007 comp07
sol = tt.greedy(inst)
report = sol.evaluate(inst)
print("hard violations:", report["hard"])
print("soft violations:", report["soft"])
for f in tt.FORMULATIONS:
    print(f"  {f}: penalised goal = {sol.goal_function(inst, penalty=1000, formulation=f)}")
print("first lines of the .sol text:\n" + "".join(sol.to_text().splitlines(True)[:3]))

# --------------------------------------------------------------------------- #
section("6. Summary table")
# --------------------------------------------------------------------------- #
for row in nb.summary():
    print(row)
