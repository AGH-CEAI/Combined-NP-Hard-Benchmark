# Combined NP-Hard Benchmark

A collection of benchmark instances for eight NP-hard combinatorial
optimisation problems, together with their **instance features** (from
[MATILDA](https://matilda.unimelb.edu.au/matilda/problems/opt) and, for SAT,
[SATzilla](https://www.cs.ubc.ca/labs/algorithms/Projects/SATzilla/) on
[SATLIB](https://www.cs.ubc.ca/~hoos/SATLIB/benchm.htm) instances) and a small,
dependency-free Python API to load instances, evaluate solutions and read
features.

| Folder | Problem | Sense | Families | Instances (files) | Feature rows |
|---|---|---|---:|---:|---:|
| `Bin-Packing-1D` | 1D bin packing | min | 11 | 8 815 | 9 055 |
| `Bin-Packing-2D` | 2D orthogonal bin packing | min | 4 | 1 204 | 420 |
| `GCP` | Graph colouring | min | 1 | 6 856 | 6 712 |
| `Knapsack problem` | 0-1 knapsack | **max** | 2 | 5 300 | 5 300 |
| `QAP` | Quadratic assignment | min | 12 | 802 | 802 |
| `SAT` | Satisfiability (MAX-SAT view) | min | 95 | 45 263 | 44 675 |
| `TSP` | Symmetric travelling salesman | min | 7 | 1 330 | 950 |
| `Timetabling` | Curriculum-based course timetabling | min | 2 | 7 021 | 4 428 |

## Layout

```
<Problem>/
    <Family 1>/ ...            instance files
    <Family 2>/ ...
    <features>.csv             MATILDA / SATzilla feature table
    <module>.py                Instance, Solution (goal function), loaders, get_features
    test_<module>.py           pytest tests of the module
benchmark.py                   registry: list_problems(), list_instances(), ... + CLI
examples.py                    runnable usage examples
test_benchmark.py              tests of the registry
```

Python 3.8+ and the standard library only. `pytest` is needed to run tests.

## Quick start

From the repository root:

```python
import benchmark as nb

nb.list_problems()
# ['Bin-Packing-1D', 'Bin-Packing-2D', 'GCP', 'Knapsack problem', 'QAP', 'SAT', 'TSP', 'Timetabling']

nb.list_families("TSP")                          # ['CLKeasy', 'CLKhard', ...]
nb.list_instances("TSP")                         # every instance with a file
nb.list_instances("TSP", family="CLKhard")       # one family
nb.list_instances("BPP2D", with_features=True)   # only those with a feature row

nb.get_features("TSP", "CLKeasy_001")
# {'feature_Mean_StdDevDist': 98.95, 'feature_Centroid_X': 188.14, ..., 'algo_CLK_Mean_Effort': 6312.19, ...}

tsp = nb.get_problem("TSP")                      # or: from benchmark import tsp
inst = tsp.load_instance("CLKeasy_001")
sol = tsp.Solution(list(range(inst.n)))          # a tour
sol.goal_function(inst)                          # tour length
sol.is_feasible(inst), sol.evaluate(inst)
```

Problems can be named by folder, module name or acronym, case-insensitively:
`"Knapsack problem"`, `"knapsack"`, `"KP"` all work.

Run `python examples.py` for a tour of every problem.

### Using a problem module directly

Every module is self-contained, so you can also copy it or put its folder on
`sys.path`:

```python
import sys; sys.path.insert(0, "QAP")
import qap

inst = qap.load_instance("tai40a")
qap.Solution(list(range(inst.n))).goal_function(inst)
```

### Command line

```bash
python benchmark.py problems
python benchmark.py summary
python benchmark.py families  SAT
python benchmark.py instances TSP --family CLKeasy --limit 5
python benchmark.py instances BPP1D --with-features
python benchmark.py features  QAP chr25a          # JSON
```

## Common API (every problem module)

| Name | Description |
|---|---|
| `PROBLEM`, `ACRONYM`, `SENSE` | folder name, short name, `"min"` / `"max"` |
| `list_families()` | instance family folders |
| `list_instances(family=None, with_features=None)` | names of instances that have a file |
| `list_feature_instances()` | names with a feature row (a few have no file) |
| `load_instance(name)` | parse into an `Instance` dataclass |
| `instance_path(name)` | path of the instance file |
| `get_features(name)` | `{column: value}` for the instance — `feature_*` columns, `algo_*` performance columns and the source/family column; numbers are `float` |
| `feature_names()` | the columns returned by `get_features` |
| `Instance` | parsed instance (dataclass; can also be built by hand) |
| `Solution` | solution representation with the methods below |
| `Solution.goal_function(instance, penalty=None)` | objective of a feasible solution. Infeasible: `inf` (`-inf` for maximisation), or `objective ± penalty * violation` when `penalty` is given |
| `Solution.objective(instance)` | raw objective, ignoring feasibility |
| `Solution.violation(instance)` | amount of constraint violation (`0` = feasible) |
| `Solution.is_feasible(instance)`, `Solution.evaluate(instance)` | feasibility flag / `{"objective", "violation", "feasible", ...}` |
| a reference heuristic | e.g. `first_fit_decreasing`, `dsatur`, `nearest_neighbour` — useful as baselines and examples |

## Problems

### Bin-Packing-1D — `bpp1d.py`
* **File:** `n`, capacity `C`, then `n` weights.
* **Solution:** `Solution(assignment)` — `assignment[i]` is the bin label of item `i` (any hashable); `Solution.from_bins([[0, 3], [1, 2]], n)`.
* **Goal:** number of bins used (min). Violation = total overload + missing items.
* **Heuristic:** `first_fit_decreasing`. `Instance.lower_bound()` gives ⌈Σw / C⌉.
* **Names:** file name without `.txt`. 240 `rgen__*` feature rows have no file.

### Bin-Packing-2D — `bpp2d.py`
* **File:** JSON with one bin type (`Objects`) and items with `Length` (width), `Height`, `Demand`.
* **Names:** the JSON `Name` field (file numbers repeat across folders). `GA` / `GAInd` instances have no features.
* **Solution:** one `Placement(bin, x, y, rotated=False)` per item copy (items with `Demand > 1` are expanded; `Instance.item_type` maps back).
* **Goal:** number of bins (min). Violation = unplaced items + forbidden rotations + area outside bins + overlap area. Orientation is fixed unless `load_instance(name, allow_rotation=True)`.
* **Heuristic:** `shelf_first_fit` (FFDH shelves).

### GCP — `gcp.py`
* **File:** DIMACS `p edge n m` / `e u v`, vertices 1..n.
* **Solution:** `Solution(colors)` — `colors[v]` for vertex `v` (0-based = DIMACS `v+1`).
* **Goal:** number of colours (min). Violation = conflicting edges + uncoloured vertices.
* **Heuristic:** `dsatur`.
* **Names:** the CSV names. Most files are numbered; the mapping (verified via `feature_Density` for every row) is `Bipartite-k → k+1`, `Cycle_Driven-k → 1001+k`, `Geometric-k → 2001+k`, `Girth_And_Degree_Inhibited-k → 3001+k`, `IDD_Graph-k → 4001+k`, `Weight_Biased-k → 5001+k`, `DIMACS-N → N`; others (`round-robin-*`, `1a`…, `*.col`) use their file name. `Flat*` and `Random*` feature rows have no graph file. `list_instances("GCP", family="DIMACS")` filters by the CSV `Source`.

### Knapsack problem — `knapsack.py`
* **File:** name, type, capacity, then `weight profit` per item.
* **Solution:** `Solution(x)` with `x[i] ∈ {0, 1}`; `Solution.from_indices([...], n)`.
* **Goal:** total profit (**max**). Violation = capacity excess.
* **Heuristic:** `greedy` (ratio order + best single item).

### QAP — `qap.py`
* **File:** QAPLIB — `n`, matrix `A`, matrix `B`.
* **Solution:** `Solution(permutation)` — `permutation[i]` = location of facility `i` (0-based); `Solution.from_one_based(...)` for QAPLIB `.sln` permutations.
* **Goal:** `Σ_ij A[i][j] · B[p[i]][p[j]]` (min).

### SAT — `sat.py`
* **File:** DIMACS CNF (SATLIB `%` end marker handled).
* **Names:** path relative to `SAT/`, e.g. `uf20-91/uf20-0388_5124.cnf` (as in the CSV `file` column).
* **Solution:** `Solution(assignment)` — `assignment[v]` (bool) for variable `v+1`; `Solution.from_literals([1, -2, 3], n)`.
* **Goal:** number of unsatisfied clauses (min; `0` ⇔ model, `is_model()`). `uuf*` families are unsatisfiable.
* **Heuristic:** `walksat`. Features: 138 SATzilla features.

### TSP — `tsp.py`
* **File:** TSPLIB `EUC_2D`, 100 cities.
* **Names:** `<family>_<k>` with `k = 001..190` = the *k*-th file of the folder (some folders count files from `000`, others from `001`; verified against the centroid features). `easyCLK-hardLKCC` and `hardCLK-easyLKCC` have no features.
* **Solution:** `Solution(tour)` — permutation of city indices `0..n-1`.
* **Goal:** closed tour length with TSPLIB rounding (`nint`); `goal_function(inst, rounding=False)` for exact distances.
* **Heuristic:** `nearest_neighbour`.

### Timetabling — `timetabling.py`
* **File:** extended ECTT (curriculum-based course timetabling; ITC-2007 track 3 and the formats of Bonutti et al. 2012). `ROOM_CONSTRAINTS` lists rooms *unsuitable* for a course.
* **Names:** the CSV name — the file name where it matches, otherwise the `Name:` header (e.g. `real/EA15.ectt` → `EasyAcademy15`). Copies present in both `real/` and `random/` resolve to `real/`.
* **Solution:** list of `Assignment(course, room, day, period)`, one per lecture; `Solution.from_text` / `to_text` read and write the standard `.sol` format.
* **Goal:** weighted soft cost (min) of a solution without hard violations. Hard: Lectures, Conflicts, RoomOccupancy, Availability (+ invalid ids). Formulation is selectable — `goal_function(inst, formulation="UD2")`:

  | | S1 RoomCap | S2 MinDays | S3 Isolated | S4 Windows | S5 RoomStab | S6 MinMaxLoad | S7 Travel | S8 RoomSuit | S9 Double |
  |---|---|---|---|---|---|---|---|---|---|
  | UD1 | 1 | 5 | 1 | | | | | | |
  | **UD2** (ITC-2007, default) | 1 | 5 | 2 | | 1 | | | | |
  | UD3 | 1 | | | 4 | | 2 | | 3 | |
  | UD4 | 1 | 1 | | 1 | | 1 | | hard | 1 |
  | UD5 | 1 | 5 | 1 | 2 | | 2 | 2 | | |

  `Solution.evaluate(inst)` returns per-constraint hard and soft counts.
* **Heuristic:** `greedy` (may leave a few lectures unscheduled on hard instances).

## Setup with uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # once, if uv is not installed
uv sync                                            # creates .venv with pytest
uv run python examples.py
uv run python benchmark.py summary
uv run pytest
```

## Running the tests

```bash
pip install pytest
python -m pytest              # all problem folders + registry (~20 s)
python -m pytest TSP          # one problem
```

The tests check goal functions on small hand-computed instances, verify that
instance names map to the right files by comparing parsed instances with
feature values (sizes, densities, centroids, capacity fractions, ...), and
run the reference heuristics on real instances.

## Data notes

* macOS resource-fork files (`._*`) and `.DS_Store` are ignored everywhere.
* Feature CSV first-column names differ (`Instances`, `instances`, `file`); `get_features` hides this.
* TSP: the CSV row `random_001` is missing one value upstream, so its columns are shifted; it is returned as stored.
* Some feature rows have no instance file (BPP-1D `rgen__*`, GCP `Flat*`/`Random*`), and some instances have no features; use `with_features=` to filter.

## References

* MATILDA — Melbourne Algorithm Test Instance Library with Data Analytics, <https://matilda.unimelb.edu.au/matilda/problems/opt>
* H. H. Hoos, T. Stützle. SATLIB, <https://www.cs.ubc.ca/~hoos/SATLIB/benchm.htm>
* A. Bonutti, F. De Cesco, L. Di Gaspero, A. Schaerf. Benchmarking curriculum-based course timetabling: formulations, data formats, instances, validation, visualization, and results. *Ann. Oper. Res.* 194 (2012).
* R. E. Burkard, S. E. Karisch, F. Rendl. QAPLIB — a quadratic assignment problem library.
* G. Reinelt. TSPLIB — a traveling salesman problem library. *ORSA J. Comput.* 3 (1991).
