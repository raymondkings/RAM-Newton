# NRM-Newton

Gradient-based robot morphology optimization pipeline based on Neural Reachability Maps (NRM) and Newton physics simulation. An optimized 6-DOF arm morphology is found for a given task, then validated with collision-free motion planning via cuRobo.

Practical course project — TUM CPS, Summer 2026.

**Team:** Julian Arkenau, Shiyuan Zhang, Jiyao Zhang, Raymond King Setia

**Supervisor**: Tim Walter

---

## Requirements

- Python ≥ 3.11
- NVIDIA GPU with the CUDA 13 toolkit installed and a compatible driver.
  cuRobo is pulled in via the `cu13` extra (`nvidia-curobo[cu13]`) and
  `torch ≥ 2.11` ships its CUDA 13 build, so the GPU and driver must support
  CUDA 13.x. Verify with `nvidia-smi` (CUDA Version ≥ 13.0) and `nvcc --version`.
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Installation

Run these steps from the project root.

**1. Install uv** (if you don't have it):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

See the [uv install docs](https://docs.astral.sh/uv/getting-started/installation/)
for other methods.

**2. Install dependencies.** This creates a virtual environment and installs
everything from `uv.lock`, including the `generative-graphik` baseline:

```bash
uv sync
```

> The `generative-graphik` GGIK baseline is pulled from a
> [fork](https://github.com/jarkenau/generative-graphik) rather than upstream.
> The upstream `revisions` branch doesn't track the package `__init__.py` files,
> so a git build runs `find_packages()` over a tree with no packages and ships
> an empty wheel; the fork adds them so the build is complete. No manual clone
> is needed — uv handles it.

**3. Verify the install:**

```bash
uv run python -c "import graphik, liegroups, torch_geometric, generative_graphik.model; print('GGIK deps ok')"
```

## Usage

There are three pipeline entry points, each pairing a task formulation with an
optimizer:

- `main_candidate_selection_static.py` — candidate-selection heuristic over static
  task poses (`optim/nrm_alpha_random_selection.py`).
- `main_candidate_selection_trajectory.py` — **our heuristic**: candidate-selection
  search extended to jointly pick a morphology and a trajectory
  (`optim/nrm_alpha_random_selection_trajectory.py`).
- `main_gradient_trajectory.py` — alternating gradient-based optimization of
  morphology and trajectory (`optim/nrm_trajectory.py`), used as the baseline to
  compare the heuristic against.

Run any of them with the default `config.json`:

```bash
uv run python main_candidate_selection_static.py
uv run python main_candidate_selection_trajectory.py
uv run python main_gradient_trajectory.py
```

To supply a different config file:

```bash
uv run python main_gradient_trajectory.py --config path/to/my_config.json
```

## Benchmark

`benchmarks/pipeline_benchmark.py` sweeps one of the entry points above across N
random seeds, running each seed with and without collision avoidance, and writes a
crash-safe CSV plus a summary figure to `benchmark_results/`. Pick the entry point
with `--entry-point` (default `candidate_static`):

```bash
uv run python benchmarks/pipeline_benchmark.py --entry-point candidate_static
uv run python benchmarks/pipeline_benchmark.py --entry-point candidate_trajectory
uv run python benchmarks/pipeline_benchmark.py --entry-point gradient_trajectory
```

To compare our heuristic against the gradient baseline, run the same
seeds/conditions through both trajectory entry points into separate output
directories so their CSVs/figures don't mix:

```bash
uv run python benchmarks/pipeline_benchmark.py --entry-point candidate_trajectory \
    --output-dir benchmark_results/candidate_trajectory
uv run python benchmarks/pipeline_benchmark.py --entry-point gradient_trajectory \
    --output-dir benchmark_results/gradient_trajectory
```

Run the default sweep (100 seeds × 2 conditions):

```bash
uv run python benchmarks/pipeline_benchmark.py
```

Common options:

```bash
uv run python benchmarks/pipeline_benchmark.py --num-seeds 5            # quick smoke test
uv run python benchmarks/pipeline_benchmark.py --seeds-start 50         # start the seed range at 50
uv run python benchmarks/pipeline_benchmark.py --timeout 1800           # per-run timeout in seconds
uv run python benchmarks/pipeline_benchmark.py --output-dir my_results  # alternate output directory
```

Resume an interrupted sweep by pointing at the existing CSV — already-completed
`(seed, condition, *sampler_values)` rows are skipped:

```bash
uv run python benchmarks/pipeline_benchmark.py --resume benchmark_results/benchmark_<timestamp>.csv
```

Outputs:

- `benchmark_results/benchmark_<timestamp>.csv` — one row per `(seed, condition, *sampler_values)`
- `benchmark_results/benchmark_results.png` — outcome breakdown and convergence plots
- `benchmark_results/configs/` — the per-run config files

### Sampler-param sweep

Each entry point exposes its own sweepable sampler params: `candidate_static` has
`(num_samples, num_line_samples, num_extra_paths, repeat_start_goal)`, while the two
trajectory entry points have just `num_poses`. Named sweeps over a tuple of those
values are defined in [benchmarks/presets.json](benchmarks/presets.json) and
selected via `--preset`; a preset's tuples must match the current `--entry-point`'s
params (the script raises an error otherwise):

```bash
uv run python benchmarks/pipeline_benchmark.py --preset main   # 12-tuple production sweep, 20 seeds
```

`--preset` sets defaults for `--num-seeds` and `--output-dir`; both can be
overridden on the CLI. To customize the swept tuples or add a new preset,
edit `benchmarks/presets.json` — no code changes needed.

All tuples write to a single CSV (rows are keyed on the full sampler tuple),
so resume works the same way as the basic sweep:

```bash
uv run python benchmarks/pipeline_benchmark.py --preset main \
    --resume benchmark_results/benchmark_<timestamp>.csv
```

When the CSV holds more than one tuple, `benchmark_results.png` renders one
subfigure row per tuple.
