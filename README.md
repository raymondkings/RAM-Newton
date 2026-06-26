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

`benchmarks/pipeline_benchmark.py` sweeps the pipeline across N random seeds,
running each seed with and without collision avoidance, and writes a
crash-safe CSV plus a summary figure per algorithm to `benchmark_results/`.

The algorithms to run, and the sampler-param tuples to sweep for each, are
defined in [benchmarks/presets.json](benchmarks/presets.json) as a flat list
of "algorithms" entries — each pinned to its own `optim_algo` (one of
`nrm_alpha_random_selection`, `nrm_alpha_random_selection_trajectory`, or
`nrm_trajectory`), its sampler-param tuples, and an `output_dir`. The script
always runs every entry in that list, one after another, in a single
invocation:

```bash
uv run python benchmarks/pipeline_benchmark.py
```

Each entry point exposes its own sweepable sampler params:
`nrm_alpha_random_selection` has `(num_samples, num_line_samples,
num_extra_paths, repeat_start_goal)`, while the two trajectory algorithms have
just `num_poses`; a preset entry's config tuples must match its `optim_algo`'s
params (the script raises an error otherwise). To customize the seeds, swept
tuples, or output directories — or to compare the heuristic against the
gradient baseline — edit `benchmarks/presets.json`; no code changes needed.

Common options (apply to every algorithm in the list):

```bash
uv run python benchmarks/pipeline_benchmark.py --num-seeds 5            # quick smoke test
uv run python benchmarks/pipeline_benchmark.py --seeds-start 50         # start the seed range at 50
uv run python benchmarks/pipeline_benchmark.py --timeout 1800           # per-run timeout in seconds
uv run python benchmarks/pipeline_benchmark.py --output-dir my_results  # parent dir; each algorithm gets its own <my_results>/<optim_algo> subdir
```

Resume an interrupted sweep by pointing at the existing CSV — already-completed
`(seed, condition, *sampler_values)` rows are skipped. This only works when
`presets.json` has a single algorithm:

```bash
uv run python benchmarks/pipeline_benchmark.py --resume benchmark_results/<optim_algo>/benchmark_<timestamp>.csv
```

Or regenerate the figure for an algorithm's existing CSVs without rerunning:

```bash
uv run python benchmarks/pipeline_benchmark.py --replot
```

Outputs (per algorithm, under its `output_dir`):

- `benchmark_<optim_algo>_<timestamp>.csv` — one row per `(seed, condition, *sampler_values)`
- `benchmark_<optim_algo>_<timestamp>.png` — outcome breakdown, one subfigure per sampler-param tuple when more than one was swept
- `configs/` — the per-run config files
