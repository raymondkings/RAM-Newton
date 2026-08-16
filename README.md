# RAM-Newton

A pipeline for robot morphology optimization. Given a target task, it uses
[Reachability Across Morphologies (RAM)](https://arxiv.org/abs/2606.09108) to
optimize a 5–7 DoF arm morphology, then validates the result with
[cuRobo](https://github.com/NVlabs/curobo) collision-free motion planning and
renders it in the [Newton physics simulator](https://github.com/newton-physics/newton).

Practical course project at TUM CPS, Summer 2026.

**Team:** Julian Arkenau, Shiyuan Zhang, Jiyao Zhang, Raymond King Setia

**Supervisor:** Tim Walter

## Table of Contents

- [Demo](#demo)
- [Requirements](#requirements)
- [Installation](#installation)
- [Repository Layout](#repository-layout)
- [Usage](#usage)
- [GPU Configuration](#gpu-configuration)
- [Documentation](#documentation)
- [Evaluation](#evaluation)
- [License and Citation](#license-and-citation)

## Demo

![RAM-Newton pipeline: morphology optimization, motion planning, and simulation](docs/media/project_summary.gif)

## Requirements

- Python 3.11 or newer
- NVIDIA GPU with CUDA 13.x (both cuRobo and torch use the CUDA 13 build). Verify with `nvidia-smi`.
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
everything from `uv.lock`:

```bash
uv sync
```

**3. Verify the install:**

```bash
uv run python -c "import torch, curobo, newton, viser; print(f'ok, cuda={torch.cuda.is_available()}')"
```

## Repository Layout

The library is organized as concept-named packages at the repo root. The
entry-point scripts, evaluations, tests, and data sit alongside them.

```text
paths.py                # canonical PROJECT_ROOT / data / weights / config paths
core/                   # data model: Morphology, Task, Environment, results
kinematics/             # forward kinematics, MDH parameters, self-collision
tasks/                  # task setup
├── environment.py      #   the single wall the arm has to reach around
└── sampling/           #   morphology + task-pose samplers, candidate cache
methods/                # optimization approaches (see below)
├── nrm_model.py        #   shared RAM surrogate network (MLP)
├── _nrm_common.py      #   checkpoint loading, pose/morphology encoding
├── candidate_selection/  # discrete-alpha candidate search (our heuristic)
├── nrm_gradient/       #   gradient-based optimizer (baseline)
├── baselines/          #   direct-IK comparison baselines (IFT-JAX, IFT-torch, HJCD-IK)
└── legacy/             #   earlier attempts that didn't pan out
planning/               # cuRobo collision-free motion planning
validation/             # reachability/collision validation of a morphology
visualization/          # live viser 3-D rendering, ground plane, d_crit, poster figures
logutils/               # optimization CSV logging/reading, timing
pipeline/               # shared plumbing: entry-point common.py, VRAM profiles
postprocess/            # figures from logged CSVs

scripts/                # the three pipeline entry points + bench_vram.py
evaluation/             # seed-sweep harness + its config.json
tests/                  # unit tests
docs/                   # deep reference documentation, demo media, poster
data/weights/           # frozen RAM checkpoints + metadata.json
config.json             # default pipeline config
```

## Usage

> **TODO:** link the quickstart Jupyter notebook here once it lands.

There are three pipeline entry points, each pairing a task formulation with an
optimizer:

- `scripts/candidate_selection_static.py` runs the candidate-selection
  heuristic over static task poses (`methods/candidate_selection/static.py`).
- `scripts/candidate_selection_trajectory.py` is **our heuristic**. It extends
  the candidate-selection search to jointly pick a morphology and a trajectory
  (`methods/candidate_selection/trajectory.py`).
- `scripts/gradient_trajectory.py` runs alternating gradient-based optimization
  of morphology and trajectory (`methods/nrm_gradient/trajectory.py`). It serves
  as the baseline to compare the heuristic against.

Run any of them with the default `config.json`:

```bash
uv run python scripts/candidate_selection_static.py
uv run python scripts/candidate_selection_trajectory.py
uv run python scripts/gradient_trajectory.py
```

To supply a different config file:

```bash
uv run python scripts/gradient_trajectory.py --config path/to/my_config.json
```

## GPU Configuration

GPU memory usage scales with the batch-size parameters in `config.json`. The
`"vram_profile"` key sets all of them at once based on available VRAM:

```json
{
  "vram_profile": "high"
}
```

Each profile stays within 85% of the target GPU's memory, leaving roughly 15%
free for the OS and the cuRobo planner:

| Profile  | Target GPU | Memory needed | Example GPUs                          |
|----------|------------|---------------|---------------------------------------|
| `low`    | 8 GB       | 5.4 GB        | RTX 3070, RTX 4060, RTX 4060 Ti 8 GB  |
| `medium` | 12 GB      | 8.9 GB        | RTX 3080 Ti, RTX 4070, RTX 4070 Super |
| `high`   | 16 GB      | 10.6 GB       | RTX 4080, RTX 4080 Super, RTX 5080    |
| `ultra`  | 32 GB      | 24.4 GB       | RTX 5090                              |

Individual keys override the profile when both are set:

```json
{
  "vram_profile": "high",
  "candidate_batch_size": 300
}
```

## Documentation

Deep reference lives in [`docs/`](docs/). Start at the
[documentation index](docs/index.md):

- [docs/architecture.md](docs/architecture.md) — the four-stage pipeline, the
  data model passed between stages, and the paper $\leftrightarrow$ code map.
- [docs/optimization.md](docs/optimization.md) — deep walkthrough of the
  morphology optimizer (differentiable preprocessing, batched optimization,
  early stopping, selection cascade).
- [docs/validation.md](docs/validation.md) — IK/FK validation, cuRobo motion
  planning, and the viser visualization.
- [docs/configuration.md](docs/configuration.md) — every `config.json` key plus
  the hard-coded knobs that live in source.

The final project poster, *Task-Driven Robotic Arm Optimization*, is in
[docs/poster/](docs/poster/). It covers the motivation and research question,
the optimizer benchmark against the direct-IK baselines, the planning success
rates, and what didn't work.

## Evaluation

`evaluation/run.py` sweeps the pipeline across N random seeds under each
planning condition. It writes a crash-safe CSV plus a summary figure per
algorithm to `evaluation_results/<optim_algo>/`.

Everything is driven by [evaluation/config.json](evaluation/config.json); no
code changes are needed. It holds:

- `conditions` — the named `ignore_obstacles` / `ignore_ground` combinations
  every seed is run under. The shipped default compares obstacles off against
  obstacles on, both with the ground ignored.
- `num_seeds` — how many seeds to sweep. An algorithm entry can override it with
  its own `num_seeds`.
- `algorithms` — a flat list of entries, each pinned to its own `optim_algo`
  (one of `candidate_selection_static`, `candidate_selection_trajectory`, or
  `gradient_trajectory`), its sampler-param tuples under `configs`, and an
  optional `output_dir` (defaults to `evaluation_results/<optim_algo>`).

The script always runs every entry in that list, one after another, in a single
invocation:

```bash
uv run python evaluation/run.py
```

Each entry point exposes its own sweepable sampler params, and an entry's
`configs` tuples must match them positionally or the script raises an error:

| `optim_algo` | sampler params |
| --- | --- |
| `candidate_selection_static` | `num_samples`, `num_line_samples`, `num_extra_paths`, `repeat_start_goal`, `num_plan_candidates` |
| `candidate_selection_trajectory` | `num_poses`, `num_plan_candidates` |
| `gradient_trajectory` | `num_poses` |

`num_plan_candidates` (success@k) can be supplied as the last element of each
`configs` tuple or as a standalone key that crosses with `configs` (e.g.
`"num_plan_candidates": [1, 10]` doubles the sweep). With `k > 1`, a seed counts
as a success if any of the top k candidates plans successfully. `gradient_trajectory`
has no candidate pool and does not take this param.

Common options (apply to every algorithm in the list):

```bash
uv run python evaluation/run.py --num-seeds 5            # quick smoke test
uv run python evaluation/run.py --seeds-start 50         # start the seed range at 50
uv run python evaluation/run.py --timeout 1800           # per-run timeout in seconds
uv run python evaluation/run.py --output-dir my_results  # parent dir; each algorithm gets its own <my_results>/<optim_algo> subdir
uv run python evaluation/run.py --no-results-csv         # disposable sweep: no CSV, report, or resume
```

Resume an interrupted sweep by pointing at the existing CSV. Already-completed
`(seed, condition, *sampler_values)` rows are skipped. This only works when
`evaluation/config.json` has a single algorithm:

```bash
uv run python evaluation/run.py --resume evaluation_results/<optim_algo>/evaluation_<optim_algo>_<timestamp>.csv
```

Or regenerate the figure for an algorithm's existing CSVs without rerunning:

```bash
uv run python evaluation/run.py --replot
```

Outputs (per algorithm, under its `output_dir`):

- `evaluation_<optim_algo>_<timestamp>.csv` holds one row per `(seed, condition, *sampler_values)`.
- `evaluation_<optim_algo>_<timestamp>.png` shows the outcome breakdown, one subfigure per sampler-param tuple when more than one was swept.
- `configs/` holds the per-run config files.

## License and Citation

Released under the [MIT License](LICENSE). Parts of `methods/` derive from
[TimWalter/ram](https://github.com/TimWalter/ram); see the headers in those
files.

If you use this software, cite it via [CITATION.cff](CITATION.cff).
