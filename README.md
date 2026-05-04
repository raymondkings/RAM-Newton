# NRM-Newton

Gradient-based robot morphology optimization pipeline based on Neural Reachability Maps (NRM) and Newton physics simulation.

Practical course project — TUM CPS, Summer 2026.

**Team:** Julian Arkenau, Shiyuan Zhang, Jiyao Zhang, Raymond King Setia

**Supervisor**: Tim Walter

---

## Requirements

Python ≥ 3.10. Warp and Newton fall back to CPU if no NVIDIA GPU is available, but torch-based optimization will be significantly slower without one.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it, then:

```bash
uv sync
```

This creates a virtual environment and installs all dependencies from `uv.lock`. To run scripts:

```bash
uv run python <file>
```

## Usage

Run the pipeline with the default `config.json`:

```bash
uv run python main.py
```

To supply a different config file:

```bash
uv run python main.py --config path/to/my_config.json
```

The pipeline will:
1. Sample an initial 6-DOF robot morphology
2. Create the task environment with goal targets
3. Optionally optimize the morphology via the NRM gradient loop
4. Validate the (optimized) morphology against the task
5. Optionally open the Newton viewer for visual inspection

## Configuration

All runtime behaviour is controlled by a JSON file (default: `config.json` in the project root). The file is a flat object — pass any subset of these keys; unrecognised keys are forwarded to the pipeline as-is.

| Key | Type | Default | Description |
|---|---|---|---|
| `seed` | `int` | `0` | Global random seed. Controls initial morphology sampling and all torch/numpy RNG state, enabling reproducible runs. |
| `optimize` | `bool` | `true` | Run the NRM-based gradient optimisation loop. When `false` the sampled initial morphology is passed directly to validation, which is useful for quick sanity-checks or debugging validation alone. |
| `visualize` | `bool` | `true` | Open the Newton interactive viewer after validation. Requires a display; set to `false` for headless / CI runs. |

Example minimal config for a fast headless run:

```json
{
    "seed": 42,
    "optimize": false,
    "visualize": false
}
```
