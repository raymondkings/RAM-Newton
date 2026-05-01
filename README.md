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

Or activate the environment directly:

```bash
source .venv/bin/activate
```

## Usage

To run the pipeline use:

```bash
python run_pipeline.py clean         # validate a clean morphology
python run_pipeline.py self_collide  # validate a self-colliding morphology
```

Both modes print a `ValidationResult` and open a browser-based Newton viewer.

## Interface Types

| Type | Key Fields |
|---|---|
| `Morphology` | `params (n_links, 3)` — columns `[α, a, d]`; `link_radius` |
| `Environment` | `obstacles: list[Box\|Sphere\|Capsule]`; `base_pose (4,4)` |
| `Task` | `environment`, `goal_poses (N,4,4)`, `reachable_region` |
| `ValidationResult` | `self_collision_free`, `env_collision_free`, collision counts |

## Validation Module

Takes an optimized `Morphology` and `Task`, builds a Newton scene, runs collision detection,
and returns a `ValidationResult` with self-collision and environment-collision counts.
`render_scene` opens a browser-based Newton viewer on the given port.
