# NRM-Newton

Gradient-based robot morphology optimization pipeline based on Neural Reachability Maps (NRM) and Newton physics simulation. An optimized 6-DOF arm morphology is found for a given task, then validated with collision-free motion planning via cuRobo.

Practical course project — TUM CPS, Summer 2026.

**Team:** Julian Arkenau, Shiyuan Zhang, Jiyao Zhang, Raymond King Setia

**Supervisor**: Tim Walter

---

## Requirements

- Python ≥ 3.11
- NVIDIA GPU with CUDA (required by cuRobo for motion planning; optimization also benefits significantly)
- [cuRobo](https://nvlabs.github.io/curobo/latest/getting-started/installation.html) installed separately (not in `uv.lock`)

### GGIK Baseline And cuRobo Environment

The GGIK baseline uses Tim Walter / UTIAS generative-graphIK code and extra
dependencies that are not installed by the default `uv sync`. The current
adapter expects the generative-graphIK repo at `/tmp/generative-graphik` by
default:

```bash
git clone --branch revisions https://github.com/utiasSTARS/generative-graphik /tmp/generative-graphik
```

If `/tmp` is cleaned by the system, clone this repo again before running GGIK.

Install GGIK dependencies into the same virtual environment that contains
cuRobo. For example, after activating the cuRobo environment:

```bash
source /path/to/curobo/.venv/bin/activate

git config --global url."https://github.com/".insteadOf "ssh://git@github.com/"
git config --global url."https://github.com/".insteadOf "git@github.com:"

UV_CACHE_DIR=/tmp/uv-cache uv pip install "graphIK @ git+https://github.com/utiasSTARS/graphIK.git@generative_ik"
UV_CACHE_DIR=/tmp/uv-cache uv pip install torch-geometric
```

Check the installation:

```bash
python -c "import graphik, liegroups, torch_geometric; print('GGIK deps ok')"
```

When the cuRobo environment is active, run `main.py` with `--active` so `uv`
uses that environment instead of the project `.venv`:

```bash
uv run --active python main.py
```

Without `--active`, `uv run` may ignore the active cuRobo environment and fail
to import `graphik`, `liegroups`, `torch_geometric`, or cuRobo.

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

Run the full pipeline with the default `config.json`:

```bash
uv run python main.py
```

To supply a different config file:

```bash
uv run python main.py --config path/to/my_config.json
```

The pipeline:
1. Samples a random initial 6-DOF robot morphology
2. Builds the task — an L-shaped room environment with goal poses
3. Optimizes the morphology via the NRM gradient loop (100 iterations)
4. Plans a collision-free trajectory through all goal poses using cuRobo (GPU TrajOpt + graph search)
5. Animates the trajectory in the Newton/Viser viewer (if `visualize: true`)

## Configuration

All runtime behaviour is controlled by a JSON file (default: `config.json` in the project root).

| Key | Type | Default | Description |
|---|---|---|---|
| `seed` | `int` | `42` | Global random seed for morphology sampling and all RNG state. |
| `visualize` | `bool` | `true` | Open the Newton/Viser viewer after planning. Set to `false` for headless runs. On planning failure, also gates the debug static render. |
| `debug` | `bool` | `false` | Enable per-iteration loss logging during optimization. On planning failure, renders a static scene showing the collision geometry (requires `visualize: true`). |
| `ignore_ground` | `bool` | `false` | Exclude the ground plane from cuRobo collision checking. Useful for isolating whether ground collisions are causing planning failures. |
| `ignore_obstacles` | `bool` | `false` | Exclude task obstacles (the L-shaped walls) from cuRobo collision checking. Useful for testing reachability without environment constraints. |

Example config for a headless run with full collision checking:

```json
{
    "seed": 42,
    "visualize": false,
    "debug": false,
    "ignore_ground": false,
    "ignore_obstacles": false
}
```

Example config for debugging a planning failure (static scene with collision spheres shown):

```json
{
    "seed": 42,
    "visualize": true,
    "debug": true,
    "ignore_ground": false,
    "ignore_obstacles": false
}
```
