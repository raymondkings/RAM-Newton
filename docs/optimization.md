# Optimization

Three optimization pipelines are available:

| Entry point | Optimizer module | Morphology search | Task |
| --- | --- | --- | --- |
| `main_candidate_selection_static.py` | `optim/nrm_alpha_random_selection.py` | discrete-alpha **candidate search** | static goal poses |
| `main_candidate_selection_trajectory.py` | `optim/nrm_alpha_random_selection_trajectory.py` | discrete-alpha **candidate search** | optimized trajectory |
| `main_gradient_trajectory.py` | `optim/nrm_trajectory.py` | **gradient descent** on one seed | optimized trajectory |

## Shared foundation

A morphology is a sequence of links `[α, a, d]` (MDH twist and two lengths).
Only lengths `[a, d]` are optimized; the twist `α` is fixed. Candidate search
enumerates it; the gradient pipeline inherits it from the seed.

The NRM scorer maps `model(morphology, pose) → logit`, where `sigmoid(logit)` is
the predicted reachability of that pose (`pose` = `[pos(3), rot_6d(6)]`).

Length preprocessing (`_preprocess_lengths`) runs in three steps:
1. **normalize:** rescale all lengths so total arm length is 1.
2. **squash:** zero out any segment shorter than `2·link_radius`. A straight-through
   estimator keeps this step differentiable.
3. **normalize again:** rescale back to total length 1.

The reachability loss pushes predicted probability toward 1 over every task pose
(`BCEWithLogits` for candidate search, `MSE(1, sigmoid)` for the trajectory
term), minimized by AdamW.

## A — candidate selection, static

`nrm_alpha_random_selection.py · optimize_morphology`. Fixed goal poses, no trajectory.

1. **Enumerate** alpha candidates per DOF in `CANDIDATE_DOF` (default 5,6,7) over
   $\{-\pi/2, 0, \pi/2\}$; drop $\geq 3$ consecutive zero twists.
2. **Sample** a valid initial `[a, d]` per candidate.
3. **Optimize** all candidates batched, with per-candidate **early stop** (prob
   moves < `1e-4` for 5 steps → freeze).
4. **Select:** distribution filter + last-link `d ≥ 0` → keep **top 2.5 %** by NRM
   prob → IK/FK validate survivors → pick max `ik_success_pose_rate`.

## B — candidate selection, trajectory

`nrm_alpha_random_selection_trajectory.py · optimize_morphology_and_trajectory`.
Uses the same candidate machinery and selection cascade as A (`candidate_base`).
The task is an ordered trajectory with fixed start/goal and optimized intermediate
poses. Each candidate alternates morphology/trajectory steps (see
[ratio](#alternating-ratio)) and is validated on its produced trajectory.

**Trajectory loss** (`_trajectory_loss_and_stats_batched`), weighted sum:

| Term | Weight | | Term | Weight |
| --- | --- | --- | --- | --- |
| reachability | 1.0 | | wall clearance (hinge < 0.025 m) | 500.0 |
| smoothness | 0.2 | | wall repulsion | 0.005 |
| distance-step variance | 6.0 | | rotation-step variance | 2.0 |
| position/rotation deviation, endpoint side | 0 (off) | | | |

## C — gradient, trajectory

`nrm_trajectory.py · optimize_morphology_and_trajectory`. No enumeration or
selection: one seed morphology with alternating length/trajectory steps (equivalent
to a single candidate of B). Validation runs inline every `eval_interval` steps.
Smoothing weights are softer (0.1 / 3.0 / 1.0). The output is the optimized seed.

## Alternating ratio

Trajectory pipelines (B, C) use alternating block-coordinate descent: lengths are
optimized with the path frozen, then poses are optimized with the morphology frozen.
Two constants set the schedule (same in both files):

```python
num_iteratives = 50   # outer rounds
num_steps      = 20   # gradient steps per phase
```

Each round runs 20 length steps then 20 pose steps, repeated 50 times: 1000
morphology steps and 1000 trajectory steps for 2000 gradient steps total.
