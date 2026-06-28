# Optimization

Currently we have 3 optimization pipelines:

| Entry point | Optimizer module | Morphology search | Task |
| --- | --- | --- | --- |
| `main_candidate_selection_static.py` | `optim/nrm_alpha_random_selection.py` | discrete-alpha **candidate search** | static goal poses |
| `main_candidate_selection_trajectory.py` | `optim/nrm_alpha_random_selection_trajectory.py` | discrete-alpha **candidate search** | optimized trajectory |
| `main_gradient_trajectory.py` | `optim/nrm_trajectory.py` | **gradient descent** on one seed | optimized trajectory |

## Shared foundation

- **Morphology** = a sequence of links `[α, a, d]` (MDH twist + two lengths).
  **Only lengths `[a, d]` are optimized.** The twist `α` is fixed — candidate
  search *enumerates* it, the gradient pipeline inherits it from the seed.
- **NRM scorer:** `model(morphology, pose) → logit`; `sigmoid(logit)` is the
  predicted reachability of that pose (`pose` = `[pos(3), rot_6d(6)]`).
- **Length preprocessing** (`_preprocess_lengths`) cleans up the raw lengths
  before scoring, in three steps:
  1. **normalize** — rescale all lengths so the total arm length is 1 (keeps the
     numbers in the range the NRM was trained on);
  2. **squash** — set to 0 any segment thinner than the link itself, i.e. shorter
     than the capsule diameter `2·link_radius`. These "stubs" aren't real links,
     so they're dropped. A straight-through estimator lets gradients pass through
     this hard cutoff, so the step stays differentiable;
  3. **normalize again** — rescale back to total length 1 now that stubs are gone.
- **Reachability loss:** push predicted prob → 1 over every task pose
  (`BCEWithLogits` for candidate search, `MSE(1, sigmoid)` for the trajectory
  term). Minimized by AdamW.

## A — candidate selection, static

`nrm_alpha_random_selection.py · optimize_morphology`. Fixed goal poses, no trajectory.

1. **Enumerate** alpha candidates per DOF in `CANDIDATE_DOF` (default 5,6,7) over
   `{-π/2, 0, π/2}`; drop ≥3 consecutive zero twists.
2. **Sample** a valid initial `[a, d]` per candidate.
3. **Optimize** all candidates batched, with per-candidate **early stop** (prob
   moves < `1e-4` for 5 steps → freeze).
4. **Select:** distribution filter + last-link `d ≥ 0` → keep **top 2.5 %** by NRM
   prob → IK/FK validate survivors → pick max `ik_success_pose_rate`.

## B — candidate selection, trajectory

`nrm_alpha_random_selection_trajectory.py · optimize_morphology_and_trajectory`.
Reuses A's candidate machinery and selection cascade (`candidate_base`), plus:

- Task is an ordered **trajectory**; start/goal fixed, **intermediate poses optimized**.
- Each candidate alternates morphology/trajectory steps (see [ratio](#alternating-ratio))
  and is validated on the trajectory it produced.

**Trajectory loss** (`_trajectory_loss_and_stats_batched`), weighted sum:

| Term | Weight | | Term | Weight |
| --- | --- | --- | --- | --- |
| reachability | 1.0 | | wall clearance (hinge < 0.025 m) | 500.0 |
| smoothness | 0.2 | | wall repulsion | 0.005 |
| distance-step variance | 6.0 | | rotation-step variance | 2.0 |
| position/rotation deviation, endpoint side | 0 (off) | | | |

## C — gradient, trajectory

`nrm_trajectory.py · optimize_morphology_and_trajectory`. Simplest path: no
enumeration, no selection — one seed morphology, alternating length/trajectory
steps (like a single candidate of B). Validation is **inline** every
`eval_interval` steps. Softer smoothing (0.1 / 3.0 / 1.0). Output morphology *is*
the optimized seed.

## Alternating ratio

Trajectory pipelines (B, C) optimize morphology and trajectory **one at a time,
never together** (alternating block-coordinate descent). They tune the lengths
for a while with the path frozen, then tune the path for a while with the
morphology frozen, and repeat.

Two module constants set the schedule (same in both files):

```python
num_iteratives = 50   # how many times we switch back and forth (outer rounds)
num_steps      = 20   # gradient steps per phase
```

One **round** looks like this, and it repeats 50 times:

```
20 length steps  (path frozen)  →  20 pose steps  (morphology frozen)
└──────── morphology phase ─────────┘ └──────── trajectory phase ────────┘
```

So per round the split is **20 : 20 = 1 : 1** between the two phases. Over all
50 rounds that is:

- morphology: `50 × 20 = 1000` steps
- trajectory: `50 × 20 = 1000` steps
- **2000 gradient steps total.**
