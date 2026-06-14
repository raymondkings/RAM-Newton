# Validation & Visualization

Validation happens at **two points** in the pipeline, and the final result is
shown interactively in viser.

## 1. Validation during optimization (fast, NRM-adjacent)

Each top-probability candidate is checked with real IK/FK before final selection
(`validation/optimization_validation.py`, using the `IK` / `FK` wrappers in
`util/kinematics.py`):

1. Build a temporary cuRobo robot (URDF) from the candidate morphology.
2. Sample a pose subset (`percentage_poses`) and solve IK with
   `number_random_seed` seeds.
3. Run FK on the solutions and measure error.

Metrics returned (and logged to the CSV):

- **`ik_success_pose_rate`** — fraction of goal poses with at least one IK
  solution. This is the **primary final-selection criterion**.
- **`pos_err`** (m), **`rot_err`** (rad), and **`se3_distance`** — the combined
  SE(3) error `√(pos²/8 + rot²/(2π²))`, the same weighting as the paper's SE(3)
  norm (`c₁=1/8`, `c₂=1/(2π²)`). These constants are chosen so each component is
  bounded to ½ — translation across the unit ball `B³` (diameter 2: `c₁·2² = ½`)
  and rotation up to `π` (`c₂·π² = ½`) — so position and orientation contribute
  equally and the combined distance maxes out at 1.

## 2. Final validation: cuRobo motion planning

The selected morphology is validated end-to-end with a **collision-free motion
plan** through the task goals (`validation/curobo_planner.py`,
`CuroboPlanner`). Unlike step 1 (which only checks pose reachability), this
proves the arm can actually *move* between goals without collision.

Construction writes the morphology to a temp URDF, builds collision spheres
(capsules → spheres) with a self-collision ignore map then builds the world scene.

Key methods:

| Method | Purpose |
| --- | --- |
| `plan_sequence(goals, start_q)` | Chains `start → goal[0] → … → goal[N-1]`, feeding each plan's end as the next start. Returns a `PlanResult` (full or partial path + `failed_at_goal`). |
| `is_q_feasible(q)` | True if a config is collision-free (self + world) and within joint limits. Used to reject a bad start. |
| `default_start_q()` | Rejection-samples the joint-limit box for a feasible start config. |
| `_diagnose_failure(...)` | Rich diagnostics on failure (IK / graph / trajopt stage, joint-limit and collision reports). |

`ignore_ground` / `ignore_obstacles` toggle the world collision geometry for both
IK and planning.

## 3. Visualization (viser)

`validation/render.py` renders the result in an interactive viser scene; the
mode depends on the outcome (`main.run_plan`):

- **Full success** → `animate_plan` plays the trajectory.
- **Partial** → animate up to `failed_at_goal`, with a translucent **ghost
  robot** at the closest reachable IK config.
- **No path** (debug) → `render_scene` shows a static, joint-draggable scene.

Scene elements include goal-pose axes (colored by reach status), the
end-effector frame, and the ground grid. Three live diagnostic panels track the
robot's state along the trajectory:

- **Joint-limit panel** — each joint's current position against its `[lower,
  upper]` limits, updated every frame so it's obvious when a joint runs into a
  bound.
- **Yoshikawa manipulability** — the manipulability index `√det(JJᵀ)` (soft
  variant for dof < 6) plotted over the trajectory; the EEF path is also colored
  green→red by the same value.
- **Singular values** — one progress bar per singular value σᵢ of the tool
  Jacobian, normalized to the largest σ seen on the trajectory. A bottom bar
  collapsing toward zero signals an approaching singularity.

`validation/visualize.py` is a simpler Newton-based animation fallback.

### Self-collision critical distance (d_crit)

During trajectory playback, `validation/critical_distance.py` shows how close
the robot comes to colliding **with itself**. It is purely diagnostic — it can
be toggled off on Viser and never interrupts playback
(`build_critical_distance_monitor` swallows all errors and returns `None` if
plotly is missing or the sphere model is inconsistent).

It reuses cuRobo's per-link collision spheres — the same model the planner
checks — so a single link may carry more than one sphere
(`CuroboPlanner.robot_spheres_world` gives the world-frame spheres per frame).

**How d_crit is computed.** Each frame, for every sphere pair *(i, j)*:

```
gap = ‖pᵢ − pⱼ‖ − rᵢ − rⱼ      # surface-to-surface; negative = penetration
```

`d_crit` is the **smallest gap over all valid pairs**. A pair is excluded when:

- both spheres belong to the same link, or
- the two links are effectively adjacent — the ignore map from
  [`build_self_collision_ignore`](../util/kinematics.py) excludes a pair when
  every link strictly between them is degenerate (zero-length), or
- either sphere is disabled (radius ≤ 0).

There is **no per-link representative sphere**: a single global `argmin` over all
valid sphere pairs returns the closest pair *(sᵢ, sⱼ)*, so for a link with many
spheres whichever one is nearest the other link wins.

**What it shows:**

| Element | Purpose |
| --- | --- |
| Plotly **timeline** | `d_crit` over the whole trajectory, with severity bands (safe > 0.10 m, warn > 0.02 m, danger ≤ 0.02 m, penetration < 0), the global-min marker, and a playback cursor. |
| Live **label** | Current `d_crit` value, severity dot, and the closest link-pair label (e.g. `L3 ↔ L5`). |
| 3-D **highlight** | Recolors the critical pair's spheres by severity and draws a surface-to-surface gap ruler whose length *is* `d_crit`. |

Relevant config: `visualize` (render at all), `debug` (static debug render on
failure), `ignore_ground` / `ignore_obstacles`, `plan_goal_start`.
