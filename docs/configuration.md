# Configuration

Configuration is split across **two places**:

1. `config.json` — runtime parameters read by `main.py`.
2. **Module-level constants** at the top of
   `optim/nrm_alpha_random_selection.py` — these control the optimizer's
   strategy and are *not* surfaced in `config.json`.

Run with a different file via `uv run python main.py --config path/to/my.json`.

## `config.json` reference

| Key | Type | Default | Effect |
| --- | --- | --- | --- |
| `seed` | int | `0` | Global Random Number Generator seed (Python / Torch / CUDA / NumPy) and the alpha + validation generators. |
| `dof` | int | `7` | DOF of the initial sampled morphology. Only used when `use_cached_optimized_morphology` is `false`, mainly to obtain `link_radius` and the compute device. The candidate search DOFs come from `CANDIDATE_DOF` in source. |
| `visualize` | bool | `true` | Render the final plan in viser. |
| `debug` | bool | `true` | Verbose logging; on plan failure, render a static debug scene. Maps to the optimizer's `logging`. |
| `num_iterations` | int | `500` | Max optimization steps per candidate (before early stopping). |
| `eval_interval` | int | `20` | Validation cadence (in steps) for non-active optimizers. The active optimizer validates only the final survivors and has no per-step cadence. |
| `number_random_seed` | int | `32` | Number of IK seeds (random restarts) cuRobo uses during validation. |
| `percentage_poses` | number | `1` | Pose subset for validation. `≤1` = fraction of goal poses; `>1` = absolute count. |
| `candidate_batch_size` | int | `300` | Candidates optimized per chunk. |
| `distribution_batch_size` | int | `1024` | Candidates checked per chunk in the distribution filter. |
| `learning_rate_length` | float | `0.01` | AdamW learning rate for the `[a, d]` lengths. Passed as the optimizer's `learning_rate`. |
| `learning_rate_angle` | float | `0.05` | Unused in the active code path. |
| `ignore_ground` | bool | `true` | Skip the ground plane in collision checks (validation + planning). |
| `ignore_obstacles` | bool | `true` | Skip world obstacles in collision checks. |
| `plan_goal_start` | bool | `false` | If true, the final planner runs only a start→goal pair instead of the full planner pose set. |
| `use_cached_optimized_morphology` | bool | `true` | **If true, skip optimization** and load the latest `iteration == 2` morphology from `cached_optimization_csv`. If false, run the optimizer. (Code fallback if the key is absent: `false`.) |
| `cached_optimization_csv` | string | `"output"` | Directory (or file) the cached morphology is loaded from. |
| `plot.enabled` | bool | `false` | Generate candidate-selection plots after optimization. |
| `plot.output_dir` | string | `output/figures` | Where plots are written. |


## Hard-coded knobs (not in `config.json`)

These live at the top of `optim/nrm_alpha_random_selection.py`.

| Constant | Default | Effect |
| --- | --- | --- |
| `CANDIDATE_DOF` | `"all"` | Which DOFs the candidate search covers. `"all"` → `(5, 6, 7)`; also accepts `"5,6"`, `"7"`, etc. **This overrides config `dof` for the search.** |
| `DEFAULT_NUM_ALPHA_CANDIDATES` | `"ALL"` | How many alpha combinations to enumerate per DOF. `"ALL"` = exhaustive; an int = random subset. |
| `DEFAULT_CANDIDATE_BATCH_SIZE` | `64` | Fallback if `candidate_batch_size` is absent from the params dict. |
| `DEFAULT_DISTRIBUTION_BATCH_SIZE` | `128` | Fallback for `distribution_batch_size`. |
| `ZERO_ALPHA_RUN_EXCLUSION_LENGTH` | `3` | Reject alpha candidates with this many consecutive zero-twist entries. |
| `DELTA_EARLY_STOPPING` | `1e-4` | Per-candidate early stop: probability change below this counts as "stable". |
| `EARLY_STOPPING_PATIENCE` | `5` | Consecutive stable steps before a candidate is frozen. |
| `TOP_PROBABILITY_FRACTION` | `0.025` | Fraction of distribution-valid candidates (top by NRM prob) that proceed to IK/FK validation. |
| `LINK_RADIUS` | `0.025` | Capsule radius; sets the `2·r` minimum length threshold. Defined in `task/morphology_sampler.py` and on `Morphology`. |
