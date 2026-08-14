# Architecture

How the pipeline is put together: the stages a run moves through, the data
passed between them, and where each concept from the RAM paper lives in the
code. For the file tree, see [Repository Layout](../README.md#repository-layout)
in the README.

## Pipeline overview

Every entry point in [`scripts/`](../scripts/) follows the same shape: build a
`Task` (an environment plus a set of goal poses), optimize a morphology against
the frozen RAM checkpoint (`data/weights/checkpoint_5-7.pth`), then validate the
winner with collision-free motion planning via cuRobo and render it in viser.

![The four-stage optimization pipeline, from the project poster](media/optimization_pipeline.png)

**1. Kinematic structure sampling.** Twist angles are discrete, so the search
starts by enumerating alpha candidates from `{-π/2, 0, π/2}` per DoF and
sampling a valid initial morphology for each. Candidates with three or more
consecutive zero-twist links are excluded during generation.

**2. Alternating optimization.** Link lengths `[a, d]` are optimized with AdamW
against the frozen surrogate; twists stay fixed at their discrete values. The
trajectory step runs only in the trajectory pipelines — it holds the morphology
fixed and moves the intermediate poses, leaving start and goal pinned. It does
not run at all in `candidate_selection_static`, which optimizes the morphology
alone.

**3. Filtering and selection.** Survivors go through the distribution checker
(link-validity rejection, collision, Yoshikawa manipulability), then the top
`TOP_PROBABILITY_FRACTION` — 2.5% — by RAM probability is kept.

**4. Final selection.** The top candidates get IK/FK validation, are ranked by
IK pose success rate with `_tie_score` breaking ties, and the winner is handed
to cuRobo. `k` here is the `num_plan_candidates` parameter swept in the
[evaluation harness](../README.md#evaluation): with `k > 1`, each of the top k
candidates is planned until one succeeds.

Setting `use_cached_optimized_morphology` skips stages 1–4 entirely and replays
the most recent morphology from `output/`.

## Data model

The dataclasses passed between stages live in [`core/`](../core/) and are
documented inline.

| Type | Contents |
| --- | --- |
| `Morphology` | `params` of shape `[n_links, 3]`, one MDH token `[α, a, d]` per link, plus `link_radius`. `n_links` is DoF + 1, since the end-effector frame also needs a transform. Joint angles are *not* part of a morphology. |
| `Task` | An `Environment`, `goal_poses` of shape `[n_goals, 4, 4]` visited in order, and an optional `start_q` (defaults to the all-zeros rest pose). |
| `Environment` | A list of static obstacles, each a `Box` or a `Sphere`. |
| `PlanResult` | `success`, the joint-config `path`, and on failure `failed_at_goal` plus the `best_ik_q` reached. |

## Paper ↔ code map

Where each concept lands in the source. Entries are derived from the code and
the [poster](poster/); anything that depends on the paper's own terminology is
marked *(inferred)* and worth confirming against
[arXiv:2606.09108](https://arxiv.org/abs/2606.09108).

| Concept | Implementation |
| --- | --- |
| Morphology as a sequence of MDH tokens `[α, a, d]` | [`core/morphology.py`](../core/morphology.py) |
| Discrete twist set `{-π/2, 0, π/2}` | `ALPHA_VALUES` in [`tasks/sampling/fixed_alpha_candidates.py`](../tasks/sampling/fixed_alpha_candidates.py) |
| Reachability surrogate | `MLP` in [`methods/nrm_model.py`](../methods/nrm_model.py): an LSTM `Encoder` over the link sequence feeding an MLP `Decoder` |
| Reachability probability for a pose | `torch.sigmoid(logit)` over the decoder output, averaged across poses |
| Pose encoding, SE(3) → 9D | `_se3_to_vector` in [`methods/_nrm_common.py`](../methods/_nrm_common.py): 3D position + 6D rotation |
| Length preprocessing (normalize → squash → normalize) | `_preprocess_lengths`, same file |
| Straight-through gradient past the squash | `SquasherSTE`, same file *(inferred: named for the STE technique, not necessarily the paper's term)* |
| Frozen checkpoint, gradients to inputs only | `_load_model`, same file — weights get `requires_grad_(False)` |
| Top-2.5% probability cut | `TOP_PROBABILITY_FRACTION` in [`methods/candidate_selection/_common.py`](../methods/candidate_selection/_common.py) |
| Distribution validity check | [`validation/distribution_checker.py`](../validation/distribution_checker.py), using FK, collision, and Yoshikawa manipulability |
| Tie-break heuristic among final candidates | `_tie_score` in `methods/candidate_selection/_common.py` |
| IK/FK validation | [`validation/optimization_validation.py`](../validation/optimization_validation.py) |
| Forward kinematics / MDH transforms | [`kinematics/kinematics.py`](../kinematics/kinematics.py), [`kinematics/mdh.py`](../kinematics/mdh.py) |
| Self-collision via capsule pairs | [`kinematics/self_collision.py`](../kinematics/self_collision.py) |
| Collision-free motion planning | [`planning/curobo_planner.py`](../planning/curobo_planner.py) |
| success@k | `num_plan_candidates`, threaded through [`pipeline/common.py`](../pipeline/common.py) and [`evaluation/run.py`](../evaluation/run.py) |
