"""VRAM benchmark for RAM-Newton pipeline batch-size parameters.

Peak VRAM is the maximum of two sequential phases, so both are measured:
  - candidate_batch_size          (NRM forward + backward during optimization)
  - distribution_batch_size       (post-optimization FK check)
Also measured: dynamic_rejection_batch_size, and direct_presampling_batch_size
(which is genuinely flat).

Run from the repo root:
    uv run python scripts/bench_vram.py

Output shows measured peak MiB at each tested value, then validates every
profile against the capacity of the GPU it targets.

This measures each phase in isolation, so the numbers are a LOWER bound on
full-pipeline usage. A PASS is necessary but not sufficient -- confirm with a
real run before trusting a profile on new hardware.
"""

import gc
import os

import torch

# Expandable segments reduce fragmentation OOM, matching the allocator used
# in practice on large runs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from methods._nrm_common import _load_model
from methods.candidate_selection.static import _candidate_loss_and_prob
from tasks.sampling.fixed_alpha_candidates import (
    _dynamic_reject_morph_batched,
    generate_alpha_candidates,
    sample_fixed_alpha_morphology_candidates,
)
from tasks.sampling.pose_sampler import (
    NUM_EXTRA_PATHS,
    NUM_SAMPLES,
    REPEAT_START_GOAL,
)

DOF = 7
SEQ_LEN = DOF + 1  # 8 for DOF=7
# Derived from the samplers rather than hardcoded: task_sampler() returns
# (num_extra_paths + 2) * num_samples + 2 * repeat poses (see its docstring).
# This is 460 at the defaults. A hardcoded 300 here silently under-measured
# every profile by ~1.5x, since peak scales with candidate_batch_size*num_poses.
NUM_POSES = (NUM_EXTRA_PATHS + 2) * NUM_SAMPLES + 2 * REPEAT_START_GOAL
LINK_RADIUS = 0.025
SEED = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise SystemExit("No CUDA device found — this benchmark requires a GPU.")

props = torch.cuda.get_device_properties(device)
total_mib = props.total_memory / (1024**2)
print(f"GPU: {props.name}  |  Total VRAM: {total_mib:.0f} MiB\n")


# ---------------------------------------------------------------------------


def mib(n_bytes: int) -> float:
    return n_bytes / (1024**2)


def reset() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def _run(label: str, values: list[int], fn) -> None:
    """Run fn(v) for each v, print a table row with peak VRAM."""
    print(f"=== {label} ===")
    print(f"  {'value':>8}  {'peak_MiB':>10}  {'pct_of_total':>14}")
    for v in values:
        reset()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            fn(v)
            pk = mib(torch.cuda.max_memory_allocated(device))
            pct = 100 * pk / total_mib
            print(f"  {v:>8}  {pk:>10.0f}  {pct:>13.1f}%")
        except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError):
            print(f"  {v:>8}  {'OOM':>10}")
        except Exception as e:
            print(f"  {v:>8}  ERROR: {e}")
    print()


# ---------------------------------------------------------------------------

print("Loading NRM model ...")
reset()
model = _load_model(device)
model_mib = mib(torch.cuda.memory_allocated(device))
print(
    f"  Model baseline: {model_mib:.0f} MiB  ({100 * model_mib / total_mib:.1f}% of total)\n"
)

# Synthetic task vector [num_poses, 9]
torch.manual_seed(SEED)
task_vec = torch.randn(NUM_POSES, 9, device=device)

# Phase 1 -----------------------------------------------------------------
# candidate_batch_size: NRM forward + backward pass. Scales with
# candidate_batch_size * num_poses -- ~55 MiB/candidate at the default 460
# poses on DOF=7 (it was ~36 when measured against 300 poses).


def bench_candidate_batch(batch_size: int) -> None:
    alpha = torch.zeros(batch_size, SEQ_LEN, 1, device=device)
    lengths = (
        torch.rand(batch_size, SEQ_LEN, 2, device=device) * 0.5 + 0.1
    ).requires_grad_(True)
    opt = torch.optim.AdamW([lengths], lr=0.01, weight_decay=0.0)
    opt.zero_grad(set_to_none=True)
    loss, _ = _candidate_loss_and_prob(
        model=model,
        alpha_candidates=alpha,
        length_candidates=lengths,
        task_vec=task_vec,
        link_radius=LINK_RADIUS,
    )
    loss.mean().backward()
    opt.step()


_run(
    f"candidate_batch_size  (NRM fwd+bwd, num_poses={NUM_POSES}, DOF={DOF})",
    [32, 64, 128, 150, 200, 300, 400, 512],
    bench_candidate_batch,
)

# Phase 2 -----------------------------------------------------------------
# dynamic_rejection_batch_size: runs _reject_morph (1000 FK evals/morphology).

n_morph = 256
_morph_fixed = torch.cat(
    [
        torch.zeros(n_morph, SEQ_LEN, 1, device=device),
        torch.full((n_morph, SEQ_LEN, 2), 0.3, device=device),
    ],
    dim=-1,
)


def bench_dynamic_rejection(batch_size: int) -> None:
    _dynamic_reject_morph_batched(morph=_morph_fixed, batch_size=batch_size)


_run(
    f"dynamic_rejection_batch_size  (morphology reject, n_morph={n_morph})",
    [32, 64, 128, 256],
    bench_dynamic_rejection,
)

# Phase 3 -----------------------------------------------------------------
# direct_presampling_batch_size: draw N morphologies from the original sampler.
# Expected to be negligible (morphology tensors are tiny).

generator = torch.Generator(device=device)
generator.manual_seed(SEED)
alpha_candidates, _, _ = generate_alpha_candidates(
    "ALL",
    seq_len=SEQ_LEN,
    device=device,
    generator=generator,
)
print(f"Alpha candidates for DOF={DOF}: {alpha_candidates.shape[0]}\n")


def bench_direct_presample(presample_size: int) -> None:
    torch.manual_seed(SEED)
    sample_fixed_alpha_morphology_candidates(
        alpha_candidates=alpha_candidates[:50],  # small set so it finishes fast
        seed=SEED,
        link_radius=LINK_RADIUS,
        batch_size=64,
        direct_presampling_batch_size=presample_size,
        dynamic_rejection_batch_size=32,
        logging=False,
    )


_run(
    "direct_presampling_batch_size  (warm-start morphology draw, expected ~flat)",
    [100, 300, 500, 1000],
    bench_direct_presample,
)

# Phase 3b ----------------------------------------------------------------
# distribution_batch_size: chunk size for check_morphology_distribution, which
# runs FK on-GPU over candidates that are still resident after optimization.
#
# This is NOT negligible and was previously left unmeasured. In full-pipeline
# runs, going from 256 to 512 raised peak by ~1800 MiB, and at 512 it sets a
# ~9080 MiB floor that dominates whenever candidate_batch_size < 192.


# Needs more morphologies than the largest batch size under test, or the chunk
# is clamped by the pool and every value above n_morph reports the same peak.
# Real DOF7 runs carry ~4000 candidates into this phase.
n_morph_dist = 2048
_morph_dist = torch.cat(
    [
        torch.zeros(n_morph_dist, SEQ_LEN, 1, device=device),
        torch.full((n_morph_dist, SEQ_LEN, 2), 0.3, device=device),
    ],
    dim=-1,
)


def bench_distribution_batch(batch_size: int) -> None:
    from methods.candidate_selection._common import _distribution_valid_mask

    _distribution_valid_mask(
        processed_morphologies=_morph_dist,
        batch_size=batch_size,
        logging=False,
        desc="bench",
    )


_run(
    f"distribution_batch_size  (post-optimization FK check, n_morph={n_morph_dist})",
    [128, 256, 512, 1024],
    bench_distribution_batch,
)

print(
    "Note: peak VRAM is max(candidate phase, distribution phase) -- both matter.\n"
    "fixed_alpha_batch_size controls the outer fixed-alpha loop chunk and is\n"
    "typically not a bottleneck.\n"
)

# Phase 4 -----------------------------------------------------------------
# Profile validation: measure actual peak for each profile's candidate_batch_size
# and compare against the expected values in vram_profiles.py.

from pipeline.vram_profiles import (  # noqa: E402
    _PROFILES,
    CUDA_CONTEXT_MIB,
    PROFILE_EXPECTED_PEAK_MIB,
    PROFILE_TARGET_VRAM_MIB,
    VRAM_BUDGET_FRACTION,
    profile_budget_mib,
)

TOLERANCE_PCT = 10.0  # within ±10% of the recorded peak → model still holds

# Two independent questions are asked per profile, and both must pass:
#   1. model:  does the measured peak still match PROFILE_EXPECTED_PEAK_MIB?
#   2. budget: does that peak fit in 85% of the GPU the profile is FOR?
# Only (2) is what a user cares about. Checking (1) alone is circular -- the
# expected values are a fit to this very measurement, so it can never reveal
# that a profile is too big for its target card.
print("=== Profile Validation ===")
print(
    f"  {'profile':>8}  {'cand_bs':>7}  {'target':>7}  {'expected':>9}"
    f"  {'actual':>8}  {'delta':>7}  {'budget85':>9}  {'result':>7}"
)
all_pass = True
for name, profile in _PROFILES.items():
    bs = profile["candidate_batch_size"]
    expected = PROFILE_EXPECTED_PEAK_MIB[name]
    target_gb = PROFILE_TARGET_VRAM_MIB[name] // 1024
    budget = profile_budget_mib(name)
    reset()
    try:
        bench_candidate_batch(bs)
        actual = mib(torch.cuda.max_memory_allocated(device))
        delta_pct = 100.0 * (actual - expected) / expected

        # bench_candidate_batch isolates the NRM phase, so it is a LOWER bound
        # on the full pipeline peak. Charge the CUDA context against the budget.
        fits_budget = (actual + CUDA_CONTEXT_MIB) <= budget
        matches_model = abs(delta_pct) <= TOLERANCE_PCT

        if not fits_budget:
            result = "FAIL"
        elif not matches_model:
            result = "DRIFT"
        else:
            result = "PASS"
        if result != "PASS":
            all_pass = False
        print(
            f"  {name:>8}  {bs:>7}  {str(target_gb) + 'GB':>7}  {expected:>9.0f}"
            f"  {actual:>8.0f}  {delta_pct:>+6.1f}%  {budget:>9.0f}  {result:>7}"
        )
    except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError):
        # An OOM is the single most important failure signal, so it must not be
        # allowed to leave all_pass untouched.
        all_pass = False
        print(
            f"  {name:>8}  {bs:>7}  {str(target_gb) + 'GB':>7}  {expected:>9.0f}"
            f"  {'OOM on this GPU':>28}  {'FAIL':>7}"
        )
    except Exception as e:
        all_pass = False
        print(f"  {name:>8}  {bs:>7}  ERROR: {e}  {'FAIL':>7}")

print()
if all_pass:
    print(
        f"All profiles fit within {VRAM_BUDGET_FRACTION:.0%} of their target GPU "
        "and match their recorded peaks."
    )
else:
    print(
        "FAIL = the profile does not fit its target GPU; its batch sizes must be\n"
        "reduced in pipeline/vram_profiles.py (peak is driven by\n"
        "candidate_batch_size, with a floor set by distribution_batch_size).\n"
        "DRIFT = it still fits, but the peak no longer matches\n"
        "PROFILE_EXPECTED_PEAK_MIB; re-record those values.\n"
        "NOTE: this benchmark measures the NRM phase in isolation, so it is a\n"
        "lower bound -- the full pipeline peaks higher. Confirm with a real run."
    )
