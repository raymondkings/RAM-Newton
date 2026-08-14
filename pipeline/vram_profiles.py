"""VRAM profiles for the NRM-Newton pipeline.

Each profile maps a GPU-memory class to a set of batch-size parameters that
together stay within the available VRAM.  New contributors should set
``"vram_profile"`` in ``config.json`` and leave the individual batch-size keys
unset; expert users can add explicit overrides on top of any profile.

Override semantics: explicit keys in config.json always win over the profile.

Available profiles
------------------
Measured on RTX 5090, DOF=7, 460 task poses (the real pose count produced by
``task_sampler``: ``(num_extra_paths + 2) * num_samples + 2 * repeat``), running
the full ``candidate_selection_static`` pipeline end to end.  The static
pipeline is the binding case -- ``candidate_selection_trajectory`` peaks lower
at the same settings (9.1 GB vs 14.4 GB at candidate_batch_size=256).

Peak VRAM is the MAXIMUM of two sequential phases, not a single driver:

    peak_MiB ~= max(
        173 + 55.43 * candidate_batch_size,      # NRM fwd+bwd over all poses
        distribution_phase(distribution_batch_size),
    )

The candidate term scales with ``candidate_batch_size * num_poses``; the 55.43
MiB/candidate slope is specific to 460 poses.  The distribution term is NOT
negligible -- measured on the same hardware:

    distribution_batch_size   128     256     512    1024
    peak_MiB                 2392    4620    9076   17989

At 512 it sets a ~9076 MiB floor that dominates whenever candidate_batch_size
< 192, so lowering candidate_batch_size alone will not get a run under budget
once that floor binds.  Both knobs have to move together.

Each profile is sized so its measured peak, plus roughly 500 MiB of CUDA
context, stays within 85% of the target GPU -- leaving ~15% for the OS and the
cuRobo planner.  Every value below was confirmed by a real run with the
allocator capped at that 85% budget.

Peaks are torch.cuda.max_memory_allocated (live tensors).  nvidia-smi reports
considerably more on a roomy GPU, because the caching allocator expands into
free VRAM and only reclaims under pressure -- do not read it as a requirement.

Measured with CANDIDATE_DOF="all" as well as 7: running all three DOF groups
peaks at 5495 MiB vs 5494 MiB for DOF 7 alone, i.e. memory is released between
groups and the sequence does not accumulate.

Profile  target_GPU  candidate_batch  measured_peak  85%_budget
low      8 GB        96                5494 MiB      6963 MiB
medium   12 GB       160               9077 MiB     10445 MiB
high     16 GB       192              10816 MiB     13926 MiB
ultra    32 GB       448              25005 MiB     27853 MiB

Run ``uv run python scripts/bench_vram.py`` to re-check on your own GPU.
"""

_PROFILES: dict[str, dict[str, int]] = {
    "low": {
        "candidate_batch_size": 96,
        "distribution_batch_size": 256,
        "direct_presampling_batch_size": 500,
        "dynamic_rejection_batch_size": 128,
    },
    "medium": {
        "candidate_batch_size": 160,
        "distribution_batch_size": 512,
        "direct_presampling_batch_size": 500,
        "dynamic_rejection_batch_size": 192,
    },
    "high": {
        "candidate_batch_size": 192,
        "distribution_batch_size": 512,
        "direct_presampling_batch_size": 500,
        "dynamic_rejection_batch_size": 256,
    },
    "ultra": {
        "candidate_batch_size": 448,
        "distribution_batch_size": 1024,
        "direct_presampling_batch_size": 500,
        "dynamic_rejection_batch_size": 256,
    },
}

VALID_PROFILES = tuple(_PROFILES)

# Nominal VRAM (MiB) of the GPU class each profile targets.  Used both to
# document the intent and to warn when a profile is run on a GPU too small for
# it -- see check_profile_fits_gpu().
PROFILE_TARGET_VRAM_MIB: dict[str, int] = {
    "low": 8 * 1024,
    "medium": 12 * 1024,
    "high": 16 * 1024,
    "ultra": 32 * 1024,
}

# Measured peak VRAM of the FULL static pipeline (MiB), from real end-to-end
# runs on RTX 5090 at DOF=7 / 460 poses -- not a synthetic micro-benchmark.
# These are torch.cuda.max_memory_allocated values; add ~500 MiB of CUDA
# context to get the process footprint a GPU actually has to supply.
#
# These are the UNPRESSURED peaks, reproducible to the MiB across runs (the
# 'high' figure was identical in three). On a GPU near capacity the allocator
# starts free-and-retry cycles and the high-water mark can climb ~20% above
# these values -- 'high' was seen at 13188 MiB when capped to a 16 GB budget,
# though the run still completed. Absorbing that spike is what the 15% headroom
# in VRAM_BUDGET_FRACTION is for; do not spend it on larger batch sizes.
PROFILE_EXPECTED_PEAK_MIB: dict[str, float] = {
    "low": 5494,
    "medium": 9077,
    "high": 10816,
    "ultra": 25005,
}

# Roughly what the CUDA context costs outside the caching allocator.
CUDA_CONTEXT_MIB = 500

# Fraction of the GPU a profile is allowed to occupy, leaving the rest for the
# OS/display and the cuRobo planner.
VRAM_BUDGET_FRACTION = 0.85


def profile_budget_mib(profile_name: str) -> float:
    """Return the peak-VRAM budget (MiB) a profile must stay within."""
    return VRAM_BUDGET_FRACTION * PROFILE_TARGET_VRAM_MIB[profile_name]


def check_profile_fits_gpu(profile_name: str) -> str | None:
    """Warn if ``profile_name`` cannot fit the GPU actually present.

    Picking the profile named after your GPU is the documented workflow, so the
    failure mode worth catching is picking one sized for a LARGER card: the run
    then OOMs minutes in, after presampling. Compare the profile's measured peak
    against this machine's real VRAM and return a warning string (or None).

    Returns None when no CUDA device is visible -- a CPU-only or
    driver-less environment is not something to warn about here.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        total_mib = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    except Exception:
        return None

    needed = PROFILE_EXPECTED_PEAK_MIB[profile_name] + CUDA_CONTEXT_MIB
    budget = VRAM_BUDGET_FRACTION * total_mib
    if needed <= budget:
        return None

    fitting = [
        name
        for name in VALID_PROFILES
        if PROFILE_EXPECTED_PEAK_MIB[name] + CUDA_CONTEXT_MIB
        <= VRAM_BUDGET_FRACTION * total_mib
    ]
    suggestion = (
        f" Consider vram_profile={fitting[-1]!r}."
        if fitting
        else " Even the 'low' profile does not fit; lower candidate_batch_size "
        "and distribution_batch_size explicitly."
    )
    return (
        f"[Warning] vram_profile={profile_name!r} needs about {needed:.0f} MiB "
        f"but this GPU has {total_mib:.0f} MiB total "
        f"({budget:.0f} MiB at {VRAM_BUDGET_FRACTION:.0%}). The run may OOM."
        + suggestion
    )


def resolve_vram_profile(data: dict) -> dict:
    """Apply VRAM profile defaults to a raw config dict.

    Reads ``data["vram_profile"]``, fills in any batch-size keys that are NOT
    already present (explicit config values always override the profile), then
    removes the ``vram_profile`` key so it does not pollute the
    ``argparse.Namespace`` seen by the rest of the pipeline.

    If ``"vram_profile"`` is absent the dict is returned unchanged.
    """
    profile_name = data.pop("vram_profile", None)
    if profile_name is None:
        return data

    if profile_name not in _PROFILES:
        raise ValueError(
            f"Unknown vram_profile {profile_name!r}. "
            f"Valid options: {', '.join(VALID_PROFILES)}"
        )

    for key, value in _PROFILES[profile_name].items():
        data.setdefault(key, value)

    # Only meaningful when the profile's own batch sizes are in play; an
    # explicit candidate_batch_size override changes the peak entirely.
    if data.get("candidate_batch_size") == _PROFILES[profile_name].get(
        "candidate_batch_size"
    ):
        warning = check_profile_fits_gpu(profile_name)
        if warning:
            print(warning)

    return data
