# -----------------------------------------------------------------------------
# Fixed-alpha morphology candidate sampler.
#
# This helper reuses task.morphology_sampler's original sampling and rejection
# logic, but conditions sampled morphologies on externally supplied alpha
# candidates.  It avoids the bad pattern of taking one initial morphology's
# [a, d] lengths and blindly swapping in many different alpha combinations.
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import Tensor
from tqdm import tqdm

from task.morphology_sampler import (
    LINK_RADIUS,
    _sample_link_type,
    _reject_link_type,
    _reject_link_twist,
    _sample_link_length,
    _reject_link_length,
    _reject_morph,
    sample_morph,
)


DEFAULT_FIXED_ALPHA_BATCH_SIZE = 512
DEFAULT_FIXED_ALPHA_RANDOM_TRIES = 5000
# direct sampling requires high GPU memory... (1000 is the max for 5090 with 32GB)
DEFAULT_DIRECT_PRESAMPLING_BATCH_SIZE = 500
DEFAULT_DYNAMIC_REJECTION_BATCH_SIZE = 128
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_INITIAL_CANDIDATE_CACHE_ROOT = _PROJECT_ROOT / "initial_candidates"
_INITIAL_CANDIDATE_CACHE_FILENAME = "candidates.json"

# Alpha candidate generation constants.  ALPHA_VALUES index 1 is the actual
# zero-twist alpha value, so checking for zero-alpha means checking index == 1,
# not index == 0.
ALPHA_VALUES = (-math.pi / 2.0, 0.0, math.pi / 2.0)
ZERO_ALPHA_INDEX = 1
DEFAULT_ZERO_ALPHA_RUN_EXCLUSION_LENGTH = 3


# ------------------------------ alpha candidates -----------------------------


def _flat_ids_to_base3_digits(flat_ids: Tensor, seq_len: int) -> Tensor:
    """Convert flat ids in [0, 3^seq_len) to base-3 digits [N, seq_len].

    Each digit indexes ALPHA_VALUES:
        0 -> -pi/2
        1 -> 0
        2 -> +pi/2
    """
    digits = []
    x = flat_ids.clone()
    for _ in range(seq_len):
        digits.append(x % 3)
        x = x // 3
    return torch.stack(digits, dim=1)


def _has_forbidden_zero_run(
    alpha_indices: Tensor,
    forbidden_run_length: int = DEFAULT_ZERO_ALPHA_RUN_EXCLUSION_LENGTH,
) -> Tensor:
    """Return True for rows containing too many consecutive alpha=0 entries.

    alpha_indices stores indices into ALPHA_VALUES.  Therefore alpha=0 means
    alpha_indices == ZERO_ALPHA_INDEX, not alpha_indices == 0.

    With forbidden_run_length=3, sequences containing 000, 0000, ... are
    rejected.  This matches task.morphology_sampler._reject_link_twist(), which
    rejects three consecutive zero twists.
    """
    if alpha_indices.ndim != 2:
        raise ValueError(
            f"Expected alpha_indices shape [N, seq_len], got {tuple(alpha_indices.shape)}."
        )
    if forbidden_run_length <= 1:
        return (alpha_indices == ZERO_ALPHA_INDEX).any(dim=1)

    run = torch.zeros(
        alpha_indices.shape[0],
        dtype=torch.long,
        device=alpha_indices.device,
    )
    bad = torch.zeros(
        alpha_indices.shape[0],
        dtype=torch.bool,
        device=alpha_indices.device,
    )

    for j in range(alpha_indices.shape[1]):
        is_zero = alpha_indices[:, j] == ZERO_ALPHA_INDEX
        run = torch.where(is_zero, run + 1, torch.zeros_like(run))
        bad |= run >= forbidden_run_length

    return bad


def _valid_alpha_flat_ids(
    seq_len: int,
    device: torch.device,
    forbidden_zero_run_length: int = DEFAULT_ZERO_ALPHA_RUN_EXCLUSION_LENGTH,
) -> Tensor:
    """All alpha ids excluding candidates with forbidden zero-alpha runs."""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive.")
    total = 3**seq_len
    flat_ids = torch.arange(total, device=device)
    alpha_indices = _flat_ids_to_base3_digits(flat_ids, seq_len)
    valid = ~_has_forbidden_zero_run(
        alpha_indices,
        forbidden_run_length=forbidden_zero_run_length,
    )
    return flat_ids[valid]


def _resolve_num_alpha_candidates(
    requested: int | str | None,
    max_valid_alpha_candidates: int,
) -> tuple[int, bool]:
    """Resolve 'ALL' or numeric candidate count.

    Returns:
        (num_candidates, use_all)
    """
    if requested is None:
        requested = "ALL"

    if isinstance(requested, str):
        if requested.upper() != "ALL":
            raise ValueError(
                "num_alpha_candidates must be 'ALL' or a positive integer, "
                f"got {requested!r}."
            )
        return max_valid_alpha_candidates, True

    requested_int = int(requested)
    if requested_int <= 0:
        raise ValueError("num_alpha_candidates must be positive or 'ALL'.")

    if requested_int >= max_valid_alpha_candidates:
        return max_valid_alpha_candidates, True

    return requested_int, False


def generate_alpha_candidates(
    requested_num_candidates: int | str | None,
    *,
    seq_len: int,
    device: torch.device,
    generator: torch.Generator,
    forbidden_zero_run_length: int = DEFAULT_ZERO_ALPHA_RUN_EXCLUSION_LENGTH,
) -> tuple[Tensor, int, bool]:
    """Generate alpha candidates after excluding forbidden zero-alpha runs.

    Args:
        requested_num_candidates:
            "ALL" to use every valid alpha combination, or a positive integer
            to randomly choose that many.  If the integer is larger than the
            valid maximum, all valid combinations are used.
        seq_len:
            Number of MDH rows, normally dof + 1.  For DOF=6, seq_len=7.
        device:
            Output device.
        generator:
            Torch generator used only when a random subset is requested.
        forbidden_zero_run_length:
            Reject alpha sequences containing this many or more consecutive
            alpha=0 entries.  Use 3 to match the original sampler distribution.

    Returns:
        alpha_candidates:
            Tensor [N, seq_len, 1] with values in {-pi/2, 0, pi/2}.
        max_valid:
            Number of valid alpha combinations after the run-length filter.
        use_all:
            Whether the returned tensor contains all valid alpha combinations.
    """
    if forbidden_zero_run_length <= 0:
        raise ValueError("forbidden_zero_run_length must be positive.")

    valid_flat_ids = _valid_alpha_flat_ids(
        seq_len=seq_len,
        device=device,
        forbidden_zero_run_length=forbidden_zero_run_length,
    )
    max_valid = int(valid_flat_ids.numel())
    num_candidates, use_all = _resolve_num_alpha_candidates(
        requested_num_candidates,
        max_valid_alpha_candidates=max_valid,
    )

    if use_all:
        chosen_ids = valid_flat_ids
    else:
        perm = torch.randperm(max_valid, generator=generator, device=device)
        chosen_ids = valid_flat_ids[perm[:num_candidates]]

    alpha_indices = _flat_ids_to_base3_digits(chosen_ids, seq_len)
    alpha_values = torch.tensor(ALPHA_VALUES, device=device, dtype=torch.float32)
    alpha_candidates = alpha_values[alpha_indices].unsqueeze(-1)

    return alpha_candidates, max_valid, use_all


def _alpha_values_to_indices(alpha_values: Tensor) -> Tensor:
    """Map alpha values in {-pi/2, 0, pi/2} to ALPHA_VALUES indices.

    We use nearest-level mapping instead of exact equality so small dtype or
    device differences do not break matching.  Returned indices use:
        0 -> -pi/2
        1 -> 0
        2 -> +pi/2
    """
    if alpha_values.ndim != 2:
        raise ValueError(
            f"Expected alpha_values shape [N, seq_len], got {tuple(alpha_values.shape)}."
        )

    levels = alpha_values.new_tensor(ALPHA_VALUES)
    distances = (alpha_values.unsqueeze(-1) - levels.view(1, 1, 3)).abs()
    return distances.argmin(dim=-1)


def _direct_presample_matching_alpha_candidates(
    alpha_candidates: Tensor,
    *,
    num_presamples: int,
    seed_already_set: bool = True,
    logging: bool = True,
) -> tuple[Tensor, Tensor]:
    """Warm-start fixed-alpha sampling using the original morphology sampler.

    This is only a one-shot shortcut:
      1. draw num_presamples valid morphologies from the original sampler;
      2. if a sampled morphology's alpha sequence matches an alpha candidate,
         keep the first such morphology;
      3. repeated alpha sequences are ignored;
      4. unmatched alpha candidates are returned for later fixed-alpha sampling.

    Args:
        alpha_candidates: Tensor [N, seq_len, 1].
        num_presamples: Number of original-sampler morphologies to draw once.
        seed_already_set: Documentation flag only; the caller sets torch RNGs.
        logging: Whether to print a summary.

    Returns:
        presampled_result:
            Tensor [M, seq_len, 3] containing matched morphologies, ordered by
            the original alpha_candidates order.
        remaining_mask:
            Bool Tensor [N], True for alpha candidates still needing fixed-alpha
            rejection sampling.
    """
    del seed_already_set  # RNG state is intentionally controlled by the caller.

    if num_presamples <= 0:
        remaining_mask = torch.ones(
            alpha_candidates.shape[0], dtype=torch.bool, device=alpha_candidates.device
        )
        empty = torch.empty(
            0,
            alpha_candidates.shape[1],
            3,
            device=alpha_candidates.device,
            dtype=alpha_candidates.dtype,
        )
        return empty, remaining_mask

    device = alpha_candidates.device
    dtype = alpha_candidates.dtype
    num_candidates, seq_len, _ = alpha_candidates.shape
    dof = seq_len - 1

    candidate_indices = _alpha_values_to_indices(alpha_candidates.squeeze(-1))

    # Alpha candidates should already be unique, but keeping the first index makes
    # this robust to accidental duplicates.
    candidate_lookup: dict[tuple[int, ...], int] = {}
    for i, row in enumerate(candidate_indices.detach().cpu().tolist()):
        candidate_lookup.setdefault(tuple(int(x) for x in row), i)

    direct_morphs = sample_morph(
        num_robots=num_presamples,
        dof=dof,
        analytically_solvable=False,
        device=device,
    ).to(device=device, dtype=dtype)

    direct_alpha_indices = _alpha_values_to_indices(direct_morphs[..., 0])

    result = torch.empty(num_candidates, seq_len, 3, device=device, dtype=dtype)
    matched = torch.zeros(num_candidates, dtype=torch.bool, device=device)

    for sample_idx, row in enumerate(direct_alpha_indices.detach().cpu().tolist()):
        candidate_idx = candidate_lookup.get(tuple(int(x) for x in row))
        if candidate_idx is None:
            continue
        if bool(matched[candidate_idx]):
            continue
        result[candidate_idx] = direct_morphs[sample_idx]
        matched[candidate_idx] = True

    if logging:
        print(
            "[Info] Direct sampler presampling: "
            f"matched {int(matched.sum().item())}/{num_candidates} alpha candidates "
            f"from {num_presamples} original sampler draws."
        )

    return result[matched], ~matched


# ------------------------ fixed-alpha morphology sampler ----------------------


def _set_sampler_seed(seed: int | None, device: torch.device) -> None:
    """Set torch RNGs in the same style as morphology_sampler.py."""
    if seed is None:
        return
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _initial_candidate_cache_path(alpha_candidates: Tensor, seed: int | None) -> Path:
    dof = alpha_candidates.shape[1] - 1
    seed_label = "None" if seed is None else str(int(seed))
    return (
        _INITIAL_CANDIDATE_CACHE_ROOT
        / f"DOF{dof}_seed{seed_label}"
        / _INITIAL_CANDIDATE_CACHE_FILENAME
    )


def _dof_cache_label(dofs: list[int]) -> str:
    sorted_dofs = sorted(int(dof) for dof in dofs)
    if len(sorted_dofs) > 1 and sorted_dofs == list(
        range(sorted_dofs[0], sorted_dofs[-1] + 1)
    ):
        return f"{sorted_dofs[0]}-{sorted_dofs[-1]}"
    return "_".join(str(dof) for dof in sorted_dofs)


def _initial_candidate_cache_path_for_dofs(
    alpha_candidates_by_dof: dict[int, Tensor],
    seed: int | None,
) -> Path:
    seed_label = "None" if seed is None else str(int(seed))
    return (
        _INITIAL_CANDIDATE_CACHE_ROOT
        / f"DOF{_dof_cache_label(list(alpha_candidates_by_dof))}_seed{seed_label}"
        / _INITIAL_CANDIDATE_CACHE_FILENAME
    )


def _alpha_signature(alpha_candidates: Tensor) -> list[list[int]]:
    return [
        [int(v) for v in row]
        for row in _alpha_values_to_indices(
            alpha_candidates.detach().squeeze(-1).cpu()
        ).tolist()
    ]


def _load_cached_initial_candidates(
    alpha_candidates: Tensor,
    *,
    seed: int | None,
    link_radius: float,
    logging: bool,
) -> Tensor | None:
    """Load cached initial morphologies if they match this alpha set."""
    cache_path = _initial_candidate_cache_path(alpha_candidates, seed)
    if not cache_path.is_file():
        return None

    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        metadata = data["metadata"]
        morphologies = data["morphologies"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        if logging:
            print(
                f"[Warning] Could not read initial candidate cache {cache_path}: {exc}"
            )
        return None

    dof = alpha_candidates.shape[1] - 1
    expected_alpha_signature = _alpha_signature(alpha_candidates)
    cache_is_compatible = (
        metadata.get("dof") == dof
        and metadata.get("seq_len") == alpha_candidates.shape[1]
        and metadata.get("seed") == seed
        and abs(float(metadata.get("link_radius", float("nan"))) - float(link_radius))
        <= 1e-9
        and metadata.get("alpha_signature") == expected_alpha_signature
    )
    if not cache_is_compatible:
        if logging:
            print(
                "[Warning] Initial candidate cache metadata does not match current "
                f"alpha candidates; resampling. cache={cache_path}"
            )
        return None

    cached = torch.tensor(
        morphologies,
        dtype=alpha_candidates.dtype,
        device=alpha_candidates.device,
    )
    expected_shape_tail = (alpha_candidates.shape[1], 3)
    if cached.ndim != 3 or tuple(cached.shape[1:]) != expected_shape_tail:
        if logging:
            print(
                "[Warning] Initial candidate cache has unexpected shape "
                f"{tuple(cached.shape)}; resampling. cache={cache_path}"
            )
        return None

    if logging:
        print(
            "[Info] Loaded initial candidate morphologies from cache: "
            f"{cache_path} ({cached.shape[0]} candidates)."
        )

    return cached


def _save_initial_candidate_cache(
    alpha_candidates: Tensor,
    morphologies: Tensor,
    *,
    seed: int | None,
    link_radius: float,
    logging: bool,
) -> None:
    """Save initial morphologies for reuse with the same DOF/seed/alpha set."""
    cache_path = _initial_candidate_cache_path(alpha_candidates, seed)
    dof = alpha_candidates.shape[1] - 1
    payload = {
        "metadata": {
            "schema_version": 1,
            "dof": dof,
            "seq_len": alpha_candidates.shape[1],
            "seed": seed,
            "link_radius": float(link_radius),
            "num_alpha_candidates": int(alpha_candidates.shape[0]),
            "num_sampled_morphologies": int(morphologies.shape[0]),
            "alpha_signature": _alpha_signature(alpha_candidates),
        },
        "morphologies": morphologies.detach().cpu().tolist(),
    }

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        if logging:
            print(
                f"[Warning] Could not write initial candidate cache {cache_path}: {exc}"
            )
        return

    if logging:
        print(f"[Info] Saved initial candidate morphologies to cache: {cache_path}")


def _alpha_signatures_by_dof(
    alpha_candidates_by_dof: dict[int, Tensor],
) -> dict[str, list[list[int]]]:
    return {
        str(int(dof)): _alpha_signature(alpha_candidates)
        for dof, alpha_candidates in sorted(alpha_candidates_by_dof.items())
    }


def _load_cached_initial_candidates_by_dof(
    alpha_candidates_by_dof: dict[int, Tensor],
    *,
    seed: int | None,
    link_radius: float,
    logging: bool,
) -> dict[int, Tensor] | None:
    """Load a combined multi-DOF cache if it matches every alpha set."""
    cache_path = _initial_candidate_cache_path_for_dofs(alpha_candidates_by_dof, seed)
    if not cache_path.is_file():
        return None

    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        metadata = data["metadata"]
        morphologies_by_dof = data["morphologies_by_dof"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        if logging:
            print(
                f"[Warning] Could not read initial candidate cache {cache_path}: {exc}"
            )
        return None

    dofs = sorted(int(dof) for dof in alpha_candidates_by_dof)
    cache_is_compatible = (
        metadata.get("dofs") == dofs
        and metadata.get("seed") == seed
        and abs(float(metadata.get("link_radius", float("nan"))) - float(link_radius))
        <= 1e-9
        and metadata.get("alpha_signatures_by_dof")
        == _alpha_signatures_by_dof(alpha_candidates_by_dof)
    )
    if not cache_is_compatible:
        if logging:
            print(
                "[Warning] Multi-DOF initial candidate cache metadata does not "
                f"match current alpha candidates; resampling. cache={cache_path}"
            )
        return None

    cached_by_dof: dict[int, Tensor] = {}
    for dof in dofs:
        alpha_candidates = alpha_candidates_by_dof[dof]
        try:
            cached = torch.tensor(
                morphologies_by_dof[str(dof)],
                dtype=alpha_candidates.dtype,
                device=alpha_candidates.device,
            )
        except KeyError:
            if logging:
                print(
                    "[Warning] Multi-DOF initial candidate cache is missing "
                    f"DOF{dof}; resampling. cache={cache_path}"
                )
            return None

        expected_shape_tail = (dof + 1, 3)
        if cached.ndim != 3 or tuple(cached.shape[1:]) != expected_shape_tail:
            if logging:
                print(
                    "[Warning] Multi-DOF initial candidate cache has unexpected "
                    f"shape for DOF{dof}: {tuple(cached.shape)}; resampling. "
                    f"cache={cache_path}"
                )
            return None

        cached_by_dof[dof] = cached

    if logging:
        counts = ", ".join(f"DOF{dof}={cached_by_dof[dof].shape[0]}" for dof in dofs)
        print(
            "[Info] Loaded multi-DOF initial candidate morphologies from cache: "
            f"{cache_path} ({counts})."
        )

    return cached_by_dof


def _save_initial_candidate_cache_by_dof(
    alpha_candidates_by_dof: dict[int, Tensor],
    morphologies_by_dof: dict[int, Tensor],
    *,
    seed: int | None,
    link_radius: float,
    logging: bool,
) -> None:
    """Save one combined cache for a set of DOFs."""
    cache_path = _initial_candidate_cache_path_for_dofs(alpha_candidates_by_dof, seed)
    dofs = sorted(int(dof) for dof in alpha_candidates_by_dof)
    payload = {
        "metadata": {
            "schema_version": 1,
            "dofs": dofs,
            "seed": seed,
            "link_radius": float(link_radius),
            "num_alpha_candidates_by_dof": {
                str(dof): int(alpha_candidates_by_dof[dof].shape[0]) for dof in dofs
            },
            "num_sampled_morphologies_by_dof": {
                str(dof): int(morphologies_by_dof[dof].shape[0]) for dof in dofs
            },
            "alpha_signatures_by_dof": _alpha_signatures_by_dof(
                alpha_candidates_by_dof
            ),
        },
        "morphologies_by_dof": {
            str(dof): morphologies_by_dof[dof].detach().cpu().tolist() for dof in dofs
        },
    }

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        if logging:
            print(
                f"[Warning] Could not write initial candidate cache {cache_path}: {exc}"
            )
        return

    if logging:
        print(
            "[Info] Saved multi-DOF initial candidate morphologies to cache: "
            f"{cache_path}"
        )


def _dynamic_reject_morph_batched(
    morph: Tensor,
    *,
    batch_size: int,
) -> Tensor:
    """Run morphology_sampler._reject_morph in smaller batches.

    _reject_morph internally evaluates 1000 random joint configurations per
    morphology, so batching avoids creating a very large temporary tensor.
    """
    if morph.numel() == 0:
        return torch.empty(0, dtype=torch.bool, device=morph.device)

    rejected_chunks: list[Tensor] = []
    for start in range(0, morph.shape[0], batch_size):
        end = min(start + batch_size, morph.shape[0])
        rejected_chunks.append(_reject_morph(morph[start:end]))
    return torch.cat(rejected_chunks, dim=0)


def _sample_one_attempt_for_unresolved_alpha(
    current_alpha: Tensor,
    unresolved_global_indices: Tensor,
    *,
    result: Tensor,
    resolved: Tensor,
    dynamic_rejection_batch_size: int,
) -> int:
    """Run one random link-type/length attempt for each unresolved alpha.

    This is the core intended semantics:
        fixed alpha + one random sampler-style [link_type, a, d] attempt.

    If an alpha succeeds in this attempt, it is immediately written to result
    and will not be sampled again in later attempts.

    Returns:
        Number of newly resolved alpha candidates.
    """
    device = current_alpha.device
    dtype = current_alpha.dtype
    current_count, seq_len = current_alpha.shape
    dof = seq_len - 1

    # Same style as the original sampler: sample link type first, then sample
    # [a, d] according to that type.  Alpha is fixed externally.
    link_type = _sample_link_type(current_count, dof, device)

    structural_rejected = _reject_link_type(link_type) | _reject_link_twist(
        current_alpha,
        link_type,
    )
    structural_ok = ~structural_rejected
    if not structural_ok.any():
        return 0

    ok_alpha = current_alpha[structural_ok]
    ok_link_type = link_type[structural_ok]
    ok_global_indices = unresolved_global_indices[structural_ok]

    link_lengths = _sample_link_length(ok_link_type).to(dtype=dtype)
    length_ok = ~_reject_link_length(link_lengths)
    if not length_ok.any():
        return 0

    morph_alpha = ok_alpha[length_ok]
    morph_lengths = link_lengths[length_ok]
    morph_global_indices = ok_global_indices[length_ok]
    morph = torch.cat([morph_alpha.unsqueeze(-1), morph_lengths], dim=-1)

    dynamic_rejected = _dynamic_reject_morph_batched(
        morph,
        batch_size=dynamic_rejection_batch_size,
    )
    accepted = ~dynamic_rejected
    if not accepted.any():
        return 0

    accepted_global_indices = morph_global_indices[accepted]
    result[accepted_global_indices] = morph[accepted]
    resolved[accepted_global_indices] = True
    return int(accepted.sum().item())


def _sample_one_chunk_with_fixed_alpha(
    alpha_chunk: Tensor,
    *,
    link_radius: float,
    max_attempts_per_alpha: int,
    dynamic_rejection_batch_size: int,
) -> tuple[Tensor, int]:
    """Sample valid morphologies for one chunk of fixed alpha candidates.

    For each alpha candidate:
        - try at most max_attempts_per_alpha random sampler-style attempts;
        - as soon as one attempt passes all original sampler checks, keep it;
        - if none pass, drop this alpha candidate.

    Attempts are parallelized over different unresolved alpha candidates, not
    over the 100 attempts of one alpha.  This preserves early-return behavior:
    successful alpha candidates stop being sampled immediately.
    """
    if abs(float(link_radius) - float(LINK_RADIUS)) > 1e-9:
        raise ValueError(
            "sample_fixed_alpha_morphology_candidates currently requires "
            f"link_radius={LINK_RADIUS}, got {link_radius}."
        )

    if alpha_chunk.ndim != 3 or alpha_chunk.shape[-1] != 1:
        raise ValueError(
            f"Expected alpha_chunk shape [B, dof+1, 1], got {tuple(alpha_chunk.shape)}."
        )
    if max_attempts_per_alpha <= 0:
        raise ValueError("max_attempts_per_alpha must be positive.")
    if dynamic_rejection_batch_size <= 0:
        raise ValueError("dynamic_rejection_batch_size must be positive.")

    device = alpha_chunk.device
    dtype = alpha_chunk.dtype
    batch_size, seq_len, _ = alpha_chunk.shape

    alpha_2d = alpha_chunk.squeeze(-1)
    result = torch.empty(batch_size, seq_len, 3, device=device, dtype=dtype)
    resolved = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _attempt_idx in range(max_attempts_per_alpha):
        unresolved = torch.nonzero(~resolved, as_tuple=False).squeeze(1)
        if unresolved.numel() == 0:
            break

        current_alpha = alpha_2d[unresolved]
        _sample_one_attempt_for_unresolved_alpha(
            current_alpha,
            unresolved,
            result=result,
            resolved=resolved,
            dynamic_rejection_batch_size=dynamic_rejection_batch_size,
        )

    failed_count = int((~resolved).sum().item())
    return result[resolved], failed_count


@torch.inference_mode()
def sample_fixed_alpha_morphology_candidates(
    alpha_candidates: Tensor,
    *,
    seed: int | None = 0,
    link_radius: float = LINK_RADIUS,
    batch_size: int = DEFAULT_FIXED_ALPHA_BATCH_SIZE,
    max_attempts_per_alpha: int = DEFAULT_FIXED_ALPHA_RANDOM_TRIES,
    direct_presampling_batch_size: int = DEFAULT_DIRECT_PRESAMPLING_BATCH_SIZE,
    dynamic_rejection_batch_size: int = DEFAULT_DYNAMIC_REJECTION_BATCH_SIZE,
    logging: bool = True,
) -> Tensor:
    """Sample at most one valid morphology for each fixed alpha candidate.

    This function keeps the supplied alpha values exactly, then samples the
    remaining sampler variables (link_type and [a, d] lengths) using the
    original sampler distribution.  Each alpha candidate gets at most
    max_attempts_per_alpha random attempts.  The first successful attempt is
    kept immediately; if all attempts fail, that alpha candidate is dropped.

    Args:
        alpha_candidates:
            Tensor [N, dof+1, 1], fixed alpha candidates.
        seed:
            Seed for torch's RNG, matching the style of morphology_sampler.py.
        link_radius:
            Link radius.  Currently must equal morphology_sampler.LINK_RADIUS.
        batch_size:
            Number of fixed-alpha candidates processed per chunk.
        max_attempts_per_alpha:
            Hard-coded default is 100.  An alpha is dropped if none of these
            attempts produces a valid morphology.
        direct_presampling_batch_size:
            One-shot warm-start count.  First draw this many valid morphologies
            from the original sampler and use them to immediately satisfy
            matching alpha candidates.  Remaining candidates then use the
            fixed-alpha sampler.  Set to 0 to disable.
        dynamic_rejection_batch_size:
            Batch size used for the expensive dynamic _reject_morph call.
        logging:
            If True, show a tqdm progress bar and summary.

    Returns:
        Tensor [M, dof+1, 3] with valid sampled morphologies, where M <= N if
        some alpha candidates failed all attempts.
    """
    if alpha_candidates.ndim != 3 or alpha_candidates.shape[-1] != 1:
        raise ValueError(
            f"Expected alpha_candidates shape [N, dof+1, 1], "
            f"got {tuple(alpha_candidates.shape)}."
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if max_attempts_per_alpha <= 0:
        raise ValueError("max_attempts_per_alpha must be positive.")
    if direct_presampling_batch_size < 0:
        raise ValueError("direct_presampling_batch_size must be non-negative.")
    if dynamic_rejection_batch_size <= 0:
        raise ValueError("dynamic_rejection_batch_size must be positive.")

    device = alpha_candidates.device
    cached = _load_cached_initial_candidates(
        alpha_candidates,
        seed=seed,
        link_radius=link_radius,
        logging=logging,
    )
    if cached is not None:
        return cached

    _set_sampler_seed(seed, device)

    presampled, remaining_mask = _direct_presample_matching_alpha_candidates(
        alpha_candidates.detach(),
        num_presamples=direct_presampling_batch_size,
        logging=logging,
    )

    remaining_alpha = alpha_candidates[remaining_mask].detach()
    chunks: list[Tensor] = []
    if presampled.numel() > 0:
        chunks.append(presampled)

    total_failed = 0
    if remaining_alpha.shape[0] > 0:
        iterator = tqdm(
            range(0, remaining_alpha.shape[0], batch_size),
            desc="sampling remaining fixed-alpha morphologies",
            disable=not logging,
            dynamic_ncols=True,
        )
        for start in iterator:
            end = min(start + batch_size, remaining_alpha.shape[0])
            sampled, failed_count = _sample_one_chunk_with_fixed_alpha(
                remaining_alpha[start:end],
                link_radius=link_radius,
                max_attempts_per_alpha=max_attempts_per_alpha,
                dynamic_rejection_batch_size=dynamic_rejection_batch_size,
            )
            if sampled.numel() > 0:
                chunks.append(sampled)
            total_failed += failed_count

            if logging:
                done_remaining = min(end, remaining_alpha.shape[0])
                kept_fixed = (
                    sum(chunk.shape[0] for chunk in chunks) - presampled.shape[0]
                )
                iterator.set_postfix(
                    presampled=presampled.shape[0],
                    fixed_sampled=kept_fixed,
                    failed=total_failed,
                    remaining_done=done_remaining,
                    tries=max_attempts_per_alpha,
                )

    if not chunks:
        raise RuntimeError(
            "Fixed-alpha sampler failed to generate any valid initial morphology."
        )

    result = torch.cat(chunks, dim=0)
    if logging:
        print(
            "[Info] Fixed-alpha initial sampling: "
            f"sampled {result.shape[0]}/{alpha_candidates.shape[0]} candidates "
            f"(direct_presampled={presampled.shape[0]}, "
            f"fixed_alpha_sampled={result.shape[0] - presampled.shape[0]}, "
            f"failed={total_failed}, "
            f"direct_presampling_batch_size={direct_presampling_batch_size}, "
            f"max_fixed_attempts_per_alpha={max_attempts_per_alpha})."
        )
        if total_failed > 0:
            print(
                "[Warning] Some alpha candidates failed all fixed-alpha sampler "
                "attempts and were dropped."
            )

    _save_initial_candidate_cache(
        alpha_candidates,
        result,
        seed=seed,
        link_radius=link_radius,
        logging=logging,
    )

    return result


@torch.inference_mode()
def sample_fixed_alpha_morphology_candidates_by_dof(
    alpha_candidates_by_dof: dict[int, Tensor],
    *,
    seed: int | None = 0,
    link_radius: float = LINK_RADIUS,
    batch_size: int = DEFAULT_FIXED_ALPHA_BATCH_SIZE,
    max_attempts_per_alpha: int = DEFAULT_FIXED_ALPHA_RANDOM_TRIES,
    direct_presampling_batch_size: int = DEFAULT_DIRECT_PRESAMPLING_BATCH_SIZE,
    dynamic_rejection_batch_size: int = DEFAULT_DYNAMIC_REJECTION_BATCH_SIZE,
    logging: bool = True,
) -> dict[int, Tensor]:
    """Sample fixed-alpha initial morphologies for several DOFs with one cache.

    The single-DOF sampler and its cache remain unchanged. This wrapper adds a
    combined cache such as initial_candidates/DOF5-7_seed0/candidates.json and
    otherwise runs the same fixed-alpha sampling process independently per DOF.

    Returns:
        Mapping from dof to Tensor [M, dof+1, 3].
    """
    if not alpha_candidates_by_dof:
        raise ValueError("alpha_candidates_by_dof must not be empty.")

    normalized: dict[int, Tensor] = {}
    for dof, alpha_candidates in sorted(alpha_candidates_by_dof.items()):
        dof = int(dof)
        if alpha_candidates.ndim != 3 or alpha_candidates.shape[-1] != 1:
            raise ValueError(
                f"Expected alpha_candidates for DOF{dof} to have shape "
                f"[N, dof+1, 1], got {tuple(alpha_candidates.shape)}."
            )
        actual_dof = alpha_candidates.shape[1] - 1
        if actual_dof != dof:
            raise ValueError(
                f"DOF key {dof} does not match alpha candidate shape "
                f"{tuple(alpha_candidates.shape)}."
            )
        normalized[dof] = alpha_candidates

    cached = _load_cached_initial_candidates_by_dof(
        normalized,
        seed=seed,
        link_radius=link_radius,
        logging=logging,
    )
    if cached is not None:
        return cached

    sampled_by_dof: dict[int, Tensor] = {}
    for dof, alpha_candidates in normalized.items():
        if logging:
            print(
                f"[Info] Sampling initial candidate morphologies for DOF{dof} "
                f"({alpha_candidates.shape[0]} alpha candidates)."
            )
        sampled_by_dof[dof] = sample_fixed_alpha_morphology_candidates(
            alpha_candidates=alpha_candidates,
            seed=seed,
            link_radius=link_radius,
            batch_size=batch_size,
            max_attempts_per_alpha=max_attempts_per_alpha,
            direct_presampling_batch_size=direct_presampling_batch_size,
            dynamic_rejection_batch_size=dynamic_rejection_batch_size,
            logging=logging,
        )

    _save_initial_candidate_cache_by_dof(
        normalized,
        sampled_by_dof,
        seed=seed,
        link_radius=link_radius,
        logging=logging,
    )

    return sampled_by_dof
