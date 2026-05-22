# -----------------------------------------------------------------------------
# Fixed-alpha morphology candidate sampler.
#
# This helper reuses task.morphology_sampler's original sampling and rejection
# logic, but conditions sampled morphologies on externally supplied alpha
# candidates.  It avoids the bad pattern of taking one initial morphology's
# [a, d] lengths and blindly swapping in many different alpha combinations.
# -----------------------------------------------------------------------------

from __future__ import annotations

import math

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
)


DEFAULT_FIXED_ALPHA_BATCH_SIZE = 512
DEFAULT_FIXED_ALPHA_RANDOM_TRIES = 5000
DEFAULT_DYNAMIC_REJECTION_BATCH_SIZE = 128

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


# ------------------------ fixed-alpha morphology sampler ----------------------


def _set_sampler_seed(seed: int | None, device: torch.device) -> None:
    """Set torch RNGs in the same style as morphology_sampler.py."""
    if seed is None:
        return
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


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
    if dynamic_rejection_batch_size <= 0:
        raise ValueError("dynamic_rejection_batch_size must be positive.")

    device = alpha_candidates.device
    _set_sampler_seed(seed, device)

    chunks: list[Tensor] = []
    total_failed = 0
    iterator = tqdm(
        range(0, alpha_candidates.shape[0], batch_size),
        desc="sampling fixed-alpha initial morphologies",
        disable=not logging,
        dynamic_ncols=True,
    )
    for start in iterator:
        end = min(start + batch_size, alpha_candidates.shape[0])
        sampled, failed_count = _sample_one_chunk_with_fixed_alpha(
            alpha_candidates[start:end].detach(),
            link_radius=link_radius,
            max_attempts_per_alpha=max_attempts_per_alpha,
            dynamic_rejection_batch_size=dynamic_rejection_batch_size,
        )
        if sampled.numel() > 0:
            chunks.append(sampled)
        total_failed += failed_count

        if logging:
            done = min(end, alpha_candidates.shape[0])
            kept = sum(chunk.shape[0] for chunk in chunks)
            iterator.set_postfix(
                sampled=kept,
                failed=total_failed,
                total=done,
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
            f"with at most {max_attempts_per_alpha} tries per alpha "
            f"(failed={total_failed})."
        )
        if total_failed > 0:
            print(
                "[Warning] Some alpha candidates failed all fixed-alpha sampler "
                "attempts and were dropped."
            )

    return result
