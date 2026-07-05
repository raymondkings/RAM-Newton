# -----------------------------------------------------------------------------
# Distribution checker for optimized morphologies.
#
# This checker assumes the morphology is FINAL / processed:
# alpha should already be exactly one of {-pi/2, 0, pi/2}.
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from tasks.sampling.morphology_sampler import (
    LINK_RADIUS,
    _reject_link_type,
    _reject_link_twist,
    _reject_link_length,
    get_joint_limits,
    forward_kinematics,
    collision_check,
    geometric_jacobian,
    yoshikawa_manipulability,
)


@dataclass
class DistributionCheckReport:
    valid: bool
    reasons: list[str]

    alpha_exactly_discrete: bool
    alpha_degrees: list[float]

    link_type: list[int]
    link_type_rejected: bool
    link_twist_rejected: bool
    link_length_rejected: bool

    sampled_always_self_colliding: bool
    sampled_self_collision_rate: float
    sampled_always_low_manipulability: bool
    min_yoshikawa: float
    mean_yoshikawa: float

    num_joint_samples: int


def _as_morph_tensor(morphology: Any) -> Tensor:
    """Accept Morphology or raw Tensor and return [B, dofp1, 3]."""
    tensor = morphology.params if hasattr(morphology, "params") else morphology

    if not isinstance(tensor, Tensor):
        tensor = torch.tensor(tensor, dtype=torch.float32)

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)

    if tensor.ndim != 3 or tensor.shape[-1] != 3:
        raise ValueError(
            f"Expected morphology shape [dof+1, 3] or [B, dof+1, 3], "
            f"got {tuple(tensor.shape)}"
        )

    return tensor.float()


def _alpha_levels(device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.tensor(
        [-torch.pi / 2.0, 0.0, torch.pi / 2.0],
        device=device,
        dtype=dtype,
    )


def _alpha_exactly_discrete(alpha: Tensor) -> Tensor:
    """Exact membership in {-pi/2, 0, pi/2}; no tolerance."""
    levels = _alpha_levels(alpha.device, alpha.dtype)
    per_entry_ok = (alpha.unsqueeze(-1) == levels).any(dim=-1)
    return per_entry_ok.all(dim=-1)


def _infer_link_type_from_lengths(morph: Tensor) -> Tensor:
    """Infer sampler link_type from exact zero/nonzero pattern of a,d.

    0: a != 0, d != 0
    1: a == 0, d != 0
    2: a != 0, d == 0
    3: a == 0, d == 0
    """
    a = morph[..., 1]
    d = morph[..., 2]

    a_zero = a == 0.0
    d_zero = d == 0.0

    link_type = torch.zeros_like(a, dtype=torch.long)
    link_type[a_zero & ~d_zero] = 1
    link_type[~a_zero & d_zero] = 2
    link_type[a_zero & d_zero] = 3
    return link_type


@torch.no_grad()
def _run_dynamic_sampler_checks(
    morph: Tensor,
    num_joint_samples: int,
    seed: int | None,
    radius: float,
    yoshikawa_threshold: float,
) -> dict[str, Tensor]:
    """Replicate the dynamic part of sampler._reject_morph with diagnostics."""
    device = morph.device
    dtype = morph.dtype
    batch_size = morph.shape[0]

    joint_limits = get_joint_limits(morph)
    joint_limits = joint_limits.unsqueeze(1).expand(
        batch_size,
        num_joint_samples,
        -1,
        -1,
    )
    morph_expanded = morph.unsqueeze(1).expand(
        batch_size,
        num_joint_samples,
        -1,
        -1,
    )

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    joints = torch.rand(
        *joint_limits.shape[:-1],
        1,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    joints = joints * joint_limits[..., 0:1] + joint_limits[..., 1:2]

    poses = forward_kinematics(morph_expanded, joints)

    sampled_self_collision = collision_check(
        morph_expanded,
        poses,
        radius=radius,
    )
    sampled_always_self_colliding = sampled_self_collision.all(dim=1)
    sampled_self_collision_rate = sampled_self_collision.float().mean(dim=1)

    jacobian = geometric_jacobian(poses)
    yoshikawa = yoshikawa_manipulability(jacobian, soft=True)
    low_manip = yoshikawa < yoshikawa_threshold
    sampled_always_low_manipulability = low_manip.all(dim=1)

    return {
        "sampled_always_self_colliding": sampled_always_self_colliding,
        "sampled_self_collision_rate": sampled_self_collision_rate,
        "sampled_always_low_manipulability": sampled_always_low_manipulability,
        "min_yoshikawa": yoshikawa.amin(dim=1),
        "mean_yoshikawa": yoshikawa.mean(dim=1),
    }


@torch.no_grad()
def check_morphology_distribution(
    morphology: Any,
    *,
    num_joint_samples: int = 1000,
    seed: int | None = 0,
    radius: float = LINK_RADIUS,
    yoshikawa_threshold: float = 1e-4,
) -> list[DistributionCheckReport]:
    """Check whether morphology is inside the sampler distribution.

    Args:
        morphology:
            Morphology object or Tensor with shape [dof+1, 3] / [B, dof+1, 3].
            This should be FINAL processed morphology.
        num_joint_samples:
            Number of random joint configs. Sampler uses 1000.
    """
    morph = _as_morph_tensor(morphology)

    alpha = morph[..., 0]
    alpha_ok = _alpha_exactly_discrete(alpha)

    link_type = _infer_link_type_from_lengths(morph)
    link_lengths = morph[..., 1:3]

    link_type_rejected = _reject_link_type(link_type)
    link_twist_rejected = _reject_link_twist(alpha, link_type)
    link_length_rejected = _reject_link_length(link_lengths)

    dynamic = _run_dynamic_sampler_checks(
        morph=morph,
        num_joint_samples=num_joint_samples,
        seed=seed,
        radius=radius,
        yoshikawa_threshold=yoshikawa_threshold,
    )

    reports: list[DistributionCheckReport] = []

    for i in range(morph.shape[0]):
        reasons: list[str] = []

        if not bool(alpha_ok[i]):
            reasons.append("alpha_not_exactly_in_{-pi/2,0,pi/2}")
        if bool(link_type_rejected[i]):
            reasons.append("link_type_rejected")
        if bool(link_twist_rejected[i]):
            reasons.append("link_twist_rejected")
        if bool(link_length_rejected[i]):
            reasons.append("link_length_rejected")
        if bool(dynamic["sampled_always_self_colliding"][i]):
            reasons.append("sampled_always_self_colliding")
        if bool(dynamic["sampled_always_low_manipulability"][i]):
            reasons.append("sampled_always_low_manipulability")

        reports.append(
            DistributionCheckReport(
                valid=len(reasons) == 0,
                reasons=reasons,
                alpha_exactly_discrete=bool(alpha_ok[i]),
                alpha_degrees=(alpha[i].detach().cpu() * 180.0 / torch.pi).tolist(),
                link_type=link_type[i].detach().cpu().tolist(),
                link_type_rejected=bool(link_type_rejected[i]),
                link_twist_rejected=bool(link_twist_rejected[i]),
                link_length_rejected=bool(link_length_rejected[i]),
                sampled_always_self_colliding=bool(
                    dynamic["sampled_always_self_colliding"][i]
                ),
                sampled_self_collision_rate=float(
                    dynamic["sampled_self_collision_rate"][i].detach().cpu().item()
                ),
                sampled_always_low_manipulability=bool(
                    dynamic["sampled_always_low_manipulability"][i]
                ),
                min_yoshikawa=float(dynamic["min_yoshikawa"][i].detach().cpu().item()),
                mean_yoshikawa=float(
                    dynamic["mean_yoshikawa"][i].detach().cpu().item()
                ),
                num_joint_samples=num_joint_samples,
            )
        )

    return reports


def check_single_morphology_distribution(
    morphology: Any,
    **kwargs,
) -> DistributionCheckReport:
    reports = check_morphology_distribution(morphology, **kwargs)

    if len(reports) != 1:
        raise ValueError(
            f"Expected exactly one morphology, got batch size {len(reports)}. "
            "Use check_morphology_distribution for batches."
        )

    return reports[0]


def format_distribution_report(report: DistributionCheckReport) -> str:
    status = "VALID" if report.valid else "REJECTED"

    lines = [
        f"Distribution check: {status}",
        f"  reasons: {report.reasons}",
        f"  alpha_exactly_discrete: {report.alpha_exactly_discrete}",
        f"  alpha_degrees: {report.alpha_degrees}",
        f"  inferred_link_type: {report.link_type}",
        f"  link_type_rejected: {report.link_type_rejected}",
        f"  link_twist_rejected: {report.link_twist_rejected}",
        f"  link_length_rejected: {report.link_length_rejected}",
        f"  sampled_always_self_colliding: {report.sampled_always_self_colliding}",
        f"  sampled_self_collision_rate: {report.sampled_self_collision_rate:.6f}",
        f"  sampled_always_low_manipulability: {report.sampled_always_low_manipulability}",
        f"  min_yoshikawa: {report.min_yoshikawa:.6e}",
        f"  mean_yoshikawa: {report.mean_yoshikawa:.6e}",
        f"  num_joint_samples: {report.num_joint_samples}",
    ]

    return "\n".join(lines)
