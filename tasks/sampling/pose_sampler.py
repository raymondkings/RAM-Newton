from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

NUM_SAMPLES = 50
NUM_LINE_SAMPLES = 50
NUM_EXTRA_PATHS = 4
BETA_RANGE_DEGREES = (-30.0, 30.0)
LINE_P_RANGE = (0.0, 1.0)
REPEAT_START_GOAL = 80

from paths import INITIAL_CANDIDATES_DIR as CACHE_DIR

# ALPHA_RANGE_DEGREES, START_POSE, _default_device, and _pose_from_alpha are the
# shared arch geometry, re-exported here for callers of this module.
from tasks.sampling._pose_common import (
    ALPHA_RANGE_DEGREES,
    START_POSE,
    _default_device,
    _pose_from_alpha,
)


def _jsonable_float_list(tensor: torch.Tensor) -> list:
    return tensor.detach().cpu().tolist()


def _collapse_consecutive_duplicate_poses(
    poses: torch.Tensor,
    *,
    atol: float = 1e-7,
) -> torch.Tensor:
    """Collapse adjacent duplicate SE(3) poses while preserving order."""
    if poses.shape[0] <= 1:
        return poses

    keep = [0]
    for i in range(1, poses.shape[0]):
        is_duplicate = torch.allclose(poses[i], poses[keep[-1]], atol=atol, rtol=0.0)
        if not is_duplicate:
            keep.append(i)

    indices = torch.tensor(keep, dtype=torch.long, device=poses.device)
    return poses[indices]


def _as_pose_tensor(pose: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    pose = pose.detach().cpu().to(dtype=dtype)
    if pose.shape != (4, 4):
        raise ValueError(f"start_pose must have shape (4, 4), got {tuple(pose.shape)}.")
    return pose


def _line_pose_from_p(p: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    p = p.to(dtype=dtype)
    n = p.numel()

    poses = torch.eye(4, dtype=dtype, device=p.device).repeat(n, 1, 1)
    poses[:, :3, :3] = START_POSE[:3, :3].to(dtype=dtype, device=p.device)
    poses[:, 0, 3] = 0.20 + 0.5 * p
    poses[:, 1, 3] = 0.0
    poses[:, 2, 3] = 0.1
    return poses


def _z_rotation(beta: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    beta = beta.to(dtype=dtype)
    c = torch.cos(beta)
    s = torch.sin(beta)

    rotation = torch.eye(3, dtype=dtype, device=beta.device)
    rotation[0, 0] = c
    rotation[0, 1] = -s
    rotation[1, 0] = s
    rotation[1, 1] = c
    return rotation


def _y_rotation_negative_alpha(
    alpha: torch.Tensor, *, dtype: torch.dtype
) -> torch.Tensor:
    alpha = alpha.to(dtype=dtype)
    c = torch.cos(alpha)
    s = torch.sin(alpha)
    n = alpha.numel()

    rotation = torch.zeros(n, 3, 3, dtype=dtype, device=alpha.device)
    rotation[:, 0, 0] = c
    rotation[:, 0, 2] = -s
    rotation[:, 1, 1] = 1.0
    rotation[:, 2, 0] = s
    rotation[:, 2, 2] = c
    return rotation


def _extra_path_pose_from_alpha_beta(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    alpha = alpha.to(dtype=dtype)
    beta = beta.to(dtype=dtype, device=alpha.device)
    s_alpha = torch.sin(alpha)
    c_alpha = torch.cos(alpha)
    n = alpha.numel()

    start_rotation = START_POSE[:3, :3].to(dtype=dtype, device=alpha.device)
    rotation_z = _z_rotation(beta, dtype=dtype)
    rotation_y = _y_rotation_negative_alpha(alpha, dtype=dtype)
    rotations = (start_rotation @ rotation_z).unsqueeze(0) @ rotation_y

    poses = torch.eye(4, dtype=dtype, device=alpha.device).repeat(n, 1, 1)
    poses[:, :3, :3] = rotations
    poses[:, 0, 3] = 0.45 - 0.25 * c_alpha
    poses[:, 1, 3] = 0.25 * s_alpha * torch.cos(math.pi / 2.0 + beta)
    poses[:, 2, 3] = 0.1 + 0.25 * s_alpha * torch.sin(math.pi / 2.0 - beta)
    return poses


def _uniform_interior_values(
    *,
    count: int,
    value_range: tuple[float, float],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return count evenly spaced interior values, excluding both endpoints."""
    low, high = value_range
    step = (high - low) / float(count + 1)
    return low + step * torch.arange(1, count + 1, dtype=dtype, device="cpu")


def _sample_sorted_uniform_random(
    *,
    generator: torch.Generator,
    count: int,
    value_range: tuple[float, float],
    dtype: torch.dtype,
) -> torch.Tensor:
    low, high = value_range
    values = low + (high - low) * torch.rand(
        count, generator=generator, dtype=dtype, device="cpu"
    )
    values, _ = torch.sort(values)
    return values


def _signature(
    *,
    seed: int,
    num_samples: int,
    num_line_samples: int,
    num_extra_paths: int,
    alpha_range_degrees: tuple[float, float],
    beta_range_degrees: tuple[float, float],
    line_p_range: tuple[float, float],
    repeat: int,
) -> dict[str, Any]:
    return {
        "version": 6,
        "seed": int(seed),
        "num_samples": int(num_samples),
        "num_line_samples": int(num_line_samples),
        "num_extra_paths": int(num_extra_paths),
        "alpha_range_degrees": [
            float(alpha_range_degrees[0]),
            float(alpha_range_degrees[1]),
        ],
        "beta_range_degrees": [
            float(beta_range_degrees[0]),
            float(beta_range_degrees[1]),
        ],
        "line_p_range": [
            float(line_p_range[0]),
            float(line_p_range[1]),
        ],
        "repeat": int(repeat),
        "goal_alpha_degrees": float(alpha_range_degrees[1]),
        "path_order": ["main_alpha", "line_x", "extra_beta"],
        "total_pose_count": int(
            (num_extra_paths + 1) * num_samples + num_line_samples + 2 * repeat
        ),
        "main_rotation": "R_start @ Ry(-alpha)",
        "extra_rotation": "R_start @ Rz(beta) @ Ry(-alpha)",
        "main_translation": "[0.45 - 0.25*cos(alpha), 0, 0.1 + 0.25*sin(alpha)]",
        "line_translation": "[0.20 + 0.5*p, 0, 0.1]",
        "extra_translation": (
            "[0.45 - 0.25*cos(alpha), "
            "0.25*sin(alpha)*cos(pi/2 + beta), "
            "0.1 + 0.25*sin(alpha)*sin(pi/2 - beta)]"
        ),
        "alpha_order": "uniform_interior_ascending",
        "beta_order": "ascending",
        "line_p_order": "uniform_interior_ascending",
        "extra_alpha_degrees": "shared_with_main_alpha",
        "start_pose": _jsonable_float_list(START_POSE),
    }


def _cache_files(cache_dir: Path, seed: int) -> list[Path]:
    return sorted(cache_dir.glob(f"taskpose_seed{int(seed)}_*.json"))


def _is_non_decreasing(values: list[float]) -> bool:
    return all(values[i] <= values[i + 1] for i in range(len(values) - 1))


def _read_cached_payload(
    cache_dir: Path,
    signature: dict[str, Any],
) -> tuple[dict[str, Any], Path] | None:
    for path in _cache_files(cache_dir, int(signature["seed"])):
        try:
            with path.open("r") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        if payload.get("signature") == signature:
            if signature.get("alpha_order") == "ascending" and not _is_non_decreasing(
                payload.get("alpha_degrees", [])
            ):
                continue
            if signature.get("line_p_order") == "ascending" and not _is_non_decreasing(
                payload.get("line_p_values", [])
            ):
                continue
            if signature.get("beta_order") == "ascending" and not _is_non_decreasing(
                payload.get("extra_beta_degrees", [])
            ):
                continue
            return payload, path
    return None


def _next_cache_path(cache_dir: Path, seed: int) -> Path:
    max_idx = -1
    prefix = f"taskpose_seed{int(seed)}_"
    for path in _cache_files(cache_dir, seed):
        stem = path.stem
        if not stem.startswith(prefix):
            continue
        try:
            max_idx = max(max_idx, int(stem.removeprefix(prefix)))
        except ValueError:
            continue
    return cache_dir / f"taskpose_seed{int(seed)}_{max_idx + 1}.json"


def _build_planner_goal_poses(
    *,
    start_pose: torch.Tensor,
    main_path_poses: torch.Tensor,
    goal_pose: torch.Tensor,
) -> torch.Tensor:
    planner_goal_poses = torch.cat(
        [
            start_pose.unsqueeze(0),
            main_path_poses,
            goal_pose.unsqueeze(0),
        ],
        dim=0,
    )
    return _collapse_consecutive_duplicate_poses(planner_goal_poses)


def _planner_goal_poses_from_payload(
    payload: dict[str, Any],
    signature: dict[str, Any],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if "planner_goal_poses" in payload:
        return torch.tensor(payload["planner_goal_poses"], dtype=dtype, device=device)

    num_samples = int(signature["num_samples"])
    sampled_poses = torch.tensor(payload["sampled_poses"], dtype=dtype, device=device)
    main_path_poses = sampled_poses[:num_samples]
    start_pose = torch.tensor(payload["single_start_pose"], dtype=dtype, device=device)
    goal_pose = torch.tensor(payload["single_goal_pose"], dtype=dtype, device=device)
    return _build_planner_goal_poses(
        start_pose=start_pose,
        main_path_poses=main_path_poses,
        goal_pose=goal_pose,
    )


def _write_cache(
    cache_dir: Path,
    signature: dict[str, Any],
    *,
    alpha_degrees: torch.Tensor,
    line_p_values: torch.Tensor,
    extra_beta_degrees: torch.Tensor,
    start_pose: torch.Tensor,
    sampled_poses: torch.Tensor,
    goal_pose: torch.Tensor,
    goal_poses: torch.Tensor,
    planner_goal_poses: torch.Tensor,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _next_cache_path(cache_dir, int(signature["seed"]))
    payload = {
        "signature": signature,
        "alpha_degrees": _jsonable_float_list(alpha_degrees),
        "line_p_values": _jsonable_float_list(line_p_values),
        "extra_beta_degrees": _jsonable_float_list(extra_beta_degrees),
        "sampled_poses": _jsonable_float_list(sampled_poses),
        "single_start_pose": _jsonable_float_list(start_pose),
        "single_goal_pose": _jsonable_float_list(goal_pose),
        "goal_poses": _jsonable_float_list(goal_poses),
        "planner_goal_poses": _jsonable_float_list(planner_goal_poses),
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
    return path


def task_sampler(
    seed: int | None = None,
    start_pose: torch.Tensor | None = None,
    num_samples: int = NUM_SAMPLES,
    num_line_samples: int = NUM_LINE_SAMPLES,
    num_extra_paths: int = NUM_EXTRA_PATHS,
    alpha_range_degrees: tuple[float, float] = ALPHA_RANGE_DEGREES,
    beta_range_degrees: tuple[float, float] = BETA_RANGE_DEGREES,
    line_p_range: tuple[float, float] = LINE_P_RANGE,
    repeat: int = REPEAT_START_GOAL,
    cache_dir: str | Path = CACHE_DIR,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
    return_planner_goal_poses: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Sample task poses and cache them under ``initial_candidates``.

    Returns:
        By default, optimizer tensor
        [(num_extra_paths + 2)*num_samples + 2*repeat, 4, 4]. The first block is
        repeated START_POSE, the middle block is sampled path poses, and the
        final block is the first-path max-alpha goal pose repeated ``repeat``.
        If return_planner_goal_poses is True, also returns planner poses containing
        only start, the main y=0 half-circle path, and the final main goal, with
        adjacent duplicates collapsed.
    """
    if seed is None:
        seed = int(torch.initial_seed() % (2**32))
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if num_line_samples < 0:
        raise ValueError("num_line_samples must be non-negative.")
    if num_extra_paths < 0:
        raise ValueError("num_extra_paths must be non-negative.")
    if repeat < 0:
        raise ValueError("repeat must be non-negative.")
    if alpha_range_degrees[0] > alpha_range_degrees[1]:
        raise ValueError("alpha_range_degrees must be ordered as (min, max).")
    if beta_range_degrees[0] > beta_range_degrees[1]:
        raise ValueError("beta_range_degrees must be ordered as (min, max).")
    if line_p_range[0] > line_p_range[1]:
        raise ValueError("line_p_range must be ordered as (min, max).")

    device = torch.device(device) if device is not None else _default_device()
    cache_dir = Path(cache_dir)
    start_cpu = (
        _as_pose_tensor(start_pose, dtype=dtype)
        if start_pose is not None
        else START_POSE.to(dtype=dtype)
    )

    signature = _signature(
        seed=seed,
        num_samples=num_samples,
        num_line_samples=num_line_samples,
        num_extra_paths=num_extra_paths,
        alpha_range_degrees=alpha_range_degrees,
        beta_range_degrees=beta_range_degrees,
        line_p_range=line_p_range,
        repeat=repeat,
    )
    signature["start_pose"] = _jsonable_float_list(start_cpu)

    cached = _read_cached_payload(cache_dir, signature)
    if cached is not None:
        payload, cache_path = cached
        print(f"[Info] Loaded task pose cache: {cache_path}")
        goal_poses = torch.tensor(payload["goal_poses"], dtype=dtype, device=device)
        if return_planner_goal_poses:
            planner_goal_poses = _planner_goal_poses_from_payload(
                payload,
                signature,
                dtype=dtype,
                device=device,
            )
            return goal_poses, planner_goal_poses
        return goal_poses

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    alpha_degrees = _uniform_interior_values(
        count=num_samples,
        value_range=alpha_range_degrees,
        dtype=dtype,
    )
    alpha = torch.deg2rad(alpha_degrees).to(device=device)
    main_path_poses = _pose_from_alpha(alpha, dtype=dtype)

    line_p_values = _uniform_interior_values(
        count=num_line_samples,
        value_range=line_p_range,
        dtype=dtype,
    )
    if num_line_samples > 0:
        line_path_poses = _line_pose_from_p(
            line_p_values.to(device=device), dtype=dtype
        )
    else:
        line_path_poses = torch.empty(0, 4, 4, dtype=dtype, device=device)

    extra_beta_degrees = _sample_sorted_uniform_random(
        generator=generator,
        count=num_extra_paths,
        value_range=beta_range_degrees,
        dtype=dtype,
    )
    extra_path_poses = []
    for beta_degree in extra_beta_degrees:
        beta = torch.deg2rad(beta_degree).to(device=device)
        extra_path_poses.append(
            _extra_path_pose_from_alpha_beta(alpha, beta, dtype=dtype)
        )

    if extra_path_poses:
        extra_path_poses_tensor = torch.cat(extra_path_poses, dim=0)
    else:
        extra_path_poses_tensor = torch.empty(0, 4, 4, dtype=dtype, device=device)

    sampled_poses = torch.cat(
        [main_path_poses, line_path_poses, extra_path_poses_tensor], dim=0
    )

    goal_alpha = torch.deg2rad(
        torch.tensor([alpha_range_degrees[1]], dtype=dtype, device=device)
    )
    goal_pose = _pose_from_alpha(goal_alpha, dtype=dtype)[0]
    start = start_cpu.to(device=device)

    goal_poses = torch.cat(
        [
            start.unsqueeze(0).repeat(repeat, 1, 1),
            sampled_poses,
            goal_pose.unsqueeze(0).repeat(repeat, 1, 1),
        ],
        dim=0,
    )
    expected_count = (num_extra_paths + 1) * num_samples + num_line_samples + 2 * repeat
    if goal_poses.shape[0] != expected_count:
        raise RuntimeError(
            f"task pose count mismatch: expected {expected_count}, got "
            f"{goal_poses.shape[0]}."
        )

    planner_goal_poses = _build_planner_goal_poses(
        start_pose=start,
        main_path_poses=main_path_poses,
        goal_pose=goal_pose,
    )

    cache_path = _write_cache(
        cache_dir,
        signature,
        alpha_degrees=alpha_degrees,
        line_p_values=line_p_values,
        extra_beta_degrees=extra_beta_degrees,
        start_pose=start,
        sampled_poses=sampled_poses,
        goal_pose=goal_pose,
        goal_poses=goal_poses,
        planner_goal_poses=planner_goal_poses,
    )
    print(f"[Info] Wrote task pose cache: {cache_path}")
    if return_planner_goal_poses:
        return goal_poses, planner_goal_poses
    return goal_poses


def create_task_pose_sets(
    seed: int | None = None,
    *,
    start_pose: torch.Tensor | None = None,
    num_samples: int = NUM_SAMPLES,
    num_line_samples: int = NUM_LINE_SAMPLES,
    num_extra_paths: int = NUM_EXTRA_PATHS,
    repeat: int = REPEAT_START_GOAL,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    optimizer_goal_poses, planner_goal_poses = task_sampler(
        seed=seed,
        start_pose=start_pose,
        num_samples=num_samples,
        num_line_samples=num_line_samples,
        num_extra_paths=num_extra_paths,
        repeat=repeat,
        device=device,
        dtype=dtype,
        return_planner_goal_poses=True,
    )
    return optimizer_goal_poses, planner_goal_poses


def create_start_goal_poses(
    *,
    start_pose: torch.Tensor | None = None,
    alpha_range_degrees: tuple[float, float] = ALPHA_RANGE_DEGREES,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    device = torch.device(device) if device is not None else _default_device()
    start = (
        _as_pose_tensor(start_pose, dtype=dtype)
        if start_pose is not None
        else START_POSE.to(dtype=dtype)
    ).to(device=device)
    goal_alpha = torch.deg2rad(
        torch.tensor([alpha_range_degrees[1]], dtype=dtype, device=device)
    )
    goal_pose = _pose_from_alpha(goal_alpha, dtype=dtype)[0]
    return torch.stack([start, goal_pose], dim=0)
