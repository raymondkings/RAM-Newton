from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch


NUM_SAMPLES = 300
NUM_GAUSSIAN_NOISE_SAMPLES = 3
GAUSSIAN_MEAN = 0.0
GAUSSIAN_VARIANCE = math.radians(5.0) ** 2
ALPHA_RANGE_DEGREES = (0.0, 180.0)
REPEAT_START_GOAL = 10

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "initial_candidates"

# Interprets the handwritten START_POSE as the alpha=0 orientation with the
# missing +x tool-axis entry restored in the first row.
START_POSE = torch.tensor(
    [
        [0.0, 0.0, 1.0, 0.2],
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.175],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=torch.float32,
)


def _default_device() -> torch.device:
    try:
        return torch.get_default_device()
    except AttributeError:
        return torch.empty(()).device


def _jsonable_float_list(tensor: torch.Tensor) -> list:
    return tensor.detach().cpu().tolist()


def _as_pose_tensor(pose: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    pose = pose.detach().cpu().to(dtype=dtype)
    if pose.shape != (4, 4):
        raise ValueError(f"start_pose must have shape (4, 4), got {tuple(pose.shape)}.")
    return pose


def _pose_from_alpha(alpha: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    """Create base poses from alpha values in radians.

    The rotation block follows the user's derivation:
        [[sin(a),  0, cos(a)],
         [0,      -1, 0     ],
         [cos(a),  0, -sin(a)]]

    Translation follows the corrected formula:
        x = 0.45 - 0.25*cos(alpha)
        z = 0.175 + 0.25*sin(alpha)
    """
    alpha = alpha.to(dtype=dtype)
    s = torch.sin(alpha)
    c = torch.cos(alpha)
    n = alpha.numel()

    poses = torch.eye(4, dtype=dtype, device=alpha.device).repeat(n, 1, 1)
    poses[:, 0, 0] = s
    poses[:, 0, 1] = 0.0
    poses[:, 0, 2] = c
    poses[:, 1, 0] = 0.0
    poses[:, 1, 1] = -1.0
    poses[:, 1, 2] = 0.0
    poses[:, 2, 0] = c
    poses[:, 2, 1] = 0.0
    poses[:, 2, 2] = -s

    poses[:, 0, 3] = 0.45 - 0.25 * c
    poses[:, 1, 3] = 0.0
    poses[:, 2, 3] = 0.175 + 0.25 * s
    return poses


def _z_rotation_transform(angles: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    angles = angles.to(dtype=dtype)
    c = torch.cos(angles)
    s = torch.sin(angles)
    n = angles.numel()

    transforms = torch.eye(4, dtype=dtype, device=angles.device).repeat(n, 1, 1)
    transforms[:, 0, 0] = c
    transforms[:, 0, 1] = -s
    transforms[:, 1, 0] = s
    transforms[:, 1, 1] = c
    return transforms


def _signature(
    *,
    seed: int,
    num_samples: int,
    num_gaussian_noise_samples: int,
    gaussian_mean: float,
    gaussian_variance: float,
    alpha_range_degrees: tuple[float, float],
    repeat: int,
) -> dict[str, Any]:
    return {
        "version": 1,
        "seed": int(seed),
        "num_samples": int(num_samples),
        "num_gaussian_noise_samples": int(num_gaussian_noise_samples),
        "gaussian_mean": float(gaussian_mean),
        "gaussian_variance": float(gaussian_variance),
        "alpha_range_degrees": [
            float(alpha_range_degrees[0]),
            float(alpha_range_degrees[1]),
        ],
        "repeat": int(repeat),
        "translation_z_mode": "0.175 + 0.25*sin(alpha)",
        "alpha_order": "ascending",
        "noise_multiplication": "base_pose @ local_z_rotation_noise",
        "start_pose": _jsonable_float_list(START_POSE),
    }


def _cache_files(cache_dir: Path, seed: int) -> list[Path]:
    return sorted(cache_dir.glob(f"taskpose_seed{int(seed)}_*.json"))


def _read_cached_goal_poses(
    cache_dir: Path,
    signature: dict[str, Any],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    for path in _cache_files(cache_dir, int(signature["seed"])):
        try:
            with path.open("r") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        if payload.get("signature") == signature:
            return torch.tensor(payload["goal_poses"], dtype=dtype, device=device)
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


def _write_cache(
    cache_dir: Path,
    signature: dict[str, Any],
    *,
    alpha_degrees: torch.Tensor,
    gaussian_noise_angles: torch.Tensor,
    base_sampled_poses: torch.Tensor,
    goal_pose: torch.Tensor,
    goal_poses: torch.Tensor,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _next_cache_path(cache_dir, int(signature["seed"]))
    payload = {
        "signature": signature,
        "alpha_degrees": _jsonable_float_list(alpha_degrees),
        "gaussian_noise_angles_radians": _jsonable_float_list(gaussian_noise_angles),
        "base_sampled_poses": _jsonable_float_list(base_sampled_poses),
        "single_goal_pose": _jsonable_float_list(goal_pose),
        "goal_poses": _jsonable_float_list(goal_poses),
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
    return path


def task_sampler(
    seed: int | None = None,
    start_pose: torch.Tensor | None = None,
    num_samples: int = NUM_SAMPLES,
    num_gaussian_noise_samples: int = NUM_GAUSSIAN_NOISE_SAMPLES,
    gaussian_mean: float = GAUSSIAN_MEAN,
    gaussian_variance: float = GAUSSIAN_VARIANCE,
    alpha_range_degrees: tuple[float, float] = ALPHA_RANGE_DEGREES,
    repeat: int = REPEAT_START_GOAL,
    cache_dir: str | Path = CACHE_DIR,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Sample task poses and cache them under ``initial_candidates``.

    Returns:
        Tensor [repeat + num_samples*num_gaussian_noise_samples + repeat, 4, 4].
        The first block is repeated START_POSE, the middle block is noisy sampled
        poses, and the final block is the alpha=180 goal pose repeated ``repeat``.
    """
    if seed is None:
        seed = int(torch.initial_seed() % (2**32))
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if num_gaussian_noise_samples <= 0:
        raise ValueError("num_gaussian_noise_samples must be positive.")
    if gaussian_variance < 0.0:
        raise ValueError("gaussian_variance must be non-negative.")
    if repeat <= 0:
        raise ValueError("repeat must be positive.")
    if alpha_range_degrees[0] > alpha_range_degrees[1]:
        raise ValueError("alpha_range_degrees must be ordered as (min, max).")

    device = torch.device(device) if device is not None else _default_device()
    cache_dir = Path(cache_dir)
    start = (
        _as_pose_tensor(start_pose, dtype=dtype)
        if start_pose is not None
        else START_POSE.to(dtype=dtype)
    )

    signature = _signature(
        seed=seed,
        num_samples=num_samples,
        num_gaussian_noise_samples=num_gaussian_noise_samples,
        gaussian_mean=gaussian_mean,
        gaussian_variance=gaussian_variance,
        alpha_range_degrees=alpha_range_degrees,
        repeat=repeat,
    )
    signature["start_pose"] = _jsonable_float_list(start)

    cached = _read_cached_goal_poses(
        cache_dir,
        signature,
        dtype=dtype,
        device=device,
    )
    if cached is not None:
        return cached

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    low, high = alpha_range_degrees
    alpha_degrees = low + (high - low) * torch.rand(
        num_samples, generator=generator, dtype=dtype, device="cpu"
    )
    alpha_degrees, _ = torch.sort(alpha_degrees)
    alpha = torch.deg2rad(alpha_degrees)
    base_sampled_poses = _pose_from_alpha(alpha, dtype=dtype)

    noise_std = math.sqrt(float(gaussian_variance))
    gaussian_noise_angles = torch.normal(
        mean=float(gaussian_mean),
        std=noise_std,
        size=(num_samples, num_gaussian_noise_samples),
        generator=generator,
        dtype=dtype,
        device="cpu",
    )
    noise_tf = _z_rotation_transform(
        gaussian_noise_angles.reshape(-1), dtype=dtype
    ).reshape(num_samples, num_gaussian_noise_samples, 4, 4)
    sampled_poses = (base_sampled_poses[:, None, :, :] @ noise_tf).reshape(-1, 4, 4)

    goal_alpha = torch.tensor([math.pi], dtype=dtype)
    goal_pose = _pose_from_alpha(goal_alpha, dtype=dtype)[0]

    goal_poses = torch.cat(
        [
            start.unsqueeze(0).repeat(repeat, 1, 1),
            sampled_poses,
            goal_pose.unsqueeze(0).repeat(repeat, 1, 1),
        ],
        dim=0,
    )

    _write_cache(
        cache_dir,
        signature,
        alpha_degrees=alpha_degrees,
        gaussian_noise_angles=gaussian_noise_angles,
        base_sampled_poses=base_sampled_poses,
        goal_pose=goal_pose,
        goal_poses=goal_poses,
    )
    return goal_poses.to(device=device)


def create_task(
    seed: int | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return task_sampler(seed=seed, device=device, dtype=dtype)
