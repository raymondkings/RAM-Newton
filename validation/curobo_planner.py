"""Motion planning via cuRobo (NVIDIA GPU-accelerated trajectory optimisation).

Install from source into this project's venv:
    git clone https://github.com/NVlabs/curobo && cd curobo
    nvidia-smi | grep CUDA   # check CUDA version
    uv pip install --python /u/halle/arke/home_at/Documents/nrm-newton/.venv/bin/python .[cu12]
See https://nvlabs.github.io/curobo/latest/getting-started/installation.html
"""
from __future__ import annotations

import math
import os
import tempfile
import xml.etree.ElementTree as ET

import torch
import numpy as np
from scipy.spatial.transform import Rotation

from interface import Morphology, Task
from interface.environment import Box, Sphere as EnvSphere, Capsule as EnvCapsule
from validation.planner import PlanResult

try:
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import JointState, GoalToolPose
    from curobo.scene import Scene, Cuboid, Sphere as SceneSphere

    CUROBO_AVAILABLE = True
    _CUROBO_IMPORT_ERROR: Exception | None = None
except ImportError as e:
    CUROBO_AVAILABLE = False
    _CUROBO_IMPORT_ERROR = e


# ---------------------------------------------------------------------------
# URDF generation from MDH parameters
# ---------------------------------------------------------------------------

def _joint_limits(morph: Morphology) -> list[tuple[float, float]]:
    """Per-joint angle limits derived from MDH a-offsets to prevent non-adjacent self-collision.

    When joint j rotates by θ, the horizontal distance between the d-capsule axes of
    link_{j-1} and link_{j+1} (approximating α≈0) obeys the law of cosines:

        dist(θ) = √(a_j² + a_{j+1}² + 2·a_j·a_{j+1}·cos(θ))

    Self-collision occurs when dist(θ) < 2r.  Solving for the critical angle:

        cos(θ_crit) = (4r² − a_j² − a_{j+1}²) / (2·a_j·a_{j+1})

    The joint is limited to |θ| ≤ θ_crit.
    """
    r = morph.link_radius
    n_joints = morph.n_links - 1
    limits: list[tuple[float, float]] = []

    for j in range(n_joints):
        a_j = abs(morph.a[j].item())
        a_j1 = abs(morph.a[j + 1].item())  # always valid: j+1 ≤ n_links-1

        if a_j > 0.0 and a_j1 > 0.0:
            cos_crit = (4.0 * r**2 - a_j**2 - a_j1**2) / (2.0 * a_j * a_j1)
            if cos_crit <= -1.0:
                # Even fully folded (θ=π) the capsules stay > 2r apart — no limit needed
                limits.append((-math.pi, math.pi))
            elif cos_crit >= 1.0:
                # Even at θ=0 the capsules overlap — restrict to ±π/2 as a safe fallback
                limits.append((-math.pi / 2.0, math.pi / 2.0))
            else:
                theta_crit = math.acos(cos_crit)
                limits.append((-theta_crit, theta_crit))
        else:
            # One of the a-offsets is zero: no lateral displacement → no folding risk
            limits.append((-math.pi, math.pi))

    return limits


def _morphology_to_urdf(morph: Morphology) -> str:
    """Generate a URDF XML string for the given MDH morphology.

    For Modified DH, the full joint transform is:
        T_i(θ) = Rₓ(αᵢ) · Tₓ(aᵢ) · R_z(θᵢ) · T_z(dᵢ)

    Because T_z and R_z share the same axis they commute, so:
        T_i(θ) = Rₓ(αᵢ) · Tₓ(aᵢ) · T_z(dᵢ) · R_z(θᵢ)  =  T_origin · R_z(θᵢ)

    T_origin maps to a URDF joint origin with:
        xyz = (aᵢ, −sin(αᵢ)·dᵢ, cos(αᵢ)·dᵢ)
        rpy = (αᵢ, 0, 0)
        axis = z
    """
    n = morph.n_links
    n_joints = n - 1  # revolute DOFs; last MDH row becomes the fixed EE joint

    robot = ET.Element("robot", name="mdh_robot")
    ET.SubElement(robot, "link", name="base_link")
    for i in range(n):
        ET.SubElement(robot, "link", name=f"link_{i}")

    joint_lims = _joint_limits(morph)

    for i in range(n):
        alpha = morph.alpha[i].item()
        a = morph.a[i].item()
        d = morph.d[i].item()
        sa, ca = math.sin(alpha), math.cos(alpha)

        parent = "base_link" if i == 0 else f"link_{i - 1}"
        child = f"link_{i}"
        joint_type = "fixed" if i == n_joints else "revolute"

        jnt = ET.SubElement(robot, "joint", name=f"joint_{i}", type=joint_type)
        ET.SubElement(jnt, "parent", link=parent)
        ET.SubElement(jnt, "child", link=child)
        ET.SubElement(
            jnt, "origin",
            xyz=f"{a:.8f} {-sa * d:.8f} {ca * d:.8f}",
            rpy=f"{alpha:.8f} 0.0 0.0",
        )
        ET.SubElement(jnt, "axis", xyz="0 0 1")
        if joint_type == "revolute":
            lower, upper = joint_lims[i]
            ET.SubElement(
                jnt, "limit",
                lower=f"{lower:.6f}",
                upper=f"{upper:.6f}",
                effort="1000",
                velocity=f"{math.pi:.6f}",
            )

    return '<?xml version="1.0"?>\n' + ET.tostring(robot, encoding="unicode")


# ---------------------------------------------------------------------------
# Collision-sphere approximation of MDH capsule geometry
# ---------------------------------------------------------------------------

def _capsule_spheres(
    center: list[float],
    half_height: float,
    axis: list[float],
    radius: float,
) -> list[dict]:
    """Approximate a capsule with uniformly-spaced spheres along its axis."""
    n = max(2, math.ceil(2.0 * half_height / radius) + 1)
    return [
        {
            "center": [
                center[j] + (-1.0 + 2.0 * k / (n - 1)) * half_height * axis[j]
                for j in range(3)
            ],
            "radius": float(radius),
        }
        for k in range(n)
    ]


def _build_sphere_dict(morph: Morphology) -> dict[str, list[dict]]:
    """Per-link collision sphere dicts matching Newton's capsule geometry.

    Newton attaches:
      - d-capsule of MDH row i  → body i  (link_i), along local z
      - a-capsule of MDH row i  → body i-1 (link_{i-1}), along local x  (i > 0)

    So link_j carries: d-capsule[j]  +  a-capsule[j+1].
    """
    n = morph.n_links
    r = morph.link_radius
    sphere_dict: dict[str, list[dict]] = {}

    for j in range(n):
        spheres: list[dict] = []

        d_val = morph.d[j].item()
        if abs(d_val) > 2.0 * r:
            spheres += _capsule_spheres(
                center=[0.0, 0.0, -d_val / 2.0],
                half_height=abs(d_val) / 2.0,
                axis=[0.0, 0.0, 1.0],
                radius=r,
            )

        if j < n - 1:
            a_next = morph.a[j + 1].item()
            if abs(a_next) > 2.0 * r:
                spheres += _capsule_spheres(
                    center=[a_next / 2.0, 0.0, 0.0],
                    half_height=abs(a_next) / 2.0,
                    axis=[1.0, 0.0, 0.0],
                    radius=r,
                )

        sphere_dict[f"link_{j}"] = spheres or [{"center": [0.0, 0.0, 0.0], "radius": float(r)}]

    return sphere_dict


def _self_collision_ignore(morph: Morphology) -> dict[str, list[str]]:
    """Ignore adjacent and two-hop link pairs.

    One-hop (link_j / link_{j+1}): share a joint endpoint → always in false contact.

    Two-hop (link_j / link_{j+2}): link_j's sphere set contains a-capsule[j+1], which
    is physically located at the joint between bodies j and j+1.  That places it
    immediately adjacent to d-capsule[j+2] on link_{j+2}, so cuRobo reports false
    contact there too.  The dict must also be symmetric — cuRobo checks both orderings.
    """
    links = ["base_link"] + [f"link_{i}" for i in range(morph.n_links)]
    ignore: dict[str, list[str]] = {lnk: [] for lnk in links}
    for i, a in enumerate(links):
        for b in links[i + 1 : i + 3]:   # one-hop and two-hop
            ignore[a].append(b)
            ignore[b].append(a)
    return ignore


# ---------------------------------------------------------------------------
# World / scene construction
# ---------------------------------------------------------------------------

def _build_scene(task: Task, base_pose_inv: torch.Tensor) -> "Scene":
    """Convert task obstacles to a cuRobo Scene in robot-local frame."""
    R_inv = base_pose_inv[:3, :3].float().numpy()
    t_inv = base_pose_inv[:3, 3].float().numpy()

    def _to_local(p_world: np.ndarray) -> np.ndarray:
        return R_inv @ p_world + t_inv

    cuboids: list[Cuboid] = []
    scene_spheres: list[SceneSphere] = []

    # Ground plane — top face at z = 0, depth 1 m (matches Newton's add_ground_collision)
    ground_center = _to_local(np.array([0.0, 0.0, -0.5]))
    cuboids.append(Cuboid(
        name="ground",
        dims=[100.0, 100.0, 1.0],
        pose=ground_center.tolist() + [1.0, 0.0, 0.0, 0.0],
    ))

    for idx, obs in enumerate(task.environment.obstacles):
        if isinstance(obs, Box):
            c_l = _to_local(obs.center.float().numpy())
            q_w = Rotation.from_quat(obs.rotation[[1, 2, 3, 0]].float().numpy())  # wxyz→xyzw
            q_l = (Rotation.from_matrix(R_inv) * q_w).as_quat()  # xyzw
            pose = c_l.tolist() + [float(q_l[3]), float(q_l[0]), float(q_l[1]), float(q_l[2])]
            cuboids.append(Cuboid(
                name=f"box_{idx}",
                dims=(2.0 * obs.half_extents).float().tolist(),
                pose=pose,
            ))
        elif isinstance(obs, EnvSphere):
            c_l = _to_local(obs.center.float().numpy())
            scene_spheres.append(SceneSphere(
                name=f"sphere_{idx}",
                radius=float(obs.radius),
                pose=c_l.tolist() + [1.0, 0.0, 0.0, 0.0],
            ))
        elif isinstance(obs, EnvCapsule):
            # Approximate with a bounding box
            c_l = _to_local(obs.center.float().numpy())
            q_w = Rotation.from_quat(obs.rotation[[1, 2, 3, 0]].float().numpy())
            q_l = (Rotation.from_matrix(R_inv) * q_w).as_quat()
            pose = c_l.tolist() + [float(q_l[3]), float(q_l[0]), float(q_l[1]), float(q_l[2])]
            side = float(obs.radius * 2)
            length = float(2.0 * obs.half_height + 2.0 * obs.radius)
            cuboids.append(Cuboid(name=f"capsule_{idx}", dims=[side, side, length], pose=pose))

    return Scene(cuboid=cuboids, sphere=scene_spheres)


# ---------------------------------------------------------------------------
# Pose conversion
# ---------------------------------------------------------------------------

def _mat_to_goal_pose(
    pose_4x4: torch.Tensor,
    tool_frames: list[str],
    device: torch.device,
) -> "GoalToolPose":
    """Convert a (4, 4) homogeneous matrix to a cuRobo GoalToolPose."""
    pos = pose_4x4[:3, 3].float().reshape(1, 1, 1, 1, 3).to(device)
    q_xyzw = Rotation.from_matrix(pose_4x4[:3, :3].float().cpu().numpy()).as_quat()
    quat = torch.tensor(
        [float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])],
        dtype=torch.float32,
        device=device,
    ).reshape(1, 1, 1, 1, 4)
    return GoalToolPose(tool_frames=tool_frames, position=pos, quaternion=quat)


# ---------------------------------------------------------------------------
# CuroboPlanner
# ---------------------------------------------------------------------------

class CuroboPlanner:
    """cuRobo MotionPlanner wrapper for a fixed (Morphology, Task) pair.

    Builds the URDF, collision spheres, and Scene at construction, then
    exposes plan() for repeated fast planning calls.
    """

    def __init__(
        self,
        morph: Morphology,
        task: Task,
        device: str | None = None,
        num_ik_seeds: int = 32,
        num_trajopt_seeds: int = 4,
    ) -> None:
        if not CUROBO_AVAILABLE:
            raise ImportError(
                "cuRobo is not installed. "
                "See https://nvlabs.github.io/curobo/latest/getting-started/installation.html"
            )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        n = morph.n_links
        n_joints = n - 1
        ee_link = f"link_{n - 1}"

        dtype = morph.params.dtype
        base_pose_f32 = task.environment.base_pose.to(torch.float32)
        self._base_pose_inv = torch.linalg.inv(base_pose_f32).to(dtype)

        # Write URDF to a temp file (cuRobo requires a file path)
        urdf_str = _morphology_to_urdf(morph)
        tmp = tempfile.NamedTemporaryFile(suffix=".urdf", delete=False, mode="w")
        tmp.write(urdf_str)
        tmp.flush()
        tmp.close()
        self._urdf_path = tmp.name

        robot_dict = {
            "kinematics": {
                "urdf_path": self._urdf_path,
                "base_link": "base_link",
                "tool_frames": [ee_link],
                "collision_spheres": _build_sphere_dict(morph),
                "collision_sphere_buffer": 0.002,
                "self_collision_ignore": _self_collision_ignore(morph),
            }
        }

        scene = _build_scene(task, torch.linalg.inv(base_pose_f32))

        config = MotionPlannerCfg.create(
            robot=robot_dict,
            scene_model=scene,
            num_ik_seeds=num_ik_seeds,
            num_trajopt_seeds=num_trajopt_seeds,
        )
        self._planner = MotionPlanner(config)
        self._planner.warmup(enable_graph=True, num_warmup_iterations=3)

    def plan(
        self,
        goal_pose_world: torch.Tensor,
        start_q: torch.Tensor | None = None,
        max_attempts: int = 5,
    ) -> tuple[PlanResult, torch.Tensor | None]:
        """Plan a trajectory from start_q to goal_pose_world.

        Returns (PlanResult, goal_q) where goal_q is the final joint config,
        or None if no solution was found.
        """
        dtype = self._base_pose_inv.dtype
        n_joints = len(self._planner.joint_names)
        if start_q is None:
            start_q = torch.zeros(n_joints, dtype=dtype)

        goal_local = self._base_pose_inv @ goal_pose_world.to(dtype)
        goal = _mat_to_goal_pose(goal_local, self._planner.tool_frames, self._device)

        start_state = JointState.from_position(
            start_q.float().unsqueeze(0).to(self._device),
            joint_names=self._planner.joint_names,
        )

        result = self._planner.plan_pose(goal, start_state, max_attempts=max_attempts)

        if result is None or not result.success.any():
            return PlanResult(success=False, path=[], n_iterations=0, n_nodes=0), None

        interp = result.get_interpolated_plan()
        n_joints = len(self._planner.joint_names)
        # interp.position may have leading batch/seed dims → flatten to [T, n_joints]
        positions = interp.position.cpu().to(dtype).reshape(-1, n_joints)
        path = [positions[t] for t in range(positions.shape[0])]
        return (
            PlanResult(success=True, path=path, n_iterations=1, n_nodes=len(path)),
            path[-1],
        )

    def __del__(self) -> None:
        try:
            os.unlink(self._urdf_path)
        except Exception:
            pass
