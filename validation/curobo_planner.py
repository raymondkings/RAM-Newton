import os
import tempfile
import traceback

import torch
import numpy as np

from interface import Morphology, Task
from interface.plan_result import PlanResult
from util.mdh import to_urdf
from util.kinematics import (
    forward_kinematics,
    build_sphere_dict,
    build_self_collision_ignore,
    _mat_to_goal_pose,
    build_scene,
)
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import JointState


class CuroboPlanner:
    """cuRobo MotionPlanner wrapper for a fixed (Morphology, Task) pair."""

    def __init__(
        self,
        morph: Morphology,
        task: Task,
        device: torch.device,
        num_ik_seeds: int = 32,
        num_trajopt_seeds: int = 4,
        ignore_ground: bool = False,
        ignore_obstacles: bool = False,
    ) -> None:
        self._device = device
        self._morph = morph

        n = morph.n_links
        ee_link = f"link_{n - 1}"

        dtype = morph.params.dtype
        base_pose_f32 = task.environment.base_pose.to(torch.float32)
        self._base_pose_f32 = base_pose_f32
        self._base_pose_inv = torch.linalg.inv(base_pose_f32).to(dtype)

        # Shared robot dict and scene (used by both IK and motion planner)
        urdf_str = to_urdf(morph)
        tmp = tempfile.NamedTemporaryFile(suffix=".urdf", delete=False, mode="w")
        tmp.write(urdf_str)
        tmp.flush()
        tmp.close()
        self._urdf_path = tmp.name

        sphere_dict = build_sphere_dict(morph)
        robot_dict = {
            "kinematics": {
                "urdf_path": self._urdf_path,
                "base_link": "base_link",
                "tool_frames": [ee_link],
                "collision_spheres": sphere_dict,
                "collision_link_names": list(sphere_dict.keys()),
                "collision_sphere_buffer": 0.01,
                "self_collision_buffer": {k: 0.0 for k in sphere_dict},
                "self_collision_ignore": build_self_collision_ignore(morph),
            }
        }
        self.scene = build_scene(
            task,
            torch.linalg.inv(base_pose_f32),
            ignore_ground=ignore_ground,
            ignore_obstacles=ignore_obstacles,
        )

        # Motion planner for trajectory planning
        config = MotionPlannerCfg.create(
            robot=robot_dict,
            scene_model=self.scene,
            collision_cache={"cuboid": 20},
            num_ik_seeds=num_ik_seeds,
            num_trajopt_seeds=num_trajopt_seeds,
        )
        self._planner = MotionPlanner(config)
        self._planner.warmup(enable_graph=True, num_warmup_iterations=3)

    def robot_spheres_world(self, q: torch.Tensor) -> np.ndarray:
        """Return collision sphere positions for joint config q in world frame.

        Returns an (N, 4) float32 array of (x, y, z, radius) values.
        """
        n = self._morph.n_links
        q_cpu = q.float().cpu()
        # MDH has n rows; last row is the EE link (theta=0)
        theta = torch.zeros(n, 1, device="cpu")
        theta[: n - 1] = q_cpu.unsqueeze(1)
        mdh = self._morph.params.float().cpu()
        link_poses = forward_kinematics(mdh, theta)  # [n, 4, 4] in robot-local frame
        base = self._base_pose_f32.float().cpu()
        link_poses_world = (base @ link_poses).numpy()  # [n, 4, 4] in world frame

        sphere_dict = build_sphere_dict(self._morph)
        results = []
        for j in range(n):
            T = link_poses_world[j]
            for s in sphere_dict.get(f"link_{j}", []):
                c = np.array(s["center"] + [1.0])
                p = T @ c
                results.append([p[0], p[1], p[2], s["radius"]])

        return (
            np.array(results, dtype=np.float32)
            if results
            else np.zeros((0, 4), dtype=np.float32)
        )

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

        gp = self._planner.graph_planner
        if gp is not None:
            q_check = start_q.float().unsqueeze(0).to(self._device)  # [1, n_joints]
            start_feasible = gp.check_samples_feasibility(q_check)
            if not start_feasible.all():
                print("[cuRobo] start state is in collision (self or world)")
                kin = self._planner.compute_kinematics(start_state)
                if kin.robot_spheres is not None:
                    # spheres: [batch, horizon, num_spheres, 4] — (x, y, z, r) in robot-local frame
                    # ground top face is at z=0 in local frame
                    spheres = kin.robot_spheres.reshape(-1, 4)
                    ground_penetration = spheres[:, 2] - spheres[:, 3]  # z - r
                    if (ground_penetration < 0).any():
                        worst = ground_penetration.min().item()
                        print(
                            f"[cuRobo]   -> ground plane collision (deepest penetration: {worst:.4f} m)"
                        )
                    else:
                        print(
                            "[cuRobo]   -> no ground collision; likely self-collision"
                        )

        try:
            result = self._planner.plan_pose(
                goal, start_state, max_attempts=max_attempts
            )
        except Exception:
            traceback.print_exc()
            raise

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

    def plan_sequence(
        self,
        goal_poses_world: torch.Tensor,
        start_q: torch.Tensor,
        max_attempts: int = 5,
    ) -> tuple[PlanResult, torch.Tensor | None]:
        """Plan a trajectory through a sequence of goal poses.

        Chains individual plans: start_q → goal[0] → goal[1] → … → goal[N-1].
        Returns the concatenated path and the final joint config, or None on failure.
        """
        print(
            f"Planning sequence of {goal_poses_world.shape[0]} goals with cuRobo (GPU TrajOpt + graph search)..."
        )
        full_path: list[torch.Tensor] = []
        current_q = start_q

        for i in range(goal_poses_world.shape[0]):
            result, goal_q = self.plan(goal_poses_world[i], current_q, max_attempts)
            if not result.success or goal_q is None:
                return PlanResult(
                    success=False,
                    path=full_path,
                    n_iterations=0,
                    n_nodes=len(full_path),
                    failed_at_goal=i,
                ), None
            full_path.extend(result.path)
            current_q = goal_q

        return PlanResult(
            success=True, path=full_path, n_iterations=1, n_nodes=len(full_path)
        ), current_q

    def __del__(self) -> None:
        try:
            os.unlink(self._urdf_path)
        except Exception:
            pass


def interpolate_path(
    path: list[torch.Tensor], step: float = 0.02
) -> list[torch.Tensor]:
    """Densify a joint-space path so consecutive frames are at most `step` apart."""
    if len(path) < 2:
        return path
    out = [path[0]]
    for q_a, q_b in zip(path[:-1], path[1:]):
        delta = q_b - q_a
        n = max(1, int(torch.ceil(delta.norm() / step).item()))
        for k in range(1, n + 1):
            out.append(q_a + (k / n) * delta)
    return out
