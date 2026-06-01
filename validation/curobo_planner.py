import contextlib
import io
import os
import tempfile
import traceback

import torch
import numpy as np
from scipy.spatial.transform import Rotation

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
    """cuRobo MotionPlanner wrapper for a fixed (Morphology, Task) pair.

    Failure diagnostics — see ``_diagnose_failure`` — classify each failure as one of:
    ``joint limits`` (start_q out of bounds), ``IK`` (no IK seed converged on the goal),
    ``collision`` (IK reached the goal but graph seeding rejected every seed for being
    in collision), ``TrajOpt`` (trajectory ends short of the goal), or ``trajectory``
    (trajectory reached the goal but at least one waypoint is infeasible). For the last
    case ``_first_path_violation`` scans the returned interpolated trajectory waypoint by
    waypoint and reports the first concrete violation, since cuRobo clears its internal
    per-constraint metrics during topk reduction and we can't read them off the result.
    """

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
        self._ignore_ground = ignore_ground
        self._ignore_obstacles = ignore_obstacles

        n = morph.n_links
        ee_link = f"link_{n - 1}"

        self._dtype = morph.params.dtype

        urdf_str = to_urdf(morph)
        tmp = tempfile.NamedTemporaryFile(suffix=".urdf", delete=False, mode="w")
        tmp.write(urdf_str)
        tmp.flush()
        tmp.close()
        self._urdf_path = tmp.name

        sphere_dict = build_sphere_dict(morph)
        self._sphere_dict = sphere_dict
        self._self_collision_ignore = build_self_collision_ignore(morph)
        robot_dict = {
            "kinematics": {
                "urdf_path": self._urdf_path,
                "base_link": "base_link",
                "tool_frames": [ee_link],
                "collision_spheres": sphere_dict,
                "collision_link_names": list(sphere_dict.keys()),
                "collision_sphere_buffer": 0.01,
                "self_collision_buffer": {k: 0.005 for k in sphere_dict},
                "self_collision_ignore": self._self_collision_ignore,
            }
        }
        if self._ignore_obstacles:
            self.scene = None
        else:
            self.scene = build_scene(
                task,
                ignore_ground=self._ignore_ground,
                ignore_obstacles=self._ignore_obstacles,
            )

        config = MotionPlannerCfg.create(
            robot=robot_dict,
            scene_model=self.scene,
            collision_cache={"cuboid": 20} if self.scene is not None else None,
            num_ik_seeds=num_ik_seeds,
            num_trajopt_seeds=num_trajopt_seeds,
        )
        self._planner = MotionPlanner(config)
        self._planner.warmup(enable_graph=True, num_warmup_iterations=3)

    def check_start_feasibility(self, q: torch.Tensor) -> bool:
        """Return True if q is free of self-collision, world collision, and joint limits."""
        gp = self._planner.graph_planner
        if gp is None:
            # Graph planner not initialised (warmup not called with enable_graph=True);
            # cannot verify feasibility — skip check.
            print(
                "[Warning] check_start_feasibility: graph_planner is None, skipping check."
            )
            return True
        q_check = q.float().unsqueeze(0).to(self._device)
        return bool(gp.check_samples_feasibility(q_check).all())

    def __del__(self):
        try:
            os.unlink(self._urdf_path)
        except Exception:
            pass

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def plan(
        self,
        goal_pose: torch.Tensor,
        start_q: torch.Tensor | None = None,
        max_attempts: int = 5,
    ) -> tuple[PlanResult, torch.Tensor | None]:
        """Plan a trajectory from start_q to a goal pose in the shared world/base frame.

        Returns (PlanResult, goal_q) where goal_q is the final joint config,
        or None if no solution was found.
        """
        result, goal, start_q = self._plan_pose_raw(
            goal_pose, start_q, max_attempts
        )

        if result is None or not result.success.any():
            best_ik_q = self._diagnose_failure(result, goal, start_q)
            return PlanResult(
                success=False, path=[], n_iterations=0, n_nodes=0, best_ik_q=best_ik_q
            ), None

        path = self._result_to_path(result)
        return (
            PlanResult(success=True, path=path, n_iterations=1, n_nodes=len(path)),
            path[-1],
        )

    def plan_sequence(
        self,
        goal_poses: torch.Tensor,
        start_q: torch.Tensor | None = None,
        max_attempts: int = 5,
    ) -> tuple[PlanResult, torch.Tensor | None]:
        """Plan a trajectory through a sequence of goal poses.

        Chains individual plans: start_q → goal[0] → goal[1] → … → goal[N-1].
        Returns the concatenated path and the final joint config, or None on failure.
        """
        n_total = goal_poses.shape[0]
        print(
            f"Planning sequence of {n_total} goals with cuRobo (GPU TrajOpt + graph search)..."
        )
        full_path: list[torch.Tensor] = []
        current_q = start_q

        for i in range(n_total):
            result, goal, start_q_used = self._plan_pose_raw(
                goal_poses[i], current_q, max_attempts
            )
            if result is None or not result.success.any():
                best_ik_q = self._diagnose_failure(
                    result,
                    goal,
                    start_q_used,
                    report_header=f"Goal {i}/{n_total} failed",
                )
                return PlanResult(
                    success=False,
                    path=full_path,
                    n_iterations=0,
                    n_nodes=len(full_path),
                    failed_at_goal=i,
                    best_ik_q=best_ik_q,
                ), None

            path = self._result_to_path(result)
            full_path.extend(path)
            current_q = path[-1]

        return PlanResult(
            success=True, path=full_path, n_iterations=1, n_nodes=len(full_path)
        ), current_q

    def default_start_q(self) -> torch.Tensor:
        """IK to a candidate pose above the base; falls back to zeros if all candidates fail.

        Candidate offsets are in the shared world/base frame (z-up). The planner's own IK solver
        is reused so the start config is checked against the same scene and self-collision
        config used for planning.
        """
        dtype = self._dtype
        n_actuated = len(self._planner.joint_names)

        candidate_offsets = [
            (0.0, 0.55),
            (0.0, 0.40),
            (0.0, 0.70),
            (0.10, 0.55),
        ]
        for i, (x, z) in enumerate(candidate_offsets):
            pose = torch.eye(4, dtype=torch.float32)
            pose[0, 3] = x
            pose[2, 3] = z
            goal = _mat_to_goal_pose(
                pose, self._planner.tool_frames, self._device
            )
            ik_result = self._planner.ik_solver.solve_pose(goal)
            if (
                ik_result is None
                or ik_result.solution is None
                or ik_result.success is None
            ):
                continue
            success = ik_result.success.reshape(-1).bool()
            if not bool(success.any().item()):
                continue
            solutions = ik_result.solution.reshape(-1, n_actuated)
            best_idx = int(success.nonzero(as_tuple=False)[0].item())
            print(
                f"[Info] Self/world collision-free start config found "
                f"(candidate {i + 1}/{len(candidate_offsets)})."
            )
            return solutions[best_idx].cpu().to(dtype)

        print(
            "[Warning] All IK candidates failed — falling back to zero start configuration."
        )
        return torch.zeros(n_actuated, dtype=dtype)

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
        link_poses_world = forward_kinematics(mdh, theta).numpy()

        results = []
        for s in self._sphere_dict.get("base_link", []):
            results.append([*s["center"], s["radius"]])
        for j in range(n):
            T = link_poses_world[j]
            for s in self._sphere_dict.get(f"link_{j}", []):
                c = np.array(s["center"] + [1.0])
                p = T @ c
                results.append([p[0], p[1], p[2], s["radius"]])

        return (
            np.array(results, dtype=np.float32)
            if results
            else np.zeros((0, 4), dtype=np.float32)
        )

    # ----------------------------------------------------------------------
    # Plan execution
    # ----------------------------------------------------------------------

    def _plan_pose_raw(
        self,
        goal_pose: torch.Tensor,
        start_q: torch.Tensor | None,
        max_attempts: int,
    ):
        """Run a single cuRobo plan_pose call and return (raw_result, goal, start_q_used).

        Diagnostics are left to callers so each call site can attach its own context
        (e.g. the goal index inside a sequence).
        """
        if start_q is None:
            start_q = self.default_start_q()

        goal = _mat_to_goal_pose(
            goal_pose.to(self._dtype), self._planner.tool_frames, self._device
        )

        start_state = JointState.from_position(
            start_q.float().unsqueeze(0).to(self._device),
            joint_names=self._planner.joint_names,
        )

        # Suppress cuRobo's internal stdout chatter; we reconstruct the failure cause ourselves.
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = self._planner.plan_pose(
                    goal, start_state, max_attempts=max_attempts
                )
        except Exception:
            traceback.print_exc()
            raise

        return result, goal, start_q

    def _result_to_path(self, result) -> list[torch.Tensor]:
        n_joints = len(self._planner.joint_names)
        interp = result.get_interpolated_plan()
        positions = interp.position.cpu().to(self._dtype).reshape(-1, n_joints)
        return [positions[t] for t in range(positions.shape[0])]

    # ----------------------------------------------------------------------
    # Geometry & joint-limit helpers
    # ----------------------------------------------------------------------

    def _joint_limit_violations(self, q: torch.Tensor) -> "tuple[int, float] | None":
        """Return (#joints out of bounds, worst overshoot in radians), or None on error."""
        try:
            lims = self._planner.kinematics.get_joint_limits().position
            q_dev = q.to(lims.device).reshape(-1)
            lo, hi = lims[0], lims[1]
            below = (lo - q_dev).clamp(min=0.0)
            above = (q_dev - hi).clamp(min=0.0)
            worst = float(torch.maximum(below, above).max().item())
            n_bad = int(((below + above) > 0).sum().item())
            return n_bad, worst
        except Exception:
            return None

    def _joint_limit_detail(self, q: torch.Tensor) -> str:
        """Per-joint breakdown of out-of-bounds joints for q.

        Example: ``j2 q=+1.234 ∉ [-1.000, +1.000] (over by 0.234)``. Empty when none.
        """
        try:
            lims = self._planner.kinematics.get_joint_limits().position
            lo = lims[0].detach().cpu().tolist()
            hi = lims[1].detach().cpu().tolist()
            q_list = q.detach().cpu().reshape(-1).tolist()
        except Exception:
            return ""
        parts = []
        for i, (qi, loi, hii) in enumerate(zip(q_list, lo, hi)):
            if qi < loi:
                parts.append(
                    f"j{i} q={qi:+.3f} ∉ [{loi:+.3f}, {hii:+.3f}] (under by {loi - qi:.3f})"
                )
            elif qi > hii:
                parts.append(
                    f"j{i} q={qi:+.3f} ∉ [{loi:+.3f}, {hii:+.3f}] (over by {qi - hii:.3f})"
                )
        return ", ".join(parts)

    def _collision_report(
        self, robot_spheres: torch.Tensor
    ) -> "tuple[tuple[float, str, str] | None, tuple[float, str, str, float] | None]":
        """Inspect collision-sphere positions for world- and self-overlaps.

        Sphere positions are in the shared world/base frame.
        Returns (worst_world, worst_self), each ``None`` if no overlap; otherwise
        ``(penetration_m, cuboid_or_link_a, link_or_link_b[, center_dist_m])``.
        """
        spheres = robot_spheres.detach().reshape(-1, 4).cpu().numpy()
        link_names = list(self._sphere_dict.keys())
        sphere_link = []
        for ln in link_names:
            sphere_link.extend([ln] * len(self._sphere_dict[ln]))
        keep = spheres[:, 3] > 0
        spheres = spheres[keep]
        sphere_link = [sphere_link[i] for i, k in enumerate(keep) if k]

        cuboids = (
            list(getattr(self.scene, "cuboid", None) or [])
            if self.scene is not None
            else []
        )

        worst_world = None
        for cub in cuboids:
            px, py, pz, qw, qx, qy, qz = cub.pose
            R_cw = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            R_wc = R_cw.T
            half = np.array(cub.dims, dtype=np.float64) * 0.5
            for s, ln in zip(spheres, sphere_link):
                p_local = R_wc @ (s[:3].astype(np.float64) - np.array([px, py, pz]))
                clamped = np.clip(p_local, -half, half)
                dist = float(np.linalg.norm(p_local - clamped))
                penetration = float(s[3]) - dist
                if penetration > 0 and (
                    worst_world is None or penetration > worst_world[0]
                ):
                    worst_world = (penetration, cub.name, ln)

        worst_self = None
        for i in range(len(spheres)):
            li = sphere_link[i]
            ign_i = set(self._self_collision_ignore.get(li, [])) | {li}
            for j in range(i + 1, len(spheres)):
                lj = sphere_link[j]
                if lj in ign_i:
                    continue
                d = float(np.linalg.norm(spheres[i, :3] - spheres[j, :3]))
                pen = (spheres[i, 3] + spheres[j, 3]) - d
                if pen > 0 and (worst_self is None or pen > worst_self[0]):
                    worst_self = (pen, li, lj, d)

        return worst_world, worst_self

    def _classify_q(self, q: torch.Tensor) -> str:
        """One-line reason why joint config q is infeasible against the actual scene.

        Checks joint limits, world-collision, and self-collision. Returns ``"feasible"``
        if no violation is found.
        """
        reasons: list[str] = []
        jl_detail = self._joint_limit_detail(q)
        if jl_detail:
            reasons.append(f"joint limits: {jl_detail}")
        try:
            js = JointState.from_position(
                q.float().reshape(1, -1).to(self._device),
                joint_names=self._planner.joint_names,
            )
            kin = self._planner.compute_kinematics(js)
            if kin.robot_spheres is not None:
                world, slf = self._collision_report(kin.robot_spheres)
                if world is not None:
                    pen, name, ln = world
                    reasons.append(f"world collision ({name} ↔ {ln}, pen {pen:.3f} m)")
                if slf is not None:
                    pen, li, lj, _ = slf
                    reasons.append(f"self-collision ({li} ↔ {lj}, pen {pen:.3f} m)")
        except Exception:
            pass
        return "; ".join(reasons) if reasons else "feasible"

    def _world_collision_free_seeds(
        self, solutions: torch.Tensor
    ) -> "torch.Tensor | None":
        """Boolean mask over seeds, True where the config is free of world (environment)
        collision.

        Returns ``None`` when no collision geometry is in play (obstacles ignored or
        kinematics unavailable) so callers can skip constraint filtering entirely.
        """
        if self.scene is None:
            return None
        try:
            js = JointState.from_position(
                solutions.float().to(self._device),
                joint_names=self._planner.joint_names,
            )
            kin = self._planner.compute_kinematics(js)
            robot_spheres = getattr(kin, "robot_spheres", None)
            if robot_spheres is None:
                return None
            n = solutions.shape[0]
            spheres_per_seed = robot_spheres.reshape(n, -1, 4)
            free = torch.ones(n, dtype=torch.bool)
            for k in range(n):
                world, _ = self._collision_report(spheres_per_seed[k])
                free[k] = world is None
            return free
        except Exception:
            return None

    # ----------------------------------------------------------------------
    # Pose / IK helpers
    # ----------------------------------------------------------------------

    def _actual_pose_error_at_traj_end(
        self, result, goal
    ) -> "tuple[float, float] | tuple[None, None]":
        """True EE-to-goal residual at the trajectory's final joint state, in (m, rad).

        cuRobo's ``result.position_error``/``rotation_error`` are hinged weighted-square
        cost values — when a topk seed lands inside the convergence tolerance the hinge
        fires and they read 0 even if the EE is visibly far from the goal. This re-FKs
        the trajectory's last knot to get the actual Euclidean position and quaternion
        angular distance.
        """
        try:
            js_pos = result.js_solution.position  # (B, topk, H, dof_full)
            last_q = js_pos[..., -1, :].reshape(-1, js_pos.shape[-1])
            n_actuated = len(self._planner.joint_names)
            if last_q.shape[-1] != n_actuated:
                last_q = last_q[:, :n_actuated]
            js = JointState.from_position(
                last_q.float().to(self._device),
                joint_names=self._planner.joint_names,
            )
            kin = self._planner.compute_kinematics(js)
            tp = kin.tool_poses
            ee_pos = tp.position.reshape(-1, 3)
            ee_quat = tp.quaternion.reshape(-1, 4)
            gp = goal.position.reshape(-1, 3)[0].to(ee_pos.device)
            gq = goal.quaternion.reshape(-1, 4)[0].to(ee_quat.device)
            d_pos = torch.linalg.norm(ee_pos - gp, dim=-1)
            dot = torch.abs((ee_quat * gq).sum(dim=-1)).clamp(max=1.0)
            d_rot = 2.0 * torch.acos(dot)
            return float(d_pos.min().item()), float(d_rot.min().item())
        except Exception:
            return None, None

    def _get_best_ik_q(
        self, goal, current_q: "torch.Tensor | None" = None
    ) -> "torch.Tensor | None":
        """Return the IK joint config closest to goal in pose error, even if IK didn't converge.

        Seeded from ``current_q`` when provided so the returned config is anchored near
        the trajectory's start state. Without current_state seeding cuRobo's IK runs from
        its default seeds and may return an arbitrary unconverged solution that looks
        unrelated to the goal in the viewer.
        """
        try:
            current_state = None
            if current_q is not None:
                current_state = JointState.from_position(
                    current_q.float().reshape(1, -1).to(self._device),
                    joint_names=self._planner.joint_names,
                )
            num_seeds = self._planner.ik_solver.config.num_seeds
            ik_result = self._planner.ik_solver.solve_pose(
                goal,
                current_state=current_state,
                return_seeds=num_seeds,
            )
            if ik_result is None or ik_result.solution is None:
                return None
            n_joints = len(self._planner.joint_names)
            solutions = ik_result.solution.reshape(-1, n_joints)
            pos_err = (
                ik_result.position_error.reshape(-1)
                if ik_result.position_error is not None
                else None
            )
            rot_err = (
                ik_result.rotation_error.reshape(-1)
                if ik_result.rotation_error is not None
                else torch.zeros_like(pos_err)
                if pos_err is not None
                else None
            )
            if pos_err is not None:
                score = pos_err + rot_err
                best = int(score.argmin().item())
            else:
                best = 0
            return solutions[best].cpu().to(self._dtype)
        except Exception:
            return None

    # ----------------------------------------------------------------------
    # Failure diagnostics
    # ----------------------------------------------------------------------

    def _diagnose_failure(
        self,
        result,
        goal,
        start_q: torch.Tensor,
        report_header: str | None = None,
    ) -> "torch.Tensor | None":
        """Print a 2-3 line failure summary categorized as joint limits / IK / collision /
        TrajOpt / trajectory.

        Returns the IK config to visualize for this failure. For IK failures this is the
        exact seed whose classification was printed, so the static debug scene and the
        printed reason always describe the same joint config (and the same world-collision
        verdict). For other failure types it falls back to the closest IK seed to the goal.
        """
        prefix = f"[cuRobo] {report_header}: " if report_header else "[cuRobo] failed: "

        jl = self._joint_limit_violations(start_q)
        if jl is not None and jl[0] > 0:
            n_bad, worst = jl
            detail = self._joint_limit_detail(start_q)
            print(f"{prefix}joint limits")
            print(
                f"[cuRobo]   start_q violates {n_bad} joints (worst overshoot {worst:.3f} rad)"
            )
            if detail:
                print(f"[cuRobo]   {detail}")
            return self._get_best_ik_q(goal, current_q=start_q)

        if result is None:
            try:
                return self._diagnose_ik_failure(prefix, goal, start_q)
            except Exception:
                print(f"{prefix}IK")
                print("[cuRobo]   IK diagnostics failed (internal error)")
                return None

        try:
            self._diagnose_trajopt_failure(prefix, result, goal)
        except Exception:
            print(f"{prefix}trajectory")
            print("[cuRobo]   TrajOpt diagnostics failed (internal error)")
        return self._get_best_ik_q(goal, current_q=start_q)

    def _diagnose_ik_failure(
        self, prefix: str, goal, start_q: torch.Tensor
    ) -> "torch.Tensor | None":
        """plan_pose returned None — re-run IK, classify the seeds, and return the one to render.

        A single seed (the one closest to the goal in pos+rot error) is chosen as the
        "best seed". Its classification is what gets printed, and the same config is
        returned so the static debug scene shows exactly the pose the message describes.
        """
        try:
            num_seeds = self._planner.trajopt_solver.config.num_seeds
            start_state = JointState.from_position(
                start_q.float().unsqueeze(0).to(self._device),
                joint_names=self._planner.joint_names,
            )
            ik = self._planner.ik_solver.solve_pose(
                goal,
                current_state=start_state,
                return_seeds=num_seeds,
            )
        except Exception:
            ik = None
        if ik is None or ik.position_error is None:
            print(f"{prefix}IK")
            print(
                "[cuRobo]   IK did not converge for any seed (no diagnostic available)"
            )
            return None

        ik_cfg = self._planner.ik_solver.config
        pos_tol = getattr(ik_cfg, "position_tolerance", ik.position_tolerance)
        rot_tol = getattr(ik_cfg, "orientation_tolerance", ik.orientation_tolerance)
        pos_per_seed = ik.position_error.reshape(-1)
        rot_per_seed = (
            ik.rotation_error.reshape(-1)
            if ik.rotation_error is not None
            else torch.full_like(pos_per_seed, float("nan"))
        )
        n_returned = pos_per_seed.numel()
        pose_ok = (pos_per_seed <= pos_tol) & (rot_per_seed <= rot_tol)
        ik_success = ik.success.reshape(-1).bool() if ik.success is not None else None
        n_ik_success = int(ik_success.sum().item()) if ik_success is not None else 0
        n_joints = len(self._planner.joint_names)
        solutions = (
            ik.solution.reshape(n_returned, n_joints)
            if ik.solution is not None
            else None
        )

        # Single source of truth for both the printed "best seed" and the rendered config.
        # With environment collision active in planning, prefer the closest seed that
        # actually satisfies that constraint (world-collision-free); only fall back to the
        # closest colliding seed when no returned seed is collision-free. Score is combined
        # pos+rot error; nan rotations (no rot error reported) score on position alone.
        best_q = None
        best_idx = 0
        best_is_free = False
        if solutions is not None:
            score = pos_per_seed + torch.nan_to_num(rot_per_seed, nan=0.0)
            free = self._world_collision_free_seeds(solutions)
            if free is not None and bool(free.any().item()):
                masked = score.clone()
                masked[~free.to(score.device)] = float("inf")
                best_idx = int(masked.argmin().item())
                best_is_free = True
            else:
                best_idx = int(score.argmin().item())
            best_q = solutions[best_idx].cpu().to(self._dtype)

        # IK had valid solutions but plan_pose still returned None → graph seeding rejected them.
        if n_ik_success > 0 and solutions is not None:
            self._diagnose_graph_rejection(prefix, solutions, start_q, n_returned)
            return best_q

        n_pose_ok = int(pose_ok.sum().item())
        print(f"{prefix}IK")
        if n_pose_ok == 0:
            if best_q is not None:
                best_pos = float(pos_per_seed[best_idx].item())
                best_rot = float(rot_per_seed[best_idx].item())
                qual = "collision-free " if best_is_free else ""
                print(
                    f"[cuRobo]   pose unreachable (best {qual}seed {best_pos * 1000:.1f}mm/"
                    f"{best_rot:.3f}rad, tol {pos_tol * 1000:.1f}mm/{rot_tol:.3f}rad)"
                )
                print(f"[cuRobo]   best seed: {self._classify_q(solutions[best_idx])}")
            else:
                best_pos = float(pos_per_seed.min().item())
                best_rot = float(rot_per_seed.min().item())
                print(
                    f"[cuRobo]   pose unreachable (best {best_pos * 1000:.1f}mm/{best_rot:.3f}rad, "
                    f"tol {pos_tol * 1000:.1f}mm/{rot_tol:.3f}rad)"
                )
        else:
            pose_ok_idx = pose_ok.nonzero(as_tuple=False).reshape(-1).tolist()
            summary = (
                self._summarize_seed_reasons(solutions, pose_ok_idx)
                if solutions is not None
                else "n/a"
            )
            print(
                f"[cuRobo]   pose reachable but all {n_pose_ok}/{n_returned} solutions infeasible: {summary}"
            )
        return best_q

    def _diagnose_graph_rejection(
        self,
        prefix: str,
        solutions: torch.Tensor,
        start_q: torch.Tensor,
        n_returned: int,
    ) -> None:
        """IK produced valid seeds but graph-seeding rejected them — classify as collision."""
        gp = self._planner.graph_planner
        if gp is None:
            print(f"{prefix}collision")
            print(
                "[cuRobo]   graph planner rejected plan (no graph planner for diagnosis)"
            )
            return

        sols_dev = solutions.to(self._device).float()
        feas_graph = gp.check_samples_feasibility(sols_dev).reshape(-1).bool()
        start_dev = start_q.float().reshape(1, -1).to(self._device)
        start_ok = bool(gp.check_samples_feasibility(start_dev).all().item())
        bad = (~feas_graph).nonzero(as_tuple=False).reshape(-1).tolist()

        print(f"{prefix}collision")
        if not start_ok:
            print(f"[cuRobo]   start state in collision: {self._classify_q(start_q)}")
        elif bad:
            summary = self._summarize_seed_reasons(solutions, bad)
            print(
                f"[cuRobo]   graph planner rejected {len(bad)}/{n_returned} IK goals: {summary}"
            )
        else:
            print("[cuRobo]   graph planner rejected plan (cause unclear on re-check)")

    def _diagnose_trajopt_failure(self, prefix: str, result, goal) -> None:
        """TrajOpt ran but failed. Distinguish:
          * pose unreached at trajectory end (``TrajOpt``)
          * pose reached but a waypoint along the returned path violates a constraint (``trajectory``)
          * pose reached and waypoints clean → cuRobo's internal infeasibility was about
            dynamics (velocity/acceleration/jerk) on the smoothed trajectory.

        Uses actual EE-to-goal distance (re-FK'd) rather than cuRobo's hinged metric, and
        scans the returned interpolated trajectory ourselves since cuRobo clears the
        per-constraint metrics during topk reduction.
        """
        pos_err, rot_err = self._actual_pose_error_at_traj_end(result, goal)
        if pos_err is None or rot_err is None:
            pos_err = (
                float(result.position_error.min().item())
                if result.position_error is not None
                else float("nan")
            )
            rot_err = (
                float(result.rotation_error.min().item())
                if result.rotation_error is not None
                else float("nan")
            )
        to_cfg = self._planner.trajopt_solver.config
        pos_tol = getattr(to_cfg, "position_tolerance", result.position_tolerance)
        rot_tol = getattr(to_cfg, "orientation_tolerance", result.orientation_tolerance)

        if pos_err > pos_tol or rot_err > rot_tol:
            print(f"{prefix}TrajOpt")
            print(
                f"[cuRobo]   pose not reached (trajectory ends {pos_err * 1000:.1f}mm/"
                f"{rot_err:.3f}rad from goal, tol {pos_tol * 1000:.1f}mm/{rot_tol:.3f}rad)"
            )
            return

        # Pose was reached → at least one waypoint must violate something.
        print(f"{prefix}trajectory")
        idx, n_total, reason = self._first_path_violation(result)
        pose_str = f"pose OK ({pos_err * 1000:.1f}mm/{rot_err:.3f}rad)"
        if reason is not None:
            print(f"[cuRobo]   {pose_str}, waypoint {idx}/{n_total}: {reason}")
        else:
            print(
                f"[cuRobo]   {pose_str}, returned path clean — cuRobo's failure flag is "
                "from dynamics (velocity/acceleration/jerk over the smoothed trajectory)"
            )

    def _first_path_violation(
        self, result
    ) -> "tuple[int, int, str] | tuple[None, int, None]":
        """Scan the returned interpolated trajectory and return the first violating waypoint.

        Returns ``(idx, n_total, reason)`` for the first waypoint where a joint limit,
        world collision, or self-collision is violated. Returns ``(None, n_total, None)``
        if the entire returned path is clean.
        """
        traj = getattr(result, "interpolated_trajectory", None)
        positions = traj.position if traj is not None else None
        if positions is None:
            return None, 0, None
        waypoints = positions.reshape(-1, positions.shape[-1])
        n_actuated = len(self._planner.joint_names)
        if waypoints.shape[-1] != n_actuated:
            waypoints = waypoints[:, :n_actuated]
        n_total = int(waypoints.shape[0])
        if n_total == 0:
            return None, 0, None

        js = JointState.from_position(
            waypoints.float().to(self._device),
            joint_names=self._planner.joint_names,
        )
        kin = self._planner.compute_kinematics(js)
        robot_spheres = getattr(kin, "robot_spheres", None)
        spheres_per_wp = (
            robot_spheres.reshape(n_total, -1, 4) if robot_spheres is not None else None
        )

        lims = self._planner.kinematics.get_joint_limits().position
        w_dev = waypoints.to(lims.device)
        lo, hi = lims[0], lims[1]
        below = (lo - w_dev).clamp(min=0.0)
        above = (w_dev - hi).clamp(min=0.0)
        jl_per_wp = ((below + above).sum(dim=-1) > 0).cpu().tolist()

        for k in range(n_total):
            if jl_per_wp[k]:
                return (
                    k,
                    n_total,
                    f"joint limits: {self._joint_limit_detail(waypoints[k])}",
                )
            if spheres_per_wp is None:
                continue
            world, slf = self._collision_report(spheres_per_wp[k])
            if world is not None:
                pen, name, ln = world
                return k, n_total, f"world collision ({name} ↔ {ln}, pen {pen:.3f} m)"
            if slf is not None:
                pen, li, lj, _ = slf
                return k, n_total, f"self-collision ({li} ↔ {lj}, pen {pen:.3f} m)"

        return None, n_total, None

    def _summarize_seed_reasons(self, solutions: torch.Tensor, indices) -> str:
        """Classify each seed in ``indices`` and group identical reasons with a count."""
        reasons: dict[str, int] = {}
        for k in indices:
            r = self._classify_q(solutions[k])
            reasons[r] = reasons.get(r, 0) + 1
        return ", ".join(f"{n}× {r}" for r, n in reasons.items())


def interpolate_path(
    path: list[torch.Tensor], step: float = 0.02
) -> list[torch.Tensor]:
    """Densify a joint-space path so consecutive frames are at most ``step`` apart."""
    if len(path) < 2:
        return path
    out = [path[0]]
    for q_a, q_b in zip(path[:-1], path[1:]):
        delta = q_b - q_a
        n = max(1, int(torch.ceil(delta.norm() / step).item()))
        for k in range(1, n + 1):
            out.append(q_a + (k / n) * delta)
    return out
