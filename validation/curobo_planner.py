import contextlib
import io
import os
import tempfile

import torch
import numpy as np
from scipy.spatial.transform import Rotation

from interface import Morphology, Task
from interface.plan_result import PlanResult
from util.mdh import to_urdf
from util.kinematics import (
    build_sphere_dict,
    build_self_collision_ignore,
    build_scene,
    mat_to_goal_pose,
)
from task.morphology_sampler import geometric_jacobian, yoshikawa_manipulability
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.kinematics import Kinematics
from curobo.types import JointState


class CuroboPlanner:
    """cuRobo MotionPlanner wrapper for a fixed (Morphology, Task) pair."""

    # === Construction ===

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
        """Build a cuRobo MotionPlanner for a fixed (Morphology, Task) pair.

        Writes the morphology to a temporary URDF, constructs collision spheres and
        the world scene, then warms up the underlying ``MotionPlanner``.

        Args:
            morph: Robot morphology (MDH params, link count, dtype).
            task: Task description used to build the world collision scene.
            device: Torch device for cuRobo tensors (typically a CUDA device).
            num_ik_seeds: IK seeds per pose solve.
            num_trajopt_seeds: TrajOpt seeds per plan.
            ignore_ground: If True, omit the ground plane from the scene.
            ignore_obstacles: If True, plan without any world collision geometry.
        """
        self._device = device
        self._morph = morph
        self._ignore_ground = ignore_ground
        self._ignore_obstacles = ignore_obstacles
        self._dtype = morph.params.dtype

        with tempfile.NamedTemporaryFile(suffix=".urdf", delete=False, mode="w") as f:
            f.write(to_urdf(morph))
            self._urdf_path = f.name

        self._sphere_dict = build_sphere_dict(morph)
        sphere_links = set(self._sphere_dict)
        self._self_collision_ignore = {
            k: [v for v in vs if v in sphere_links]
            for k, vs in build_self_collision_ignore(morph).items()
            if k in sphere_links
        }

        robot_dict = {
            "kinematics": {
                "urdf_path": self._urdf_path,
                "base_link": "base_link",
                "tool_frames": [f"link_{morph.n_links - 1}"],
                "collision_spheres": self._sphere_dict,
                "collision_link_names": list(self._sphere_dict.keys()),
                "collision_sphere_buffer": 0.01,
                "self_collision_buffer": {k: 0.0 for k in self._sphere_dict},
                "self_collision_ignore": self._self_collision_ignore,
            }
        }

        self.scene = (
            None
            if ignore_obstacles
            else build_scene(
                task, ignore_ground=ignore_ground, ignore_obstacles=ignore_obstacles
            )
        )

        if self.scene is not None:
            # Sized exactly to the scene; self.scene is treated as immutable after
            # construction. If we ever add post-init scene mutation, add headroom here.
            n_cuboids = len(getattr(self.scene, "cuboid", None) or [])
            collision_cache = {"cuboid": max(n_cuboids, 1)}
        else:
            collision_cache = None
        planner_config = MotionPlannerCfg.create(
            robot=robot_dict,
            scene_model=self.scene,
            collision_cache=collision_cache,
            num_ik_seeds=num_ik_seeds,
            num_trajopt_seeds=num_trajopt_seeds,
        )
        self._planner = MotionPlanner(planner_config)
        self._planner.warmup(enable_graph=True, num_warmup_iterations=3)

        # Built lazily by tool_jacobian(): the planner's own kinematics has
        # compute_jacobian=False, so we reuse its config in a Jacobian-enabled
        # instance to read the EE Jacobian directly off cuRobo.
        self._jac_kin: Kinematics | None = None

    def __del__(self):
        path = getattr(self, "_urdf_path", None)
        if path is None:
            return
        try:
            os.unlink(path)
        except Exception:
            pass

    # === Public API — planning ===

    def plan_sequence(
        self,
        goal_poses: torch.Tensor,
        start_q: torch.Tensor | None = None,
        max_attempts: int = 5,
    ) -> tuple[PlanResult, torch.Tensor | None]:
        """Plan a chained trajectory through a sequence of EE goal poses.

        Plans ``start_q → goal[0] → goal[1] → … → goal[N-1]`` by feeding each plan's
        final joint state as the next plan's start. On failure prints a diagnostic
        and returns immediately with the partial path.

        Args:
            goal_poses: Goal EE poses as ``(N, 4, 4)`` homogeneous transforms.
            start_q: Initial joint config; if None, sampled via :meth:`default_start_q`.
            max_attempts: Max cuRobo plan_pose retries per goal.

        Returns:
            ``(PlanResult, final_q)``. On success ``final_q`` is the last joint
            config of the chained path; on failure it is ``None`` and the
            ``PlanResult`` contains the partial path, ``failed_at_goal``, and the
            closest IK config for visualization.
        """
        n_total = goal_poses.shape[0]
        full_path: list[torch.Tensor] = []
        current_q = start_q

        for i in range(n_total):
            result, goal, start_q_used = self._plan_pose(
                goal_poses[i], current_q, max_attempts
            )
            if result is None or result.success is None or not result.success.any():
                reachable_ratio = i / n_total if n_total > 0 else 0.0
                best_ik_q = self._diagnose_failure(
                    result,
                    goal,
                    start_q_used,
                    report_header=(
                        f"Goal {i}/{n_total} failed "
                        f"({reachable_ratio * 100:.1f}% reached)"
                    ),
                    goal_label=f"goal {i}",
                )
                return PlanResult(
                    success=False,
                    path=full_path,
                    n_iterations=0,
                    n_nodes=len(full_path),
                    failed_at_goal=i,
                    best_ik_q=best_ik_q,
                    reachable_ratio=reachable_ratio,
                ), None

            path, current_q = self._result_to_path(result)
            full_path.extend(path)

        print(f"[cuRobo]   reachable goals: {n_total}/{n_total} (100.0%)")
        return PlanResult(
            success=True,
            path=full_path,
            n_iterations=1,
            n_nodes=len(full_path),
            reachable_ratio=1.0,
        ), current_q

    def sequence_reachability_ratio(
        self,
        goal_poses: torch.Tensor,
        start_q: torch.Tensor | None = None,
        max_attempts: int = 5,
    ) -> float:
        """Plan the sequence and return only the reachable-goal fraction.

        Args:
            goal_poses: Goal EE poses as ``(N, 4, 4)`` homogeneous transforms.
            start_q: Initial joint config; if None, sampled via :meth:`default_start_q`.
            max_attempts: Max cuRobo plan_pose retries per goal.

        Returns:
            Fraction in ``[0, 1]``: number of goals reached over total goals.
        """
        result, _ = self.plan_sequence(goal_poses, start_q, max_attempts)
        return result.reachable_ratio

    # === Public API — feasibility & utilities ===
   
    def tool_jacobian(self, qs: torch.Tensor) -> torch.Tensor:
        """End-effector geometric Jacobian (base frame) for a batch of joint configs.

        Reads the Jacobian straight off cuRobo's kinematics. The planner builds its
        own kinematics with ``compute_jacobian=False``, so on first call we create a
        Jacobian-enabled ``Kinematics`` from the same config and cache it.

        Args:
            qs: Joint configurations, shape ``[N, dof]`` (dof = len(joint_names)).

        Returns:
            Jacobians of shape ``[N, 6, dof]``; rows 0:3 are linear, 3:6 angular —
            the layout ``yoshikawa_manipulability`` expects.
        """
        if self._jac_kin is None:
            self._jac_kin = Kinematics(
                self._planner.kinematics.config, compute_jacobian=True
            )
        js = JointState.from_position(
            qs.float().reshape(-1, len(self._planner.joint_names)).to(self._device),
            joint_names=self._planner.joint_names,
        )
        state = self._jac_kin.compute_kinematics(js)
        # tool_jacobians: [batch, horizon=1, n_tool_frames, 6, dof]; tool frame 0 is the EE.
        return state.tool_jacobians[:, 0, 0].to(self._dtype)

    def is_q_feasible(self, q: torch.Tensor) -> bool:
        """Check whether a single joint config satisfies all planner constraints.

        Args:
            q: Joint config tensor, shape ``(dof,)``.

        Returns:
            True if ``q`` is collision-free (self + world) and within joint limits.
        """
        q_check = q.float().unsqueeze(0).to(self._device)
        return bool(
            self._planner.graph_planner.check_samples_feasibility(q_check).all()
        )

    def default_start_q(self, n_samples: int = 4096, seed: int = 0) -> torch.Tensor:
        """Joint-space rejection sampling for a collision-free start config.

        Draws ``n_samples`` uniform configs inside the joint-limit box and returns
        the first one cuRobo flags as collision-free (self + world) and within
        limits.

        Args:
            n_samples: Number of uniform samples to draw and feasibility-check
                in a single batch.
            seed: RNG seed for reproducible sampling.

        Returns:
            Feasible joint config tensor of shape ``(dof,)`` on CPU in the
            morphology dtype.

        Raises:
            RuntimeError: No feasible config found in ``n_samples`` draws (scene
                over-constrained or robot in persistent self-collision).
        """
        lims = self._planner.kinematics.get_joint_limits().position
        lo, hi = lims[0], lims[1]
        dof = lo.shape[0]

        gen = torch.Generator(device=lo.device).manual_seed(seed)
        q_unit = torch.rand(
            n_samples, dof, generator=gen, device=lo.device, dtype=lo.dtype
        )
        candidates = lo + (hi - lo) * q_unit

        feasible = (
            self._planner.graph_planner.check_samples_feasibility(candidates)
            .reshape(-1)
            .bool()
        )
        feasible_idx = feasible.nonzero(as_tuple=False)

        if feasible_idx.numel() == 0:
            raise RuntimeError(
                f"No feasible start config found in {n_samples} joint-space samples "
                "(scene may be over-constrained or robot may be in persistent self-collision)."
            )

        i = int(feasible_idx[0].item())
        print(f"[Info] Feasible start config found (sample {i + 1}/{n_samples}).")
        return candidates[i].detach().cpu().to(self._dtype)

    def robot_spheres_world(self, q: torch.Tensor) -> np.ndarray:
        """Forward-kinematics the collision spheres for ``q`` into the world frame.

        Args:
            q: Joint config tensor, shape ``(dof,)``.

        Returns:
            ``(N, 4)`` float32 array of ``(x, y, z, radius)`` for every collision
            sphere on the robot, expressed in the world frame. Empty
            ``(0, 4)`` array when no spheres exist.
        """
        js = JointState.from_position(
            q.float().reshape(1, -1).to(self._device),
            joint_names=self._planner.joint_names,
        )
        spheres = self._planner.compute_kinematics(js).robot_spheres
        if spheres is None:
            return np.zeros((0, 4), dtype=np.float32)
        return spheres.reshape(-1, 4).cpu().numpy().astype(np.float32)

    # === Internal — single pose plan ===
    

    # ----------------------------------------------------------------------
    # Plan execution
    # ----------------------------------------------------------------------

    def _plan_pose(self, goal_pose, start_q, max_attempts):
        """Run a single cuRobo ``plan_pose`` call.

        cuRobo's stdout is suppressed; failure is diagnosed by callers.

        Args:
            goal_pose: Goal EE pose as a ``(4, 4)`` homogeneous transform.
            start_q: Initial joint config tensor ``(dof,)`` or None (sampled then).
            max_attempts: Max plan_pose retries.

        Returns:
            ``(result, goal, start_q_used)``: cuRobo's raw ``MotionPlanResult``,
            the converted goal pose object, and the actual start config used.
        """
        if start_q is None:
            start_q = self.default_start_q()

        goal = mat_to_goal_pose(
            goal_pose.to(self._dtype), self._planner.tool_frames, self._device
        )
        start_q_dev = start_q.float().unsqueeze(0)
        if start_q_dev.device != self._device:
            start_q_dev = start_q_dev.to(self._device)
        start_state = JointState.from_position(
            start_q_dev,
            joint_names=self._planner.joint_names,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            result = self._planner.plan_pose(
                goal, start_state, max_attempts=max_attempts
            )

        return result, goal, start_q

    def _result_to_path(self, result) -> "tuple[list[torch.Tensor], torch.Tensor]":
        """Extract the interpolated trajectory and the final joint state.

        Args:
            result: cuRobo ``MotionPlanResult`` from a successful plan.

        Returns:
            ``(waypoints, last_q)`` where ``waypoints`` is a list of joint-config
            tensors on CPU in the morphology dtype (one per interpolated waypoint,
            each shaped ``(dof,)``), and ``last_q`` is the final waypoint kept on
            the planner device for re-use as the next plan's start (avoids a
            CPU→GPU round-trip in chained sequence planning).
        """
        n_joints = len(self._planner.joint_names)
        positions_gpu = result.get_interpolated_plan().position.reshape(-1, n_joints)
        last_q = positions_gpu[-1].clone()
        positions = positions_gpu.cpu().to(self._dtype)
        return list(positions.unbind(0)), last_q

    # === Internal — feasibility checks ===

    def _joint_limit_status(self, q: torch.Tensor) -> "tuple[int, float, str]":
        """Check joint limits and report count, worst overshoot, and per-joint detail.

        Args:
            q: Joint config tensor, shape ``(dof,)``.

        Returns:
            ``(n_violating_joints, worst_overshoot_rad, detail_str)`` where
            ``detail_str`` is a comma-separated breakdown like
            ``"j2 q=+1.234 ∉ [-1.000, +1.000] (over by 0.234)"``. Returns
            ``(0, 0.0, "")`` when no joint violates its limits or limits couldn't
            be read.
        """
        try:
            lims = self._planner.kinematics.get_joint_limits().position
            lo = lims[0].detach().cpu().tolist()
            hi = lims[1].detach().cpu().tolist()
            q_list = q.detach().cpu().reshape(-1).tolist()
        except Exception:
            return 0, 0.0, ""
        parts: list[str] = []
        worst = 0.0
        n_bad = 0
        for i, (qi, loi, hii) in enumerate(zip(q_list, lo, hi)):
            if qi < loi:
                over = loi - qi
                parts.append(
                    f"j{i} q={qi:+.3f} ∉ [{loi:+.3f}, {hii:+.3f}] (under by {over:.3f})"
                )
            elif qi > hii:
                over = qi - hii
                parts.append(
                    f"j{i} q={qi:+.3f} ∉ [{loi:+.3f}, {hii:+.3f}] (over by {over:.3f})"
                )
            else:
                continue
            n_bad += 1
            if over > worst:
                worst = over
        return n_bad, worst, ", ".join(parts)

    def _collision_report(self, robot_spheres):
        """Find the worst world and self collisions for one config's spheres.

        Args:
            robot_spheres: Sphere tensor in world frame, shape ``(N, 4)`` as
                ``(x, y, z, radius)``. Zero-radius rows are skipped.

        Returns:
            ``(worst_world, worst_self)`` where each entry is ``None`` if no
            overlap exists. Otherwise:
              * ``worst_world`` = ``(penetration_m, cuboid_name, link_name)``
              * ``worst_self``  = ``(penetration_m, link_a, link_b, center_dist_m)``
        """
        spheres = robot_spheres.detach().reshape(-1, 4).cpu().numpy()
        link_names = list(self._sphere_dict.keys())
        sphere_link = [ln for ln in link_names for _ in self._sphere_dict[ln]]
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
            R_wc = Rotation.from_quat([qx, qy, qz, qw]).as_matrix().T
            half = np.array(cub.dims, dtype=np.float64) * 0.5
            for s, ln in zip(spheres, sphere_link):
                p_local = R_wc @ (s[:3].astype(np.float64) - np.array([px, py, pz]))
                dist = float(np.linalg.norm(p_local - np.clip(p_local, -half, half)))
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
        """Build a one-line human-readable reason ``q`` is infeasible.

        Checks joint limits, world collisions, and self-collisions against the
        current scene.

        Args:
            q: Joint config tensor, shape ``(dof,)``.

        Returns:
            Semicolon-joined reason string (e.g.
            ``"joint limits: j2 ...; world collision (table ↔ link_3, pen 0.012 m)"``),
            or ``"feasible"`` if no violation is found.
        """
        reasons: list[str] = []
        _, _, jl_detail = self._joint_limit_status(q)
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

    # === Internal — IK helpers ===

    def _pick_best_seed(
        self, solutions: torch.Tensor, score: torch.Tensor
    ) -> "tuple[int, bool]":
        """Pick the seed minimizing ``score``, preferring graph-feasible ones.

        Args:
            solutions: IK joint configs, shape ``(n_seeds, dof)``.
            score: Per-seed score tensor, shape ``(n_seeds,)``.

        Returns:
            ``(best_idx, was_feasible)``. ``was_feasible`` is True iff at least
            one feasible seed existed and the returned index is one of them.
        """
        try:
            feasible = (
                self._planner.graph_planner.check_samples_feasibility(
                    solutions.float().to(self._device)
                )
                .reshape(-1)
                .bool()
                .cpu()
            )
        except Exception:
            feasible = None
        if feasible is not None and bool(feasible.any().item()):
            masked = score.clone()
            masked[~feasible.to(score.device)] = float("inf")
            return int(masked.argmin().item()), True
        return int(score.argmin().item()), False

    def _get_best_ik_q(self, goal, current_q=None):
        """Solve IK for ``goal`` and return the seed closest in pose error.

        Returns a config even if no seed converged within IK tolerance, so the
        caller can visualize the planner's best guess. Seeding from ``current_q``
        keeps the returned config anchored near the trajectory start, instead of
        an arbitrary unrelated arm of the IK manifold.

        Args:
            goal: cuRobo goal pose object.
            current_q: Optional anchor for IK seeding, shape ``(dof,)``.

        Returns:
            Best IK joint config, shape ``(dof,)`` on CPU in the morphology
            dtype, or ``None`` if IK produced no usable solution.
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
            pos_err = ik_result.position_error
            if pos_err is None:
                return solutions[0].cpu().to(self._dtype)
            pos_err = pos_err.reshape(-1)
            rot_err = (
                ik_result.rotation_error.reshape(-1)
                if ik_result.rotation_error is not None
                else torch.zeros_like(pos_err)
            )
            score = pos_err + rot_err
            best, _ = self._pick_best_seed(solutions, score)
            return solutions[best].cpu().to(self._dtype)
        except Exception:
            return None

    def _actual_pose_error_at_traj_end(self, result, goal):
        """Re-FK the trajectory's last joint state and compute true EE-to-goal error.

        cuRobo's ``result.position_error`` / ``rotation_error`` are hinged
        weighted-square costs that read 0 once a topk seed lands inside the
        convergence tolerance — even when the EE is visibly far from the goal.
        This method bypasses that by recomputing the geometric residual directly.

        Args:
            result: cuRobo ``MotionPlanResult``.
            goal: cuRobo goal pose object with ``position`` and ``quaternion``.

        Returns:
            ``(pos_err_m, rot_err_rad)`` — Euclidean position error and quaternion
            angular distance over the topk last-knot configs (min). Returns
            ``(None, None)`` if recomputation fails.
        """
        try:
            js_pos = result.js_solution.position
            last_q = js_pos[..., -1, :].reshape(-1, js_pos.shape[-1])
            n_actuated = len(self._planner.joint_names)
            if last_q.shape[-1] != n_actuated:
                last_q = last_q[:, :n_actuated]
            js = JointState.from_position(
                last_q.float().to(self._device),
                joint_names=self._planner.joint_names,
            )
            kin = self._planner.compute_kinematics(js)
            ee_pos = kin.tool_poses.position.reshape(-1, 3)
            ee_quat = kin.tool_poses.quaternion.reshape(-1, 4)
            gp = goal.position.reshape(-1, 3)[0].to(ee_pos.device)
            gq = goal.quaternion.reshape(-1, 4)[0].to(ee_quat.device)
            d_pos = torch.linalg.norm(ee_pos - gp, dim=-1)
            dot = torch.abs((ee_quat * gq).sum(dim=-1)).clamp(min=0.0, max=1.0)
            d_rot = 2.0 * torch.acos(dot)
            return float(d_pos.min().item()), float(d_rot.min().item())
        except Exception:
            return None, None

    # === Internal — failure diagnostics ===

    @staticmethod
    def _log(prefix: str, category: str, *lines: str) -> None:
        """Print a diagnostic block: ``prefix+category`` then indented detail lines."""
        print(f"{prefix}{category}")
        for line in lines:
            print(f"[cuRobo]   {line}")

    def _diagnose_failure(
        self, result, goal, start_q, report_header=None, goal_label=None
    ):
        """Classify a planning failure and print a 2–3 line diagnostic summary.

        Categories: ``joint limits`` (start out of bounds), ``IK`` (no IK seed
        reached the goal), ``collision`` (IK reached the goal but graph seeding
        rejected the seeds), ``TrajOpt`` (trajectory ends short of the goal), or
        ``trajectory`` (goal reached but a waypoint violates a constraint).

        Args:
            result: cuRobo ``MotionPlanResult`` or ``None`` if plan returned None.
            goal: cuRobo goal pose object.
            start_q: The start config used for the plan, shape ``(dof,)``.
            report_header: Optional prefix label (e.g. ``"Goal 3/5 failed"``).
            goal_label: Optional short label (e.g. ``"goal 3"``) folded into the
                trajectory-violation body line so it self-identifies which goal's
                plan the waypoint belongs to.

        Returns:
            Best-effort IK joint config to render in the debug viewer, or ``None``
            if no IK config is available.
        """
        prefix = f"[cuRobo] {report_header}: " if report_header else "[cuRobo] failed: "

        n_bad, worst, detail = self._joint_limit_status(start_q)
        if n_bad > 0:
            lines = [
                f"start_q violates {n_bad} joints (worst overshoot {worst:.3f} rad)"
            ]
            if detail:
                lines.append(detail)
            self._log(prefix, "joint limits", *lines)
            return self._get_best_ik_q(goal, current_q=start_q)

        if result is None:
            try:
                return self._diagnose_ik_failure(prefix, goal, start_q)
            except Exception:
                self._log(prefix, "IK", "IK diagnostics failed (internal error)")
                return None

        try:
            self._diagnose_trajopt_failure(prefix, result, goal, goal_label=goal_label)
        except Exception:
            self._log(
                prefix, "trajectory", "TrajOpt diagnostics failed (internal error)"
            )
        return self._get_best_ik_q(goal, current_q=start_q)

    def _diagnose_ik_failure(self, prefix, goal, start_q):
        """Diagnose a failure where ``plan_pose`` returned None.

        Re-runs IK, picks a single "best seed" (closest in pos+rot error,
        preferring world-collision-free seeds when any exist), and prints its
        classification. The same config is returned so the static debug scene
        shows exactly the pose described in the printed message.

        Args:
            prefix: Header string already formatted for log lines.
            goal: cuRobo goal pose object.
            start_q: Start config used for the original plan, shape ``(dof,)``.

        Returns:
            Best IK config to visualize, shape ``(dof,)``, or ``None`` if IK
            yielded nothing usable.
        """
        try:
            num_seeds = self._planner.ik_solver.config.num_seeds
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
            self._log(
                prefix,
                "IK",
                "IK did not converge for any seed (no diagnostic available)",
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

        # Prefer closest feasible seed; else closest seed overall.
        best_q = None
        best_idx = 0
        best_is_feasible = False
        if solutions is not None:
            score = pos_per_seed + torch.nan_to_num(rot_per_seed, nan=0.0)
            best_idx, best_is_feasible = self._pick_best_seed(solutions, score)
            best_q = solutions[best_idx].cpu().to(self._dtype)

        # IK valid but plan_pose None → graph seeding rejected the seeds.
        if n_ik_success > 0 and solutions is not None:
            self._diagnose_graph_rejection(
                prefix, solutions, start_q, n_ik_success, ik_success
            )
            return best_q

        n_pose_ok = int(pose_ok.sum().item())
        lines: list[str] = []
        if n_pose_ok == 0:
            if best_q is not None:
                best_pos = float(pos_per_seed[best_idx].item())
                best_rot = float(rot_per_seed[best_idx].item())
                qual = "feasible " if best_is_feasible else ""
                lines.append(
                    f"pose unreachable (best {qual}seed {best_pos * 1000:.1f}mm/"
                    f"{best_rot:.3f}rad, tol {pos_tol * 1000:.1f}mm/{rot_tol:.3f}rad)"
                )
                lines.append(f"best seed: {self._classify_q(solutions[best_idx])}")
            else:
                best_pos = float(pos_per_seed.min().item())
                best_rot = float(rot_per_seed.min().item())
                lines.append(
                    f"pose unreachable (best {best_pos * 1000:.1f}mm/{best_rot:.3f}rad, "
                    f"tol {pos_tol * 1000:.1f}mm/{rot_tol:.3f}rad)"
                )
        else:
            pose_ok_idx = pose_ok.nonzero(as_tuple=False).reshape(-1).tolist()
            summary = (
                self._summarize_seed_reasons(solutions, pose_ok_idx)
                if solutions is not None
                else "n/a"
            )
            lines.append(
                f"pose reachable but all {n_pose_ok}/{n_returned} solutions infeasible: {summary}"
            )
        self._log(prefix, "IK", *lines)
        return best_q

    def _diagnose_graph_rejection(
        self, prefix, solutions, start_q, n_ik_success, ik_success
    ):
        """Diagnose graph-planner rejection of otherwise valid IK seeds.

        Re-checks each IK-successful seed and the start state against the graph
        planner and prints whether the start is in collision or which fraction
        of converged IK goals was rejected, with reason summaries.

        Args:
            prefix: Header string for log lines.
            solutions: IK joint configs, shape ``(n_seeds, dof)``.
            start_q: Start config used for the original plan, shape ``(dof,)``.
            n_ik_success: Number of IK seeds that converged on the pose.
            ik_success: Per-seed IK convergence flags, shape ``(n_seeds,)``.
        """
        gp = self._planner.graph_planner
        sols_dev = solutions.to(self._device).float()
        feas_graph = gp.check_samples_feasibility(sols_dev).reshape(-1).bool().cpu()
        start_dev = start_q.float().reshape(1, -1).to(self._device)
        start_ok = bool(gp.check_samples_feasibility(start_dev).all().item())
        # Only count IK-converged seeds; an IK-failed seed being graph-infeasible
        # tells us nothing useful.
        rejected_mask = (~feas_graph) & ik_success.cpu()
        bad = rejected_mask.nonzero(as_tuple=False).reshape(-1).tolist()

        if not start_ok:
            line = f"start state in collision: {self._classify_q(start_q)}"
        elif bad:
            summary = self._summarize_seed_reasons(solutions, bad)
            line = (
                f"graph planner rejected {len(bad)}/{n_ik_success} converged "
                f"IK goals: {summary}"
            )
        else:
            line = "graph planner rejected plan (cause unclear on re-check)"
        self._log(prefix, "collision", line)

    def _diagnose_trajopt_failure(self, prefix, result, goal, goal_label=None):
        """Diagnose a TrajOpt failure, distinguishing three sub-cases.

        Distinguishes:
          * pose unreached at trajectory end → ``TrajOpt``
          * pose reached but a waypoint violates a constraint → ``trajectory``
          * pose reached and waypoints clean → cuRobo's infeasibility flag is
            about dynamics (velocity/acceleration/jerk on the smoothed trajectory).

        Uses re-FK'd EE-to-goal distance instead of cuRobo's hinged metric and
        scans the returned interpolated trajectory ourselves, since cuRobo clears
        per-constraint metrics during topk reduction.

        Args:
            prefix: Header string for log lines.
            result: cuRobo ``MotionPlanResult``.
            goal: cuRobo goal pose object.
            goal_label: Optional short label (e.g. ``"goal 3"``) used to qualify
                the waypoint-violation body line.
        """
        pos_err, rot_err = self._actual_pose_error_at_traj_end(result, goal)
        if pos_err is None or rot_err is None:
            self._log(
                prefix,
                "TrajOpt",
                "could not recompute geometric EE error from trajectory "
                "(cuRobo's hinged metric is unreliable here; skipping pose check)",
            )
            return
        to_cfg = self._planner.trajopt_solver.config
        pos_tol = getattr(to_cfg, "position_tolerance", result.position_tolerance)
        rot_tol = getattr(to_cfg, "orientation_tolerance", result.orientation_tolerance)

        if pos_err > pos_tol or rot_err > rot_tol:
            self._log(
                prefix,
                "TrajOpt",
                f"pose not reached (trajectory ends {pos_err * 1000:.1f}mm/"
                f"{rot_err:.3f}rad from goal, tol {pos_tol * 1000:.1f}mm/{rot_tol:.3f}rad)",
            )
            return

        idx, n_total, reason = self._first_path_violation(result)
        pose_str = f"pose OK ({pos_err * 1000:.1f}mm/{rot_err:.3f}rad)"
        wp_owner = f"{goal_label} plan " if goal_label else ""
        if reason is not None:
            line = f"{pose_str}, {wp_owner}waypoint {idx}/{n_total}: {reason}"
        else:
            line = (
                f"{pose_str}, {wp_owner}returned path clean — cuRobo's failure flag is "
                "from dynamics (velocity/acceleration/jerk over the smoothed trajectory)"
            )
        self._log(prefix, "trajectory", line)

    def _first_path_violation(self, result):
        """Scan the interpolated trajectory and locate the first bad waypoint.

        Args:
            result: cuRobo ``MotionPlanResult``.

        Returns:
            ``(idx, n_total, reason)`` for the first waypoint violating a joint
            limit, world collision, or self-collision. Returns
            ``(None, n_total, None)`` if the entire trimmed path is clean, or
            ``(None, 0, None)`` if no interpolated trajectory is available.
        """
        positions = result.get_interpolated_plan().position
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
        below = (lims[0] - w_dev).clamp(min=0.0)
        above = (w_dev - lims[1]).clamp(min=0.0)
        jl_per_wp = ((below + above).sum(dim=-1) > 0).cpu().tolist()

        for k in range(n_total):
            if jl_per_wp[k]:
                return (
                    k,
                    n_total,
                    f"joint limits: {self._joint_limit_status(waypoints[k])[2]}",
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
        """Classify a subset of IK seeds and group identical reasons with counts.

        Args:
            solutions: All IK joint configs, shape ``(n_seeds, dof)``.
            indices: Iterable of seed indices to classify.

        Returns:
            Comma-separated string like ``"3× world collision (...); 1× joint limits ..."``.
        """
        reasons: dict[str, int] = {}
        for k in indices:
            r = self._classify_q(solutions[k])
            reasons[r] = reasons.get(r, 0) + 1
        return ", ".join(f"{n}× {r}" for r, n in reasons.items())
