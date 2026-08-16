import math
import tempfile

import torch
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.scene import Cuboid, Scene
from curobo.types import GoalToolPose, JointState
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Kinematics with mdh parameters
# ---------------------------------------------------------------------------


def transformation_matrix(
    alpha: torch.Tensor, a: torch.Tensor, d: torch.Tensor, theta: torch.Tensor
) -> torch.Tensor:
    """
    Compute the modified Denavit-Hartenberg transformation matrix.

    Args:
        alpha: twist angle, shape [..., 1]
        a: link length, shape [..., 1]
        d: link offset, shape [..., 1]
        theta: joint angle, shape [..., 1]

    Returns:
        Homogeneous transform, shape [..., 4, 4]
    """
    ca, sa = torch.cos(alpha), torch.sin(alpha)
    ct, st = torch.cos(theta), torch.sin(theta)
    zero = torch.zeros_like(alpha)
    one = torch.ones_like(alpha)

    return torch.stack(
        [
            torch.cat([ct, -st, zero, a], dim=-1),
            torch.cat([st * ca, ct * ca, -sa, -d * sa], dim=-1),
            torch.cat([st * sa, ct * sa, ca, d * ca], dim=-1),
            torch.cat([zero, zero, zero, one], dim=-1),
        ],
        dim=-2,
    )


def forward_kinematics(mdh: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Compute forward kinematics for a robot defined by MDH parameters.

    Args:
        mdh: Tensor [..., dofp1, 3], where each row is [alpha_i, a_i, d_i].
        theta: Tensor [..., dofp1, 1], joint angle for each MDH row.
               The last theta is usually zero for the end-effector transform.

    Returns:
        Tensor [..., dofp1, 4, 4], base-to-joint transforms.
    """
    transforms = transformation_matrix(
        mdh[..., 0:1], mdh[..., 1:2], mdh[..., 2:3], theta
    )

    poses = []
    pose = torch.eye(4, device=mdh.device, dtype=mdh.dtype).expand(
        *mdh.shape[:-2], 1, 4, 4
    )
    for i in range(mdh.shape[-2]):
        pose = pose @ transforms[..., i : i + 1, :, :]
        poses.append(pose)
    return torch.cat(poses, dim=-3)


def compute_link_world_poses(morph, q: torch.Tensor | None = None) -> torch.Tensor:
    """Forward kinematics at the given joint configuration (defaults to all-zero rest pose).

    Args:
        morph: Morphology whose MDH params define the robot.
        q: Joint angles [n_joints], where n_joints = n_links - 1. If None, uses zeros.

    Returns:
        Tensor [n_links, 4, 4], base-to-link transforms.
    """
    mdh = morph.params
    if q is None:
        theta = torch.zeros(mdh.shape[0], 1, device=mdh.device, dtype=mdh.dtype)
    else:
        theta = torch.cat([q.to(dtype=mdh.dtype), q.new_zeros(1)]).unsqueeze(-1)
    return forward_kinematics(mdh, theta)


# ---------------------------------------------------------------------------
# cuRobo scene building and URDF generation
# ---------------------------------------------------------------------------


def _capsule_spheres(
    center: list[float],
    half_height: float,
    axis: list[float],
    radius: float,
) -> list[dict]:
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


def mat_to_goal_pose(
    pose_4x4: torch.Tensor,
    tool_frames: list[str],
    device: torch.device,
) -> "GoalToolPose":
    """Convert a (4, 4) homogeneous matrix to a cuRobo GoalToolPose."""
    # Extract position and reshape to cuRobo batch format [1, 1, 1, 1, 3]
    pos = pose_4x4[:3, 3].float().contiguous().reshape(1, 1, 1, 1, 3).to(device)

    # Extract rotation and convert to quaternion, reordering from xyzw to wxyz
    rot_matrix = pose_4x4[:3, :3].float().cpu().numpy()
    q_xyzw = Rotation.from_matrix(rot_matrix).as_quat()
    q_wxyz = torch.tensor(
        [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]],
        dtype=torch.float32,
        device=device,
    ).reshape(1, 1, 1, 1, 4)

    return GoalToolPose(tool_frames=tool_frames, position=pos, quaternion=q_wxyz)


def build_sphere_dict(morph) -> dict[str, list[dict]]:
    """Per-link collision sphere dicts matching Newton's capsule geometry."""
    n = morph.n_links
    r = morph.link_radius
    sphere_dict: dict[str, list[dict]] = {}
    a0 = morph.a[0].item()
    if abs(a0) > 2.0 * r:
        sphere_dict["base_link"] = _capsule_spheres(
            [a0 / 2.0, 0.0, 0.0], abs(a0) / 2.0, [1.0, 0.0, 0.0], r
        )

    for j in range(n):
        spheres: list[dict] = []
        d_val = morph.d[j].item()
        if abs(d_val) > 2.0 * r:
            spheres += _capsule_spheres(
                [0.0, 0.0, -d_val / 2.0], abs(d_val) / 2.0, [0.0, 0.0, 1.0], r
            )
        if j < n - 1:
            a_next = morph.a[j + 1].item()
            if abs(a_next) > 2.0 * r:
                spheres += _capsule_spheres(
                    [a_next / 2.0, 0.0, 0.0], abs(a_next) / 2.0, [1.0, 0.0, 0.0], r
                )

        # Degenerate-link fallback: if neither d_j nor a_{j+1} produced a capsule,
        # `spheres` is empty, so `or` falls back to a single sphere at the joint origin.
        sphere_dict[f"link_{j}"] = spheres or [
            {"center": [0.0, 0.0, 0.0], "radius": float(r)}
        ]
    return sphere_dict


def _is_degenerate_link(morph, j: int, r: float) -> bool:
    """True if link_j collapses to the single fallback sphere in build_sphere_dict.

    A link has no capsule when its own offset d_j and its outgoing length a_{j+1}
    are both below 2r (the last link has no a_{j+1}). Such a zero-length body sits
    at the joint origin, so it cannot separate its neighbours.
    """
    n = morph.n_links
    has_d = abs(morph.d[j].item()) > 2.0 * r
    has_a_next = j < n - 1 and abs(morph.a[j + 1].item()) > 2.0 * r
    return not (has_d or has_a_next)


def build_self_collision_ignore(morph) -> dict[str, list[str]]:
    """Ignore each link's effective neighbours to suppress false contacts.

    Two links are effectively adjacent — and thus permanently touching at a
    shared joint — when every link strictly between them is degenerate (a
    zero-length body, see _is_degenerate_link). This skips over runs of
    degenerate links, so a real link in between (which can genuinely fold into
    contact) is left checked, matching the paper's "closest non-zero capsule"
    rule. base_link is included unconditionally; absent names are filtered by
    the caller.
    """
    chain = ["base_link"] + [f"link_{i}" for i in range(morph.n_links)]
    # base_link is never a "between" link, so its degeneracy is irrelevant.
    degenerate = [False] + [
        _is_degenerate_link(morph, i, morph.link_radius) for i in range(morph.n_links)
    ]
    ignore: dict[str, list[str]] = {lnk: [] for lnk in chain}
    for i in range(len(chain)):
        j = i + 1
        while j < len(chain):
            ignore[chain[i]].append(chain[j])
            ignore[chain[j]].append(chain[i])
            # Extend to j+1 only while link j (the next "between") is degenerate.
            if not degenerate[j]:
                break
            j += 1
    return ignore


def build_robot_dict(morph) -> tuple[dict, str]:
    """Build a cuRobo robot dict and write a temporary URDF.

    Returns:
        (robot_dict, urdf_path) — caller must delete urdf_path when done.
    """
    from kinematics.mdh import to_urdf  # local to avoid circular import

    ee_link = f"link_{morph.n_links - 1}"
    urdf_str = to_urdf(morph)
    tmp = tempfile.NamedTemporaryFile(suffix=".urdf", delete=False, mode="w")  # noqa: SIM115 -- file must persist on disk after this block
    tmp.write(urdf_str)
    tmp.flush()
    tmp.close()

    sphere_dict = build_sphere_dict(morph)
    robot_dict = {
        "kinematics": {
            "urdf_path": tmp.name,
            "base_link": "base_link",
            "tool_frames": [ee_link],
            "collision_spheres": sphere_dict,
            "collision_link_names": list(sphere_dict.keys()),
            "collision_sphere_buffer": 0.01,
            "self_collision_buffer": {k: 0.0 for k in sphere_dict},
            "self_collision_ignore": build_self_collision_ignore(morph),
        }
    }
    return robot_dict, tmp.name


def build_scene(
    task,
    ignore_ground: bool = False,
    ignore_obstacles: bool = False,
) -> Scene:
    """Convert task obstacles to a cuRobo Scene in the shared world/base frame."""
    cuboids: list[Cuboid] = []

    if not ignore_ground:
        cuboids.append(
            Cuboid(
                name="ground",
                dims=[100.0, 100.0, 1.0],
                pose=[0.0, 0.0, -0.5, 1.0, 0.0, 0.0, 0.0],
            )
        )

    if not ignore_obstacles:
        for idx, obs in enumerate(task.environment.obstacles):
            c = obs.center.float().cpu().tolist()
            q_xyzw = obs.rotation[[1, 2, 3, 0]].float().cpu().tolist()
            pose = c + [
                float(q_xyzw[3]),
                float(q_xyzw[0]),
                float(q_xyzw[1]),
                float(q_xyzw[2]),
            ]
            cuboids.append(
                Cuboid(
                    name=f"box_{idx}",
                    dims=(2.0 * obs.half_extents).float().cpu().tolist(),
                    pose=pose,
                )
            )

    return Scene(cuboid=cuboids)


# ---------------------------------------------------------------------------
# cuRobo IK and FK classes
# ---------------------------------------------------------------------------


class FK:
    """cuRobo forward kinematics."""

    def __init__(self, robot_dict: dict) -> None:
        self._model = Kinematics(KinematicsCfg.from_data_dict(robot_dict))

    def compute(
        self,
        joints: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute FK for a batch of joint configurations.

        Args:
            joints: [N, dof] joint angles.
            device: Torch device.

        Returns:
            [N, 4, 4] EE poses in the shared world/base frame.
        """
        N = joints.shape[0]
        dtype = joints.dtype

        state = self._model.compute_kinematics(
            JointState.from_position(
                joints.to(device).float(), joint_names=self._model.joint_names
            )
        )
        ee_pose = state.tool_poses.get_link_pose(self._model.tool_frames[0])
        achieved_pos = ee_pose.position.to(dtype).to(device)
        achieved_q_xyzw = (
            ee_pose.quaternion[:, [1, 2, 3, 0]].float().detach().cpu().numpy()
        )
        achieved_rot = torch.tensor(
            Rotation.from_quat(achieved_q_xyzw).as_matrix(), dtype=dtype, device=device
        )

        reached = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(N, 1, 1)
        reached[:, :3, :3] = achieved_rot
        reached[:, :3, 3] = achieved_pos

        return reached


class IK:
    """Batched cuRobo inverse kinematics solver."""

    def __init__(
        self,
        robot_dict: dict,
        scene,
        num_seeds: int = 32,
        max_batch_size: int = 1,
        self_collision_check: bool = True,
    ) -> None:
        config = InverseKinematicsCfg.create(
            robot=robot_dict,
            scene_model=scene,
            num_seeds=num_seeds,
            self_collision_check=self_collision_check,
            collision_cache={"cuboid": 20},
            max_batch_size=max_batch_size,
        )
        self._solver = InverseKinematics(config)

    def solve(
        self,
        goal_poses: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve IK for a batch of goal poses.

        Args:
            goal_poses: [N, 4, 4] goal EE poses in the shared world/base frame.
            device:     Torch device the solver runs on.

        Returns:
            joints:  [N, dof] best joint configuration per goal.
            success: [N] bool — IK converged within tolerance.
        """
        N = goal_poses.shape[0]
        goals = goal_poses.to(device=device)

        pos = goals[:, :3, 3].float().contiguous().reshape(N, 1, 1, 1, 3)
        rots_np = goals[:, :3, :3].float().cpu().numpy()
        quats = [Rotation.from_matrix(rots_np[i]).as_quat() for i in range(N)]
        quat_wxyz = torch.tensor(
            [[q[3], q[0], q[1], q[2]] for q in quats],
            dtype=torch.float32,
            device=device,
        ).reshape(N, 1, 1, 1, 4)

        result = self._solver.solve_pose(
            GoalToolPose(
                tool_frames=self._solver.tool_frames, position=pos, quaternion=quat_wxyz
            )
        )

        q_sol = result.solution
        if q_sol.ndim == 3:
            q_sol = q_sol[:, 0, :]  # drop seed dim -> [N, dof]
        joints = q_sol.to(device).float()

        return joints, result.success.cpu()


def pose_errors(
    reached_poses: torch.Tensor,
    goal_poses: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-pose position and rotation errors between two sets of SE3 poses.

    Args:
        reached_poses: [N, 4, 4] achieved EE poses.
        goal_poses:    [N, 4, 4] target EE poses.

    Returns:
        pos_err: [N] position error in metres.
        rot_err: [N] rotation error in radians.
    """
    dtype = reached_poses.dtype
    pos_err = (reached_poses[:, :3, 3] - goal_poses[:, :3, 3].to(dtype)).norm(dim=-1)
    R_rel = reached_poses[:, :3, :3] @ goal_poses[:, :3, :3].to(dtype).transpose(-1, -2)
    trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    rot_err = torch.acos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0))
    return pos_err, rot_err


def se3_distance(pos_err: torch.Tensor, rot_err: torch.Tensor) -> torch.Tensor:
    """Combine position and rotation errors into a single SE3 scalar.

    Normalised so that 2 m translation and π rotation each contribute equally,
    with a maximum possible distance of 1.

    Args:
        pos_err: [*] position error in metres.
        rot_err: [*] rotation error in radians.

    Returns:
        [*] SE3 distance in [0, 1].
    """
    return (pos_err**2 / 8.0 + rot_err**2 / (2.0 * torch.pi**2)).sqrt()
