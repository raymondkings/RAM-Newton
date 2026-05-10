"""Collision oracle wrapping a Newton model for fast joint-config collision queries."""
import newton
import warp as wp
import torch
import numpy as np
from scipy.spatial.transform import Rotation

from interface import Morphology, Task
from util.kinematics import compute_link_world_poses
from validation.mdh_to_newton import add_robot_to_builder
from validation.ground import add_ground_collision


def _to_world(body_xform: np.ndarray, p_local: np.ndarray) -> np.ndarray:
    """body_xform is (px,py,pz, qx,qy,qz,qw) — Newton's wp.transform layout."""
    t = body_xform[:3]
    q = body_xform[3:]
    return Rotation.from_quat(q).apply(p_local) + t


class CollisionOracle:
    """Builds a Newton scene once; checks any joint config in O(collide)."""

    def __init__(self, morph: Morphology, task: Task):
        builder = newton.ModelBuilder()
        add_ground_collision(builder)

        for obs in task.environment.obstacles:
            if obs.kind == "box":
                builder.add_shape_box(
                    body=-1,
                    xform=wp.transform(p=wp.vec3(*obs.center.tolist()), q=wp.quat_identity()),
                    hx=obs.half_extents[0].item(),
                    hy=obs.half_extents[1].item(),
                    hz=obs.half_extents[2].item(),
                )

        rest_poses = compute_link_world_poses(morph)
        rest_poses = task.environment.base_pose.unsqueeze(0) @ rest_poses
        add_robot_to_builder(builder, morph, rest_poses)

        self.model = builder.finalize()
        self.state = self.model.state()
        self.shape_body = self.model.shape_body.numpy()
        self.n_joints = morph.n_links - 1
        self._ground_shape = 0   # add_ground_collision() is always called first
        self._base_body = 0      # first robot body (the riser / link 0)
        self._ignore_pairs = self._build_ignore_pairs()

    def _build_ignore_pairs(self) -> set[tuple[int, int]]:
        """Self-collision pairs to ignore: bodies that are kinematically adjacent
        through zero or more shapeless intermediate bodies. They share an endpoint
        at the joint location, which the narrow-phase mistakes for penetration.
        """
        joint_parent = self.model.joint_parent.numpy()
        joint_child = self.model.joint_child.numpy()
        n_bodies = max(int(self.shape_body.max()) + 1, 1)
        bodies_with_shape = {int(b) for b in self.shape_body if int(b) != -1}

        parent_of = {-1: -1}
        for p, c in zip(joint_parent, joint_child):
            parent_of[int(c)] = int(p)

        ignore: set[tuple[int, int]] = set()
        for body in range(n_bodies):
            if body not in bodies_with_shape:
                continue
            anc = parent_of.get(body, -1)
            while anc != -1 and anc not in bodies_with_shape:
                anc = parent_of.get(anc, -1)
            if anc != -1:
                ignore.add(tuple(sorted([body, anc])))
        return ignore

    def _count_contacts(self, gap_tol: float = 1e-4) -> tuple[int, int]:
        """Count actual penetrations. Newton's narrow-phase reports closest-point
        pairs even when bodies are not penetrating, so we filter by the signed
        distance along the contact normal: gaps with positive separation > gap_tol
        are not real contacts.
        """
        contacts = self.model.collide(self.state)
        n_contacts = int(contacts.rigid_contact_count.numpy()[0])
        if n_contacts == 0:
            return 0, 0

        shape0 = contacts.rigid_contact_shape0.numpy()[:n_contacts]
        shape1 = contacts.rigid_contact_shape1.numpy()[:n_contacts]
        p0 = contacts.rigid_contact_point0.numpy()[:n_contacts]
        p1 = contacts.rigid_contact_point1.numpy()[:n_contacts]
        normal = contacts.rigid_contact_normal.numpy()[:n_contacts]
        body_q = self.state.body_q.numpy()

        n_self = 0
        n_env = 0
        for i in range(n_contacts):
            body_a = self.shape_body[shape0[i]]
            body_b = self.shape_body[shape1[i]]

            if body_a != -1 and body_b != -1:
                pair = tuple(sorted([int(body_a), int(body_b)]))
                if pair in self._ignore_pairs:
                    continue

            p0w = p0[i] if body_a == -1 else _to_world(body_q[body_a], p0[i])
            p1w = p1[i] if body_b == -1 else _to_world(body_q[body_b], p1[i])

            signed_dist = float(np.dot(p1w - p0w, normal[i]))
            if signed_dist > gap_tol:
                continue

            if body_a == -1 or body_b == -1:
                robot_body = body_b if body_a == -1 else body_a
                env_shape = shape0[i] if body_a == -1 else shape1[i]
                if robot_body == self._base_body and env_shape == self._ground_shape:
                    continue  # base is mounted to the ground — not a collision
                n_env += 1
            else:
                n_self += 1
        return n_self, n_env

    def is_collision_free(self, joint_q: torch.Tensor) -> bool:
        """joint_q: (n_joints,) torch tensor — single configuration."""
        q = joint_q.detach().cpu().numpy().astype(np.float32).reshape(-1)
        self.state.joint_q.assign(q)
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
        n_self, n_env = self._count_contacts()
        return n_self == 0 and n_env == 0

    def edge_collision_free(
        self,
        q_a: torch.Tensor,
        q_b: torch.Tensor,
        step_size: float = 0.05,
    ) -> bool:
        """Discretized straight-line check in joint space (excluding endpoint b).

        Endpoints a/b are assumed already validated by the caller.
        """
        delta = q_b - q_a
        n_steps = max(1, int(torch.ceil(delta.abs().max() / step_size).item()))
        for k in range(1, n_steps):
            t = k / n_steps
            q = q_a + t * delta
            if not self.is_collision_free(q):
                return False
        return True