import numpy as np
from scipy.spatial.transform import Rotation


def _to_world(body_xform: np.ndarray, p_local: np.ndarray) -> np.ndarray:
    """body_xform is (px,py,pz, qx,qy,qz,qw) — Newton's wp.transform layout."""
    t = body_xform[:3]
    q = body_xform[3:]
    return Rotation.from_quat(q).apply(p_local) + t


def build_self_collision_ignore_pairs(model) -> set[tuple[int, int]]:
    """Body pairs to skip for self-collision checks.

    Skips direct parent-child body pairs (capsules separated by exactly one
    joint). These share an endpoint at the joint location, so naive
    capsule-capsule distance always returns zero and always reports collision.
    Scissor collisions between these pairs must be prevented upstream via
    joint limits set at morphology construction.

    Same-body pairs (the two orthogonal capsules of one MDH link) are handled
    at query time via body_a == body_b, not here.
    """
    parents = model.joint_parent.numpy()
    children = model.joint_child.numpy()
    return {tuple(sorted((int(p), int(c)))) for p, c in zip(parents, children)}


def check_collisions(
    model,
    state,
    ignore_self_pairs: set[tuple[int, int]] | None = None,
    base_body: int = 0,
    ground_shape: int | None = None,
    gap_tol: float = 1e-4,
    debug: bool = False,
) -> dict:
    """Check collisions and classify as self / env / ignored.

    Newton's narrow-phase reports closest-point pairs even when bodies are not
    penetrating. Contacts where the signed distance along the contact normal
    exceeds gap_tol are filtered out as gaps.

    Args:
        model:             Newton Model with finalised collision shapes.
        state:             Current simulation State.
        ignore_self_pairs: Body-index pairs (sorted tuples) to skip for
                           self-collision. Build once via
                           build_self_collision_ignore_pairs().
        base_body:         Body index of the robot base. Contacts between this
                           body and ground_shape are ignored.
        ground_shape:      Shape index of the ground box. If None, base-ground
                           filtering is disabled.
        gap_tol:           Signed-distance threshold: contacts with separation
                           larger than this are treated as gaps, not collisions.
        debug:             Print each contact with its classification.
    """
    contacts = model.collide(state)
    n_contacts = int(contacts.rigid_contact_count.numpy()[0])

    if n_contacts == 0:
        return {"n_self_collisions": 0, "n_env_collisions": 0,
                "n_ignored": 0, "n_total": 0}

    ignore_self_pairs = ignore_self_pairs or set()

    shape_labels = model.shape_label  # list[str], one per shape
    shape_body = model.shape_body.numpy()
    shape0 = contacts.rigid_contact_shape0.numpy()[:n_contacts]
    shape1 = contacts.rigid_contact_shape1.numpy()[:n_contacts]
    p0_arr = contacts.rigid_contact_point0.numpy()[:n_contacts]
    p1_arr = contacts.rigid_contact_point1.numpy()[:n_contacts]
    normal_arr = contacts.rigid_contact_normal.numpy()[:n_contacts]
    body_q = state.body_q.numpy()

    def _shape_str(idx: int) -> str:
        label = shape_labels[idx] if idx < len(shape_labels) else ""
        return f"{idx} \"{label}\"" if label else str(idx)

    n_self = n_env = n_ignored = 0

    for i in range(n_contacts):
        s_a, s_b = int(shape0[i]), int(shape1[i])
        body_a = int(shape_body[s_a])
        body_b = int(shape_body[s_b])

        p0w = p0_arr[i] if body_a == -1 else _to_world(body_q[body_a], p0_arr[i])
        p1w = p1_arr[i] if body_b == -1 else _to_world(body_q[body_b], p1_arr[i])
        signed_dist = float(np.dot(p1w - p0w, normal_arr[i]))
        if signed_dist > gap_tol:
            if debug:
                print(f"  [gap] shape {_shape_str(s_a)} (body {body_a}) vs shape {_shape_str(s_b)} (body {body_b}), dist={signed_dist:.4f}")
            continue

        if body_a == -1 or body_b == -1:
            robot_body = body_b if body_a == -1 else body_a
            env_shape = s_a if body_a == -1 else s_b

            if ground_shape is not None and robot_body == base_body and env_shape == ground_shape:
                kind = "ignored:base-ground"
                n_ignored += 1
            else:
                kind = "env"
                n_env += 1
        else:
            if body_a == body_b:
                kind = "ignored:same-link"
                n_ignored += 1
            elif tuple(sorted((body_a, body_b))) in ignore_self_pairs:
                kind = "ignored:adjacent"
                n_ignored += 1
            else:
                kind = "self"
                n_self += 1

        if debug:
            print(f"  [{kind}] shape {_shape_str(s_a)} (body {body_a}) "
                  f"vs shape {_shape_str(s_b)} (body {body_b})")

    return {"n_self_collisions": n_self, "n_env_collisions": n_env,
            "n_ignored": n_ignored, "n_total": n_contacts}