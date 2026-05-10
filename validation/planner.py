"""RRT-Connect in joint space.
Each joint is assumed free to rotate in [-pi, pi]. Distance is Euclidean in
joint coordinates (no manifold wrapping for v1 — adequate when the planner's
step size is well below the wrap discontinuity).
"""
from dataclasses import dataclass
import torch

from validation.collision_oracle import CollisionOracle


@dataclass
class PlanResult:
    success: bool
    path: list[torch.Tensor]   # joint configs along the path; empty if not success
    n_iterations: int
    n_nodes: int
    kinematic_only: bool = False  # True if path ignores environment collisions


class _Tree:
    def __init__(self, root: torch.Tensor):
        self.nodes: list[torch.Tensor] = [root]
        self.parents: list[int] = [-1]

    def add(self, q: torch.Tensor, parent: int) -> int:
        self.nodes.append(q)
        self.parents.append(parent)
        return len(self.nodes) - 1

    def nearest(self, q: torch.Tensor) -> int:
        stacked = torch.stack(self.nodes)
        dists = (stacked - q.unsqueeze(0)).norm(dim=-1)
        return int(dists.argmin().item())

    def path_to(self, idx: int) -> list[torch.Tensor]:
        out = []
        while idx != -1:
            out.append(self.nodes[idx])
            idx = self.parents[idx]
        return out


def _sample(n_joints: int) -> torch.Tensor:
    return torch.empty(n_joints).uniform_(-torch.pi, torch.pi)


def _extend(
    tree: _Tree,
    q_target: torch.Tensor,
    oracle: CollisionOracle,
    step_size: float,
) -> tuple[str, int]:
    """Greedy extend toward q_target, taking unit steps until blocked or reached.
    Returns (status, last_added_index) where status ∈ {"reached", "advanced", "trapped"}.
    """
    nearest_idx = tree.nearest(q_target)
    q_near = tree.nodes[nearest_idx]
    last_idx = nearest_idx
    last_q = q_near

    while True:
        delta = q_target - last_q
        dist = delta.norm()
        if dist.item() < step_size:
            q_new = q_target
            reached = True
        else:
            q_new = last_q + (step_size / dist) * delta
            reached = False

        if not oracle.is_collision_free(q_new):
            return ("trapped" if last_idx == nearest_idx else "advanced", last_idx)
        if not oracle.edge_collision_free(last_q, q_new, step_size=step_size):
            return ("trapped" if last_idx == nearest_idx else "advanced", last_idx)

        last_idx = tree.add(q_new, last_idx)
        last_q = q_new
        if reached:
            return "reached", last_idx


def rrt_connect(
    q_start: torch.Tensor,
    q_goal: torch.Tensor,
    oracle: CollisionOracle,
    max_iter: int = 1000,
    step_size: float = 0.15,
    seed: int | None = None,
) -> PlanResult:
    """Bidirectional RRT. Trees grow from start and goal until they meet."""
    if seed is not None:
        torch.manual_seed(seed)

    if not oracle.is_collision_free(q_start):
        raise ValueError("start configuration is in collision")
    if not oracle.is_collision_free(q_goal):
        raise ValueError("goal configuration is in collision")

    n_joints = q_start.shape[0]
    tree_a = _Tree(q_start)
    tree_b = _Tree(q_goal)

    for it in range(max_iter):
        q_rand = _sample(n_joints)

        # Extend tree A toward q_rand
        status_a, idx_a = _extend(tree_a, q_rand, oracle, step_size)
        if status_a == "trapped":
            tree_a, tree_b = tree_b, tree_a
            continue

        # Try to connect tree B to the new node
        q_new_a = tree_a.nodes[idx_a]
        status_b, idx_b = _extend(tree_b, q_new_a, oracle, step_size)
        if status_b == "reached":
            n_total = len(tree_a.nodes) + len(tree_b.nodes)
            # Reconstruct path: start → idx_a (in tree_a), idx_b → goal (in tree_b)
            path_a = tree_a.path_to(idx_a)[::-1]   # root to idx_a
            path_b = tree_b.path_to(idx_b)         # idx_b back to root

            # tree_a/tree_b may have been swapped — orient by checking endpoints
            full = path_a + path_b
            if not torch.equal(full[0], q_start):
                full = full[::-1]
            return PlanResult(success=True, path=full, n_iterations=it + 1, n_nodes=n_total)

        tree_a, tree_b = tree_b, tree_a

    return PlanResult(
        success=False,
        path=[],
        n_iterations=max_iter,
        n_nodes=len(tree_a.nodes) + len(tree_b.nodes),
    )


def shortcut(
    path: list[torch.Tensor],
    oracle: CollisionOracle,
    n_attempts: int = 100,
    step_size: float = 0.05,
) -> list[torch.Tensor]:
    """Random-shortcut smoothing: try to replace path[i..j] with a straight line."""
    if len(path) < 3:
        return path
    path = list(path)
    for _ in range(n_attempts):
        n = len(path)
        if n < 3:
            break
        i = int(torch.randint(0, n - 2, (1,)).item())
        j = int(torch.randint(i + 2, n, (1,)).item())
        if oracle.edge_collision_free(path[i], path[j], step_size=step_size):
            path = path[: i + 1] + path[j:]
    return path


def interpolate_path(path: list[torch.Tensor], step: float = 0.02) -> list[torch.Tensor]:
    """Densify path with at most `step` joint-distance between consecutive frames."""
    if len(path) < 2:
        return path
    out = [path[0]]
    for q_a, q_b in zip(path[:-1], path[1:]):
        delta = q_b - q_a
        n = max(1, int(torch.ceil(delta.norm() / step).item()))
        for k in range(1, n + 1):
            out.append(q_a + (k / n) * delta)
    return out