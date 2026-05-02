import math
import torch
from interface import Morphology


def make_test_morphology() -> Morphology:
    """Simple 3-link unit-norm morphology — no collisions when outside room."""
    params = torch.tensor([
        [0.0, 0.0, 1/3],
        [0.0, 0.0, 1/3],
        [0.0, 0.0, 1/3],
    ], dtype=torch.float32)
    return Morphology(params=params, link_radius=0.05)


def make_self_colliding_morphology() -> Morphology:
    """6-link morphology that self-collides at zero joint angles.
    
    Alternating pi/2 twists cause the chain to spiral. With equal
    link lengths, a later link inevitably overlaps an earlier one.
    """
    n_links = 7
    d_per_link = 1.0 / n_links   # unit norm: sum of d = 1.0

    params = torch.tensor([
        [0.0,          0.0,  d_per_link],   # straight up
        [math.pi / 2,  0.0,  d_per_link],   # turn
        [math.pi / 2,  0.0,  d_per_link],   # turn again
        [math.pi / 2,  0.0,  d_per_link],   # turn again → pointing back down
        [math.pi / 2,  0.0,  d_per_link],   # continues spiral
        [math.pi / 2,  0.0,  d_per_link],   # overlap with earlier links
        [math.pi / 2,  0.0,  d_per_link],   # definite overlap
    ], dtype=torch.float32)
    return Morphology(params=params, link_radius=0.05)