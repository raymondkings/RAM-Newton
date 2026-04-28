import torch
from interface import Morphology


def make_test_morphology() -> Morphology:
    """Simple 3-link unit-norm morphology for testing."""
    # 3 links, each contributing d = 1/3 to total reach
    # All twists = 0 (parallel joint axes)
    # Total reach: 1/3 + 1/3 + 1/3 = 1.0 ✓
    # Each |d| = 1/3 ≈ 0.333, well above 2r = 0.1 ✓
    params = torch.tensor([
        [0.0, 0.0, 1/3],
        [0.0, 0.0, 1/3],
        [0.0, 0.0, 1/3],
    ], dtype=torch.float32)
    
    return Morphology(params=params, link_radius=0.05)