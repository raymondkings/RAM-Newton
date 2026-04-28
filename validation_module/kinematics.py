import torch
from interface import Morphology


def compute_link_world_poses(
    morph: Morphology,
    joint_angles: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute world poses for each link via MDH forward kinematics.
    
    Args:
        morph: Morphology with n_links rows.
        joint_angles: shape (n_links - 1,) — angles for moving joints.
                      If None, all zeros (rest pose).
    
    Returns:
        poses: shape (n_links, 4, 4)
    """
    n = morph.n_links
    if joint_angles is None:
        joint_angles = torch.zeros(n - 1)
    
    poses = torch.zeros(n, 4, 4, dtype=torch.float32)
    T = torch.eye(4, dtype=torch.float32)
    
    for i in range(n):
        alpha = morph.alpha[i].item()
        a = morph.a[i].item()
        d = morph.d[i].item()
        theta = 0.0 if i == 0 else joint_angles[i - 1].item()
        
        ct = torch.cos(torch.tensor(theta, dtype=torch.float32))
        st = torch.sin(torch.tensor(theta, dtype=torch.float32))
        ca = torch.cos(torch.tensor(alpha, dtype=torch.float32))
        sa = torch.sin(torch.tensor(alpha, dtype=torch.float32))
        
        # Modified DH transform (Tim's paper Eq. 1)
        Ti = torch.tensor([
            [ct,        -st,        0.0,    a],
            [st * ca,    ct * ca,  -sa,    -d * sa],
            [st * sa,    ct * sa,   ca,     d * ca],
            [0.0,        0.0,       0.0,    1.0],
        ], dtype=torch.float32)
        
        T = T @ Ti
        poses[i] = T
    
    return poses