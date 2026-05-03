from dataclasses import dataclass

import torch
from jaxtyping import Float

@dataclass
class Morphology:
    """
    Each row of `params` = one link's MDH token = [α, a, d]

    α (column 0): twist between consecutive joint axes
                    discrete in {−π/2, 0, π/2}

    a (column 1): link length along previous x-axis
                    continuous in [−1, −2r] ∪ [2r, 1]
                    directly optimised by AdamW

    d (column 2): link offset along current z-axis
                    continuous in [−1, −2r] ∪ [2r, 1]
                    directly optimised by AdamW

    Geometry: each link = two orthogonal capsules of radius r,
            one of length |a| along x, one of length |d| along z.
            Either can be zero (disappears).

    Number of rows = (degrees of freedom) + 1, because the end-effector
                    frame also needs an MDH transform.

    Joint angles θ live elsewhere — they are configuration, not morphology.
    """
    params: Float[torch.Tensor, "n_links 3"]
    link_radius: float = 0.025

    @property
    def n_links(self) -> int:
        return self.params.shape[0]
    
    @property
    def alpha(self) -> Float[torch.Tensor, "n_links"]:
        return self.params[:, 0]

    @property
    def a(self) -> Float[torch.Tensor, "n_links"]:
        return self.params[:, 1]
    
    @property
    def d(self) -> Float[torch.Tensor, "n_links"]:
        return self.params[:, 2]