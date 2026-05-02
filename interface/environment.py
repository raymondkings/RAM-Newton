from dataclasses import dataclass, field
from typing import Literal
import torch

@dataclass
class Box:
    kind: Literal["box"] = "box"
    center: torch.Tensor = field(default_factory=lambda: torch.zeros(3))
    half_extents: torch.Tensor = field(default_factory=lambda: torch.ones(3) * 0.1)
    rotation: torch.Tensor = field(default_factory=lambda: torch.tensor([0., 0., 0., 1.]))

@dataclass
class Sphere:
    kind: Literal["sphere"] = "sphere"
    center: torch.Tensor = field(default_factory=lambda: torch.zeros(3))
    radius: float = 0.1

@dataclass
class Capsule:
    kind: Literal["capsule"] = "capsule"
    center: torch.Tensor = field(default_factory=lambda: torch.zeros(3))
    half_height: float = 0.1
    radius: float = 0.05
    rotation: torch.Tensor = field(default_factory=lambda: torch.tensor([0., 0., 0., 1.]))

Obstacle = Box | Sphere | Capsule


@dataclass
class Environment:
    """A scene of static obstacles."""
    obstacles: list[Obstacle] = field(default_factory=list)
    base_pose: torch.Tensor = field(default_factory=lambda: torch.eye(4))   # robot mount