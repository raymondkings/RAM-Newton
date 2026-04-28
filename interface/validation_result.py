from dataclasses import dataclass

@dataclass
class ValidationResult:
    self_collision_free: bool
    env_collision_free: bool
    n_self_collisions: int
    n_env_collisions: int
    n_samples: int