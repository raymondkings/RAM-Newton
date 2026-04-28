# nrm-newton

Gradient-based robot design pipeline using Neural Reachability Maps (NRM) and Newton physics simulator.

Practical course project — TUM CPS, Summer 2026.

**Team:** Julian, Shiyuan, Jiyao, Raymond

---

## Interface Types

| Type | Key Fields |
|---|---|
| `Morphology` | `params (n_links, 3)` — columns `[α, a, d]`; `link_radius` |
| `Environment` | `obstacles: list[Box\|Sphere\|Capsule]`; `base_pose (4,4)` |
| `Task` | `environment`, `goal_poses (N,4,4)`, `reachable_region` |
| `ValidationResult` | `self_collision_free`, `env_collision_free`, collision counts |

## Validation Module

Takes an optimized `Morphology` and `Task`, builds a Newton scene, runs collision detection,
and returns a `ValidationResult` with self-collision and environment-collision counts.
`render_scene` opens a browser-based Newton viewer on the given port.
```

## Running

```bash
python run_validation.py clean         # no collisions expected
python run_validation.py self_collide  # self-collisions expected
```