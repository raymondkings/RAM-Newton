import sys
from validation_module import (
    validate,
    render_scene,
    make_task1,
    make_test_morphology,
    make_self_colliding_morphology,
)

MODES = {
    "clean": make_test_morphology,
    "self_collide": make_self_colliding_morphology,
}

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "clean"

    if mode not in MODES:
        print(f"Unknown mode '{mode}'. Choose from: {list(MODES)}")
        sys.exit(1)

    task = make_task1()
    morph = MODES[mode]()
    debug = mode == "self_collide"

    print(f"\n=== {mode} ===")
    result = validate(morph, task, debug=debug)
    print(result)

    render_scene(morph, task)