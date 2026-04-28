import newton


def check_collisions(model, state, debug: bool = False) -> dict:
    contacts = model.collide(state)
    n_contacts = int(contacts.rigid_contact_count.numpy()[0])

    if n_contacts == 0:
        return {"n_self_collisions": 0, "n_env_collisions": 0, "n_total": 0}

    shape_body = model.shape_body.numpy()
    shape0 = contacts.rigid_contact_shape0.numpy()[:n_contacts]
    shape1 = contacts.rigid_contact_shape1.numpy()[:n_contacts]

    n_self = 0
    n_env = 0

    for i in range(n_contacts):
        body_a = shape_body[shape0[i]]
        body_b = shape_body[shape1[i]]

        is_env = body_a == -1 or body_b == -1

        if debug:
            kind = "env" if is_env else "self"
            print(f"  [{kind}] shape {shape0[i]} (body {body_a}) "
                  f"vs shape {shape1[i]} (body {body_b})")

        if is_env:
            n_env += 1
        else:
            n_self += 1

    return {"n_self_collisions": n_self, "n_env_collisions": n_env, "n_total": n_contacts}