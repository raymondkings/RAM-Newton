import newton
import warp as wp
import torch


def check_collisions(model, state) -> dict:
    contacts = model.collide(state)
    n_contacts = int(contacts.rigid_contact_count.numpy()[0])
    
    if n_contacts == 0:
        return {"n_self_collisions": 0, "n_env_collisions": 0, "n_total": 0}
    
    shape_body = model.shape_body.numpy()
    shape_labels = model.shape_label   # ← add this
    shape0 = contacts.rigid_contact_shape0.numpy()[:n_contacts]
    shape1 = contacts.rigid_contact_shape1.numpy()[:n_contacts]
    
    # DEBUG
    print(f"Total contacts: {n_contacts}")
    for i in range(n_contacts):
        b0 = shape_body[shape0[i]]
        b1 = shape_body[shape1[i]]
        print(f"  contact {i}: shape {shape0[i]} (body {b0}) vs shape {shape1[i]} (body {b1})")
    
    n_self = 0
    n_env  = 0
    for i in range(n_contacts):
        body_a = shape_body[shape0[i]]
        body_b = shape_body[shape1[i]]
        if body_a == -1 or body_b == -1:
            n_env += 1
        else:
            n_self += 1
    
    return {"n_self_collisions": n_self, "n_env_collisions": n_env, "n_total": n_contacts}