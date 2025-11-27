import matplotlib.pyplot as plt
from typing import Tuple

class Contact:
    def __init__(self, x: float, y: float, shape: Tuple[float, float], friction: float):
        self.x = x
        self.y = y
        self.shape = shape
        self.friction = friction
        self.z_max = [x + shape[0] / 2, y + shape[1] / 2]
        self.z_min = [x - shape[0] / 2, y - shape[1] / 2]

def generate_footsteps(distance, step_length, foot_spread):
    contacts = []

    def append_contact(x, y):
        contact = Contact(x, y, (0.11, 0.05), 0.7)
        contacts.append(contact)
        
    append_contact(0., -foot_spread)
    append_contact(0., +foot_spread)
    x = 0.
    y = foot_spread
    while x < distance:
       if distance - x <= step_length:
           x += min(distance - x, 0.5 * step_length)
       else:
           x += step_length
       y = -y
       append_contact(x, y)
    append_contact(x, -y)
    return contacts

if __name__ == "__main__":
    traj = generate_footsteps(
        distance=2.1,
        step_length=0.3,
        foot_spread=0.1,
    )
    fig, ax = plt.subplots()
    for contact in traj:
        x, y = contact.x, contact.y
        w, h = contact.shape
        rect = plt.Rectangle((x - w/2, y - h/2), w, h, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
    X = [contact.x for contact in traj]
    Y = [contact.y for contact in traj]
    ax.scatter(X, Y, color='r', s=0.2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Footsteps (rectangles centered on contacts)")
    ax.set_aspect('equal')
    plt.show()