import numpy as np

def random_point_on_circle(angle_range=(0, 2*np.pi), radius=5):
    angle = np.random.uniform(*angle_range)
    x, y = np.cos(angle) * radius, np.sin(angle) * radius
    point = np.array([x, y])
    return point
