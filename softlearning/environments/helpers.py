import numpy as np


def random_point_in_circle(angle_range=(0, 2*np.pi), radius=(0, 25)):
    angle = np.random.uniform(*angle_range)
    radius = radius if np.isscalar(radius) else np.random.uniform(*radius)
    x, y = np.cos(angle) * radius, np.sin(angle) * radius
    point = np.array([x, y])
    return point
