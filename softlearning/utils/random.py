import numpy as np


def spherical(size=(), ndim=2):
    size = np.atleast_1d(size)
    random_normal = np.random.standard_normal((ndim, *size))
    normalized = random_normal / np.linalg.norm(random_normal, axis=0)
    return normalized
