import numpy as np
from numba import njit


@njit(cache=True)
def hat(v):
    """Skew-symmetric matrix"""
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


@njit(cache=True)
def exp_so3(theta):
    """Exponential map for SO(3)"""
    angle = np.linalg.norm(theta)
    if angle < 1e-10:
        return np.eye(3) + hat(theta)
    axis = theta / angle
    K = hat(axis)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
