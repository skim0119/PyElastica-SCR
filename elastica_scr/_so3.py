import numpy as np
from numpy.typing import NDArray
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


@njit(cache=True)  # type: ignore
def _compute_relative_rotation(
    Q_old: NDArray[np.float64],
    Q_new: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute rotation axes between two director frames using Rodrigues' formula.
    Assuming Q_old is the old director frame and Q_new is the new director frame,
    the function returns the rotation axis and magnitude that transforms Q_old to Q_new.

    Parameters
    ----------
    Q_old : numpy.ndarray
        The old director frame, of shape (dim, dim).
    Q_new : numpy.ndarray
        The new director frame, of shape (dim, dim).

    Returns
    -------
    vector_collection : numpy.ndarray
        The rotation axis, of shape (dim,).

    Notes
    -----
    Vector should be close to length 1 unless there is no rotation.
    """
    blocksize = Q_old.shape[2]
    vector_collection = np.empty((3, blocksize))

    for k in range(blocksize):
        # Q_{i+i}Q^T_{i} collection
        vector_collection[0, k] = (
            Q_new[2, 0, k] * Q_old[1, 0, k]
            + Q_new[2, 1, k] * Q_old[1, 1, k]
            + Q_new[2, 2, k] * Q_old[1, 2, k]
        ) - (
            Q_new[1, 0, k] * Q_old[2, 0, k]
            + Q_new[1, 1, k] * Q_old[2, 1, k]
            + Q_new[1, 2, k] * Q_old[2, 2, k]
        )

        vector_collection[1, k] = (
            Q_new[0, 0, k] * Q_old[2, 0, k]
            + Q_new[0, 1, k] * Q_old[2, 1, k]
            + Q_new[0, 2, k] * Q_old[2, 2, k]
        ) - (
            Q_new[2, 0, k] * Q_old[0, 0, k]
            + Q_new[2, 1, k] * Q_old[0, 1, k]
            + Q_new[2, 2, k] * Q_old[0, 2, k]
        )

        vector_collection[2, k] = (
            Q_new[1, 0, k] * Q_old[0, 0, k]
            + Q_new[1, 1, k] * Q_old[0, 1, k]
            + Q_new[1, 2, k] * Q_old[0, 2, k]
        ) - (
            Q_new[0, 0, k] * Q_old[1, 0, k]
            + Q_new[0, 1, k] * Q_old[1, 1, k]
            + Q_new[0, 2, k] * Q_old[1, 2, k]
        )

        trace = (
            (
                Q_new[0, 0, k] * Q_old[0, 0, k]
                + Q_new[0, 1, k] * Q_old[0, 1, k]
                + Q_new[0, 2, k] * Q_old[0, 2, k]
            )
            + (
                Q_new[1, 0, k] * Q_old[1, 0, k]
                + Q_new[1, 1, k] * Q_old[1, 1, k]
                + Q_new[1, 2, k] * Q_old[1, 2, k]
            )
            + (
                Q_new[2, 0, k] * Q_old[2, 0, k]
                + Q_new[2, 1, k] * Q_old[2, 1, k]
                + Q_new[2, 2, k] * Q_old[2, 2, k]
            )
        )

        # Clip the trace to between -1 and 3.
        # Any deviation beyond this is numerical error
        trace = min(trace, 3.0)
        trace = max(trace, -1.0)
        theta = np.arccos(0.5 * trace - 0.5) + 1e-14
        magnitude = -0.5 * theta / np.sin(theta)
        vector_collection *= magnitude

    return vector_collection
