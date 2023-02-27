from typing import Sequence

import numpy as np

from .common import in_so3
from .conversions import so3_to_rvec

__all__ = (
    "so3_angular_distance",
    "so3_chordal_distance",
    "so3_chordal_l2_mean",
)


def so3_angular_distance(R1: np.ndarray, R2: np.ndarray) -> float:
    """Angular or geodesic distance between two SO(3) rotations.

    Computed as the L2 norm of vec(log(R1.T @ R2)).
    """
    assert in_so3(R1)
    assert in_so3(R2)
    r = so3_to_rvec(R1.T @ R2)
    return np.linalg.norm(r)


def so3_chordal_distance(R1: np.ndarray, R2: np.ndarray) -> float:
    """Euclidean or chordal distance between two SO(3) rotations.

    Computed as the Frobenius norm of (R1 - R2).
    """
    assert in_so3(R1)
    assert in_so3(R2)
    return np.linalg.norm(R1 - R2)


def so3_chordal_l2_mean(rotation_matrices: Sequence[np.ndarray]) -> np.ndarray:
    """Returns the mean of a set of rotation matrices as the matrix
    with minimum chordal L2 distance to all matrices in the set.

    References
    ----------
    Richard Hartley, Jochen Trumpf, Yuchao Dai, and Hongdong Li. 2013. Rotation averaging.
    International journal of computer vision 103, 3 (2013), 267-305.

    Luca Carlone, Roberto Tron, Kostas Daniilidis and Frank Dellaert. 2015. Initialization
    Techniques for 3D SLAM: a Survey on Rotation Estimation and its Use in Pose Graph
    Optimization. 2015 IEEE International Conference on Robotics and Automation (ICRA)},
    4597-4604.
    """
    for i, R in enumerate(rotation_matrices):
        if not in_so3(R):
            print(f"Element {i} not in SO(3):\n{R}")

    U, s, VT = np.linalg.svd(sum(rotation_matrices))

    return U @ np.diag([1, 1, np.linalg.det(U @ VT)]) @ VT
