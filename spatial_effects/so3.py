from typing import Sequence
from math import sin, cos

import numpy as np

from .common import in_so3, EPSILON, skew
from .conversions import so3_to_rvec


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


def so3_jacobian(rot_vec: np.ndarray) -> np.ndarray:
    """Compute Jacobian from rot_vec for conversion between SE(3) matrix and tangent
    representations. For [[R, t], [0, 1]] <==> [rho, rot_vec].T,  t = J rho.

    References:
    -----------

    http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17.pdf, section 7.1.3
    https://arxiv.org/abs/1812.01537, eqs. 145, 174.
    """

    phi = np.linalg.norm(rot_vec)  # angle about rotation axis

    J = np.eye(3)
    if abs(phi) < EPSILON:
        return J

    S = skew(rot_vec)

    J += ((1 - cos(phi)) / (phi * phi)) * S + (
        (phi - sin(phi)) / (phi * phi * phi)
    ) * S @ S

    return J


def inv_so3_jacobian(rot_vec: np.ndarray) -> np.ndarray:
    """Compute Jacobian from rot_vec for conversion between SE(3) matrix and tangent
    representations. For [[R, t], [0, 1]] <==> [rho, rot_vec].T,  rho = inv(J) t.

    References:
    -----------

    http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17.pdf, section 7.1.3
    https://arxiv.org/abs/1812.01537, eq. 146.
    """

    phi = np.linalg.norm(rot_vec)  # angle about rotation axis

    J = np.eye(3)
    if abs(phi) < EPSILON:
        return J

    S = skew(rot_vec)

    J += -0.5 * S + (1 / phi / phi - (1 + cos(phi)) / (2 * phi * sin(phi))) * S @ S

    return J
