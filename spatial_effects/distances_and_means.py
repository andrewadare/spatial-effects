from typing import Sequence, Optional, Union

import numpy as np

from .common import in_so3
from .conversions import so3_to_vector

__all__ = (
    "so3_angular_distance",
    "so3_chordal_distance",
    "quaternion_distance",
    "so3_chordal_l2_mean",
    "quaternion_mean",
)


def so3_angular_distance(R1: np.ndarray, R2: np.ndarray) -> float:
    """Angular or geodesic distance between two SO(3) rotations.

    Computed as the L2 norm of vec(log(R1.T @ R2)).
    """
    assert in_so3(R1)
    assert in_so3(R2)
    r = so3_to_vector(R1.T @ R2)
    return np.linalg.norm(r)


def so3_chordal_distance(R1: np.ndarray, R2: np.ndarray) -> float:
    """Euclidean or chordal distance between two SO(3) rotations.

    Computed as the Frobenius norm of (R1 - R2).
    """
    assert in_so3(R1)
    assert in_so3(R2)
    return np.linalg.norm(R1 - R2)


def quaternion_distance(a: np.ndarray, b: np.ndarray) -> Union[float, np.ndarray]:
    """Returns distances between arrays of normalized quaternions.

    Accounts for 2-fold degeneracy (q and -q perform the same rotation).

    Parameters
    ----------
    a, b : ndarray - shape (4,) or (N, 4)
        Quaternions in N rows

    Returns
    -------
    scalar or ndarray with shape (N,)
        Distance(s) between a and b.
    """
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    d2 = np.minimum(np.sum((a - b) ** 2, axis=1), np.sum((a + b) ** 2, axis=1))
    d = np.sqrt(d2)
    if d.shape == (1,):
        return d.item()
    else:
        return d


def so3_chordal_l2_mean(rotation_matrices: Sequence[np.ndarray]) -> np.ndarray:
    """Returns the mean of a set of rotation matrices as the matrix
    with minimum chordal L2 distance to all matrices in the set.

    Reference
    ---------
    Richard Hartley, Jochen Trumpf, Yuchao Dai, and Hongdong Li. 2013. Rotation averaging.
    International journal of computer vision 103, 3 (2013), 267-305.
    """
    for i, R in enumerate(rotation_matrices):
        if not in_so3(R):
            print(f"Element {i} not in SO(3):\n{R}")

    U, s, VT = np.linalg.svd(sum(rotation_matrices))
    UVT = U @ VT

    if np.linalg.det(UVT) >= 0.0:
        return UVT

    return U @ np.diag([1, 1, -1.0]) @ VT


def quaternion_mean(
    qs: np.ndarray, weights: Optional[Sequence[float]] = None
) -> np.ndarray:
    """Compute the weighted mean over unit quaternions.

    Parameters
    ----------
    qs : ndarray - shape (N, 4)
        Array of unit quaternions
    weights : ndarray - shape (N,) or (1, N)
        Optional weight factors for each quaternion

    Returns
    -------
    Normalized orientation quaternion as ndarray with shape (4,).

    Reference
    ---------
    "Averaging Quaternions" by F.L. Markley et al., Journal of Guidance,
    Control, and Dynamics 30, no. 4 (2007): 1193-1197.
    """
    if weights is not None:
        w = np.atleast_2d(weights).T  # w should be Nx1
        Q = (w * qs).T
    else:
        Q = qs.T

    vals, vecs = np.linalg.eig(Q @ Q.T)  # Not sorted!
    vals = np.real_if_close(vals)

    return np.real_if_close(vecs[:, np.argmax(vals)])
