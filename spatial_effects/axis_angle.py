from math import pi

import numpy as np

from .common import reshape_nx3

__all__ = (
    "rotate_axis_angle",
    "rrand",
)


def rotate_axis_angle(v, r):
    """Rotate a Euclidean 3-vector or array of 3-vectors v by an axis-angle vector r
    whose norm ||r||_2 is the rotation angle.

    Parameters
    ----------
    v : ndarray - shape (3,) or (N, 3)
        Vector(s) to be rotated
    r : iterable of size 3
        Rodrigues rotation vector

    Returns
    -------
    v_rotated : ndarray
        Rotated vector(s)


    Raises
    ------
    ValueError
        If the size of r is not 3.

    References
    ----------
    https://en.wikipedia.org/wiki/Axis–angle_representation#Rotating_a_vector
    """

    r = np.asarray(r)

    if r.size != 3:
        raise ValueError(f"r (size {r.size} should have size 3.")

    v, is_1d = reshape_nx3(v)
    r = r.reshape([1, 3])

    theta = np.linalg.norm(r)
    rhat = r / theta  # shape (1, 3)
    c, s = np.cos(theta), np.sin(theta)

    v_rotated = c * v + s * np.cross(rhat, v) + (1 - c) * (v @ rhat.T) * rhat

    if is_1d:
        return v_rotated.ravel()

    return v_rotated


def rrand(n: int = 1) -> np.ndarray:
    """Generate `n` random rotation vectors on the unit sphere whose norm
    is a rotation angle in [0, 2pi).

    Parameters
    ----------
    n: Number of rotation vectors to generate

    Returns
    -------
    array containing n random axis-angle vectors normalized to random rotation angles
    """

    # Get random rotation axes by sampling from an isotropic Gaussian
    vecs = np.random.randn(n, 3)
    norms = np.linalg.norm(vecs, axis=1)

    # Resample any points that are too close to zero for safe normalization
    shorties = np.isclose(norms, 0.0)
    while np.any(shorties):
        vecs[shorties] = np.random.randn(np.count_nonzero(shorties), 3)
        norms[shorties] = np.linalg.norm(vecs[shorties], axis=1)
        shorties = np.isclose(norms, 0.0)

    # Normalize to a random value in [0, 2pi)
    vecs = 2 * pi * np.random.uniform(size=[n, 1]) * vecs / norms[:, np.newaxis]

    if n == 1:
        vecs = vecs.ravel()

    return vecs
