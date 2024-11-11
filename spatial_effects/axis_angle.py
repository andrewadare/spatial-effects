from math import pi

import numpy as np

from .common import reshape_nx3


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


def unique_rvec(r: np.ndarray):
    """Flip sign of Rodrigues vector(s) r if necessary to maintain a
    unique rotation representation. r can be a single vector with shape
    (3,) or an array of N vectors with shape (N, 3).

    References
    ----------
    https://courses.cs.duke.edu/fall13/compsci527/notes/rodrigues.pdf
    """
    r = np.atleast_2d(r)
    assert r.shape[1] == 3

    theta = np.linalg.norm(r, axis=1)
    eps = np.finfo(float).eps
    for i in np.argwhere(theta == pi).ravel():
        rx, ry, rz = r[i]
        if np.any(
            [
                rx < 0,
                abs(rx) < eps and ry < 0,
                abs(rx) < eps and abs(ry) < eps and rz < 0,
            ]
        ):
            r[i] = -r[i]

    return r.squeeze()


def rrand(n: int = 1) -> np.ndarray:
    """Sample `n` random rotation vectors from the upper half of a ball with
    radius pi.

    Parameters
    ----------
    n: Number of rotation vectors to generate

    Returns
    -------
    array containing n random axis-angle vectors normalized to random rotation angles
    """

    # Get random rotation axes by sampling from an isotropic Gaussian.
    # Constrain directions to north hemisphere.
    vecs = np.random.randn(n, 3)
    south = vecs[:, 2] < 0
    vecs[south, 2] = -vecs[south, 2]
    norms = np.linalg.norm(vecs, axis=1)

    # Resample any points that are too close to zero for safe normalization
    shorties = np.isclose(norms, 0.0)
    while np.any(shorties):
        vecs[shorties] = np.random.randn(np.count_nonzero(shorties), 3)
        norms[shorties] = np.linalg.norm(vecs[shorties], axis=1)
        shorties = np.isclose(norms, 0.0)

    # Normalize to a random value in [0, pi)
    vecs = pi * np.random.uniform(size=[n, 1]) * vecs / norms[:, np.newaxis]

    if n == 1:
        vecs = vecs.ravel()

    return vecs
