from math import pi

import numpy as np

from .common import reshape_nx3
from .quaternion import qmult, qinv


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


def qrotate(x, q):
    """Rotate x by q: x_rot = q*p*inv(q) where p is a unit quaternion from x. If
    x is a unit quaternion, then p = x.

    x can be a:
        - 3-vector (shape (3,))
        - array of N 3-vectors (N, 3)
        - unit quaternion (4,)
        - array of N unit quaternions (N, 4)
    and q must be a unit quaternion (shape (4,)) or array of unit quaternions
    (shape (N, 4)).

    Parameters
    ----------
    x : ndarray
    q : ndarray

    Returns
    -------
    ndarray
        rotated x
    """
    x = np.asarray(x)

    # Validate that q.shape is (4,) or (N, 4).
    if (q.ndim == 1 and q.shape[0] != 4) or (q.ndim == 2 and q.shape[1] != 4):
        raise ValueError(f"Invalid quaternion dimensions: {q.shape}")

    if (x.ndim == 1 and x.shape[0] == 3) or (x.ndim == 2 and x.shape[1] == 3):
        # We are rotating a vector or vectors
        p = vector_quaternion(x)  # prepend column of zeros
        x4 = np.atleast_2d(qmult(q, qmult(p, qinv(q))))
        return x4[:, 1:].squeeze()
    elif (x.ndim == 1 and x.shape[0] == 4) or (x.ndim == 2 and x.shape[1] == 4):
        # We are rotating a unit quaternion
        p = x
        x4 = np.atleast_2d(qmult(q, qmult(p, qinv(q))))
        return x4.squeeze()
    else:
        raise ValueError(f"Invalid array shape(s) x {x.shape}, q {q.shape}")
    # This also works, but is slower:
    # p = expq(x/2)
    # x4 = qmult(q, qmult(p, qinv(q)))
    # return 2*logq(x4)


def vector_quaternion(v):
    """Return quaternion(s) from 3-vector(s) v by prepending a zero or a column
    of zeros. Vector quaternions have a scalar component of zero and are used
    in rotations.

    Parameters
    ----------
    v : ndarray - shape (3,) or (N, 3)

    Returns
    -------
    ndarray - shape (4,) or (N, 4)
    """
    v2 = np.atleast_2d(v)
    nrows, ncols = v2.shape
    assert ncols == 3, "v should be a 3-vector or array of 3-vectors."
    vq = np.hstack([np.zeros([nrows, 1]), v2])
    if v.ndim == 1:
        return vq.squeeze()
    else:
        return vq


def rrand(*args):
    """Generate a random rotation vector on the unit sphere whose norm
    is a rotation angle in [0, 2pi).
    """
    if len(args) == 0:
        n = 1
    elif len(args) == 1:
        assert isinstance(args[0], int), (
            "Argument must be an integer number of rotation vectors"
            f" to generate. Received type {type(args[0])}"
        )
        n: int = args[0]
    else:
        raise ValueError("rrand accepts zero or one arguments.")

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

    if len(args) == 0:
        vecs = vecs.ravel()

    return vecs
