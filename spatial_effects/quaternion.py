"""Functions for unit Hamilton quaternions
"""
from typing import Sequence, Optional, Union
from math import sin, cos, pi

import numpy as np

from .common import in_mpi_pi


def normalize(q):
    if q.ndim == 1:
        n = qnorm(q)
        if n > 0:
            return q / n
        else:
            return qnull()
    elif q.ndim == 2:
        norms = qnorm(q)

        bad_rows = np.where(norms == 0)
        for i in bad_rows:
            q[i] = qnull()
        norms[norms == 0] = 1.0
        return q / norms[:, np.newaxis]
    else:
        raise ValueError(f"Invalid shape: {q.shape}")


def _vector_quaternion(v):
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
        p = _vector_quaternion(x)  # prepend column of zeros
        x4 = np.atleast_2d(qmult(q, qmult(p, qinv(q))))
        return x4[:, 1:].squeeze()
    elif (x.ndim == 1 and x.shape[0] == 4) or (x.ndim == 2 and x.shape[1] == 4):
        # We are rotating a unit quaternion
        p = x
        x4 = np.atleast_2d(qmult(q, qmult(p, qinv(q))))
        return x4.squeeze()
    else:
        raise ValueError(f"Invalid array shape(s) x {x.shape}, q {q.shape}")


def qinv(q):
    """Return the inverse or conjugate of a quaternion or array of quaternions.

    Parameters
    ----------
    q : ndarray - shape (4,) or (N, 4)
    """
    validate(q)

    if q.ndim == 1 and q.shape[0] == 4:
        return np.array([q[0], *-q[1:]])
    elif q.ndim == 2 and q.shape[1] == 4:
        return np.hstack([q[:, :1], -q[:, 1:]])
    else:
        raise ValueError(f"Incorrect shape: {q.shape}")


def qplus(q, w):
    """Implements the "boxplus" quaternion displacement operator q ⊞ w.

    Parameters
    ----------
    q : ndarray (4,) or (N, 4)
        Unit quaternion
    w : ndarray (3,) or (N, 3)
        Axis-angle rotation vector

    Returns
    -------
    ndarray (4,) or (N, 4)
        Unit quaternion

    References
    ----------
    See eq. (29) in "Integrating Generic Sensor Fusion Algorithms with Sound State
    Representations through Encapsulation of Manifolds"
    by C. Hertzberg et al (arXiv:1107.1119v1 [cs.RO] 6 Jul 2011)

    """
    q = np.asarray(q)
    w = np.asarray(w)

    return qmult(q, expq(0.5 * w))


def qdiff(q, p):
    """Implements the "boxminus" inverse quaternion displacement operator q ⊟ p

    Parameters
    ----------
    q, p : ndarray (4,) or (N, 4)
        Unit quaternions

    Returns
    -------
    ndarray (3,) or (N, 3)
        Rotation vector

    References
    ----------
    See eq. (29) in "Integrating Generic Sensor Fusion Algorithms with Sound State
    Representations through Encapsulation of Manifolds"
    by C. Hertzberg et al (arXiv:1107.1119v1 [cs.RO] 6 Jul 2011)
    """
    q = np.asarray(q)
    p = np.asarray(p)

    if q.shape != p.shape:
        raise ValueError(f"Shape mismatch: q {q.shape} vs. p {p.shape}")

    return 2 * logq(qmult(qinv(p), q))


def qleft(q):
    """Returns matrix or matrices for efficiently implementing quaternion product(s)
    as a left matrix-vector multiplication. If q is a single quaternion, a 4x4 matrix
    is returned. If q is an Nx4 array of quaternions, an (N, 4, 4) array is returned.

    Parameters
    ----------
    q : ndarray - shape (4,) or shape (N, 4)

    Returns
    -------
    Q : ndarray - shape (4, 4) or shape (N, 4, 4)

    Example
    -------
    q1, q2 = qrand(2)
    qmult(q1, q2) == qleft(q1) @ q2  # True
    qleft(q1) @ q2 == qright(q2) @ q1  # True
    """
    q = np.atleast_2d(q)

    Q = np.array(
        [
            [+q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]],
            [+q[:, 1], +q[:, 0], -q[:, 3], +q[:, 2]],
            [+q[:, 2], +q[:, 3], +q[:, 0], -q[:, 1]],
            [+q[:, 3], -q[:, 2], +q[:, 1], +q[:, 0]],
        ]
    ).transpose((2, 0, 1))

    return Q.squeeze()


def qright(q):
    """Returns matrix or matrices for efficiently implementing quaternion product(s)
    as a right matrix-vector multiplication. If q is a single quaternion, a 4x4 matrix
    is returned. If q is an Nx4 array of quaternions, an (N, 4, 4) array is returned.

    Parameters
    ----------
    q : ndarray - shape (4,) or shape (N, 4)

    Returns
    -------
    Q : ndarray - shape (4, 4) or shape (N, 4, 4)

    Example
    -------
    q1, q2 = qrand(2)
    qmult(q1, q2) == qleft(q1) @ q2  # True
    qleft(q1) @ q2 == qright(q2) @ q1  # True
    """
    q = np.atleast_2d(q)

    Q = np.array(
        [
            [+q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]],
            [+q[:, 1], +q[:, 0], +q[:, 3], -q[:, 2]],
            [+q[:, 2], -q[:, 3], +q[:, 0], +q[:, 1]],
            [+q[:, 3], +q[:, 2], -q[:, 1], +q[:, 0]],
        ]
    ).transpose((2, 0, 1))

    return Q.squeeze()


def qmult(a, b):
    """Multiplication op for unit quaternions, vectorized if a and b contain
    multiple quaternions in rows.

    Parameters
    ----------
    a, b : ndarray - either or both can have shape (N, 4) or shape (4,)
        Quaternions in N rows

    Returns
    -------
    q : ndarray - shape (N, 4) or shape (4,)
        Quaternion product(s)
    """
    assert isinstance(a, np.ndarray) and isinstance(
        b, np.ndarray
    ), "arguments must be numpy arrays. Received {} and {}.".format(type(a), type(b))

    if a.ndim == 1 and b.ndim == 1:
        # Single quaternions were provided
        av, bv = a[1:], b[1:]
        if len(av) != len(bv):
            raise ValueError("Length mismatch: {} vs {}".format(len(av), len(bv)))
        q0 = a[0] * b[0] - np.dot(av, bv)  # scalar component q_0
        qv = a[0] * bv + b[0] * av + np.cross(av, bv)  # vector component q_1 - q_3
        return np.array([q0, *qv])
    elif a.ndim == 1 and b.ndim == 2 and a.shape[0] == 4 and b.shape[1] == 4:
        A = qleft(a)  # (4,4)
        return np.matmul(A, b.T).T  # (4,4) x (N,4)' -> (4,N)'
    elif a.ndim == 2 and b.ndim == 1 and a.shape[1] == 4 and b.shape[0] == 4:
        A = qleft(a)  # (N,4,4)
        return np.matmul(A, b)  # (N,4,4) x (4,) -> (N,4)
    elif (
        a.ndim == 2
        and b.ndim == 2
        and a.shape[1] == 4
        and b.shape[1] == 4
        and a.shape[0] == b.shape[0]
    ):
        return _qmult_vectorized(a, b)
    else:
        raise ValueError(f"Invalid array shape(s) x {a.shape}, q {b.shape}")


def _qmult_vectorized(a, b):
    """Vectorized multiplication op for 2d arrays of unit quaternions,
    which are expected to be contained in the rows.

    Parameters
    ----------
    a, b : ndarray - shape (N, 4)
        Quaternions in N rows
    """
    a0, b0 = a[:, :1], b[:, :1]  # N x 1
    av, bv = a[:, 1:], b[:, 1:]  # N x 3

    # scalar components
    q0 = a0 * b0 - np.sum(av * bv, axis=1)[:, np.newaxis]

    # vector components
    qv = a0 * bv + b0 * av + np.cross(av, bv)

    return np.hstack([q0, qv])


def validate(x):
    assert isinstance(x, np.ndarray), f"Not of type ndarray: {type(x)}"
    assert x.ndim == 1 or x.ndim == 2, f"Incorrect rank: {x.ndim}"


def q_axis(q):
    validate(q)
    vec_norms = q_vec_norm(q)

    # Handle case(s) where vector norm == 0.
    # Set to 1 so that we will have zeros/0 -> zeros/1 below.
    if np.isscalar(vec_norms):
        if vec_norms == 0:
            vec_norms = 1.0
    else:
        vec_norms[vec_norms == 0] = 1.0

    if q.ndim == 1:
        return q[1:4] / vec_norms
    else:
        return q[:, 1:4] / vec_norms


def q_angle(q):
    validate(q)
    if q.ndim == 1:
        return in_mpi_pi(2.0 * np.arctan2(q_vec_norm(q), q[0]))
    else:
        return in_mpi_pi(2.0 * np.arctan2(q_vec_norm(q), q[:, :1]))


def expq(v):
    """Apply an exponential mapping from a rotation vector v to a unit
    quaternion q, using the correspondence between Lie algebras and Matrix Lie
    groups.

    Note: the user should call q = expq(w/2) if ||w|| is the rotation angle
    about w.

    Parameters
    ----------
    v : ndarray - shape (3,), (3, 1) or (N, 3)
        Rodrigues/axis-angle/rotation vector(s) with norm(s) encoding rotation angle

    Returns
    -------
    q : ndarray - shape (4,)
        Unit quaternion
    """
    validate(v)
    if np.array(v).ndim == 1:
        if v.size != 3:
            raise ValueError("v not a 3-vector. Received {}".format(v))
        theta = np.linalg.norm(v)
        if theta == 0:
            return qnull()
        qv = sin(theta) / theta * np.array(v)
        return np.array([cos(theta), *qv])
    elif isinstance(v, np.ndarray) and v.ndim == 2:
        if v.shape == (3, 1):
            v = v.T
        if v.shape[1] == 3:
            result = np.zeros([v.shape[0], 4])
            thetas = np.linalg.norm(v, axis=1)
            result[:, 0] = np.cos(thetas)
            nulls = thetas == 0  # mask for rows with zero norm. shape (N,)
            nz = thetas[~nulls]
            result[~nulls, 1:] = (np.sin(nz) / nz)[:, np.newaxis] * v[~nulls]
            return result
    else:
        raise ValueError(f"Invalid v: {v}")


def expq_approx(v):
    """Like expq (see those docs), but assumes the rotation angle is small such
    that cos(x) ~ 1 and sin(x) ~ x."""
    return np.array(1, *v)


def logq(q):
    """Inverse of expq(). Maps quaternions to 3-vectors."""
    return 0.5 * q_angle(q) * q_axis(q)


def qnorm(q):
    """2-norm of quaternion or array of quaternions. Returns a scalar or 1d
    array. Should be 1.0 or very nearly so for unit quaternions."""
    norms_squared = np.sum(np.atleast_2d(q) ** 2, axis=1)
    if q.ndim == 1:
        return np.sqrt(norms_squared[0])
    elif q.ndim == 2:
        return np.sqrt(norms_squared)
    else:
        raise ValueError(f"q should be 1 or 2 dimensional. q.shape: {q.shape}")


def qnull(*args):
    """Return an identity quaternion or array of identity quaternions, i.e. one or more
    quaternions of the form [1, 0, 0, 0].

    Parameters
    ----------
    n : int (optional)
        If provided, return an array with shape (n, 4). If not provided,
        return array with shape (4,).
    """
    if len(args) == 0:
        n = 1
    elif len(args) == 1:
        n = args[0]
    else:
        raise ValueError("qnull accepts zero or one arguments.")
    q = np.zeros([n, 4])
    q[:, 0] = 1.0
    return q.squeeze()


def qrand(*args):
    """Generate array of n uniform random unit quaternions.

    Parameters
    ----------
    n : int (optional)
        If provided, return an array with shape (n, 4). If not provided,
        return array with shape (4,).
    """

    if len(args) == 0:
        n = 1
    elif len(args) == 1:
        n = args[0]
    else:
        raise ValueError("qrand accepts zero or one arguments.")

    rs = np.random.random_sample((n, 3))
    a, b, c = rs[:, 0], rs[:, 1], rs[:, 2]
    d, e = 2 * pi * b, 2 * pi * c
    q = [
        np.sqrt(1.0 - a) * (np.sin(d)),
        np.sqrt(1.0 - a) * (np.cos(d)),
        np.sqrt(a) * (np.sin(e)),
        np.sqrt(a) * (np.cos(e)),
    ]
    return np.array(q).T.squeeze()


def q_vec_norm(q):
    validate(q)
    if q.ndim == 1:
        return np.linalg.norm(q[1:4])
    else:
        return np.linalg.norm(q[:, 1:4], axis=1)[:, np.newaxis]  # Nx1


def lerp(p, q, alpha):
    """Weighted linear interpolation between quaternions:

    (1 - alpha)p + alpha*q

    Parameters
    ----------
    p, q : ndarray - shape (4,)
    alpha : float
        weight factor in 0 <= alpha <= 1
    """
    if not 0 <= alpha <= 1.0:
        raise ValueError("0 <= alpha <= 1 required. Received {}".format(alpha))

    q_int = (1 - alpha) * p + alpha * q  # interpolated
    return q_int / qnorm(q_int)


def slerp(p, q, alpha):
    """Weighted spherical linear interpolation between quaternions:

    sin((1 - alpha)*a)/sin(a)*p + sin(alpha*a)/sin(a)*q

    where a is the angle between p and q.

    Parameters
    ----------
    p, q : ndarray - shape (4,)
    alpha : float
        weight factor in 0 <= alpha <= 1
    """
    if not 0 <= alpha <= 1.0:
        raise ValueError("0 <= alpha <= 1 required. Received {}".format(alpha))

    a = 0.5 * angle_between_quaternions(p, q)

    if a < 1e-12:
        return p
    q_int = sin((1 - alpha) * a) / sin(a) * p + sin(alpha * a) / sin(a) * q
    return q_int / qnorm(q_int)


def angle_between_quaternions(p: np.ndarray, q: np.ndarray) -> Union[float, np.ndarray]:
    """Compute unsigned angle(s) `a` between quaternions in Euclidean 4-space, handling
    SO(3) double cover degeneracy. In other words, ensure the angle between each pair is
    in 0 <= a <= pi. Symmetric under p, q swap.

    Parameters
    ----------
    p, q : ndarray - shape (4,) or (N, 4)
        Quaternions in N rows. Sequences or nested sequences are also valid if they
        can be cast to 2D ndarrays with 4 columns. For performance, unit normalization
        is not enforced here. It is up to the user to provide normalized quaternions.

    Returns
    -------
    a : float or ndarray
        Angle or array of angles in [0, pi]
    """
    p = np.atleast_2d(p)
    q = np.atleast_2d(q)

    if p.shape[1] != 4:
        raise ValueError("Incorrect shape for p: {}".format(p.shape))
    if q.shape[1] != 4:
        raise ValueError("Incorrect shape for q: {}".format(q.shape))
    if p.shape[0] != q.shape[0]:
        raise ValueError("Row count mismatch: {} vs {}".format(p.shape, q.shape))

    dots = np.sum(p * q, axis=1)  # scalar product(s)

    # This is intended to handle small numeric errors, like p @ q = +/-(1 + epsilon).
    dots = np.clip(dots, -1.0, +1.0)

    a = np.arccos(dots)

    # Don't be obtuse.
    obtuse = np.cos(a) < 0
    a[obtuse] = np.arccos(-np.sum(p[obtuse] * q[obtuse], axis=1))

    a *= 2  # half-angles --> angles

    if len(a) == 1:
        a = a.item()

    return a


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
