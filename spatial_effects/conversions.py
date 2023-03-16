"""Coordinate conversions and transformations
"""
from math import isclose

import numpy as np

from .common import cross_product_matrix
from .quaternion import qleft, qright, normalize, q_angle, q_axis, expq
from .axis_angle import keep_north

__all__ = (
    "hc",
    "ic",
    "quaternion_to_so3",
    "so3_to_quaternion",
    "rvec_to_quaternion",
    "quaternion_to_rvec",
    "ypr_to_quaternion",
    "quaternion_to_ypr",
    "rvec_to_so3",
    "so3_to_rvec",
    "ypr_to_so3",
    "so3_to_ypr",
    "rodrigues",
)


def vector_to_quaternion(v):
    print(
        "vector_to_quaternion is deprecated in v0.2. Please use rvec_to_quaternion instead."
    )
    return rvec_to_quaternion(v)


def quaternion_to_vector(q):
    print(
        "quaternion_to_vector is deprecated in v0.2. Please use quaternion_to_rvec instead."
    )
    return quaternion_to_rvec(q)


def vector_to_so3(v):
    print(
        "vector_to_so3 is deprecated in v0.2. Please use rvec_to_so3 or rodrigues instead."
    )
    return rvec_to_so3(v)


def so3_to_vector(R):
    print(
        "so3_to_vector is deprecated in v0.2. Please use so3_to_rvec or rodrigues instead."
    )
    return so3_to_rvec(R)


def hc(x: np.ndarray) -> np.ndarray:
    """Convert an array of points (as row vectors) from inhomogeneous to
    homogeneous coordinates. The returned points are in column vectors for
    pre-multiplication with a matrix.

    Parameters
    ----------
    x : ndarray - shape (m, n)
        Array of m row vectors in inhomogeneous coords, each of dimensionality n

    Returns
    -------
    ndarray - shape (n+1, m)
    """
    x = np.atleast_2d(x)

    return np.vstack([x.T, np.ones([1, x.shape[0]])])


def ic(x: np.ndarray, nan_to_num=True) -> np.ndarray:
    """Convert an array of points in column vectors from homogeneous to
    inhomogeneous coordinates. The returned array contains points as row
    vectors.

    Parameters
    ----------
    x : ndarray - shape (n+1, m)
        Array of m vectors in homogeneous coords, each of dimensionality n+1

    Returns
    -------
    ndarray - shape (m, n)
    """
    if not isinstance(x, np.ndarray):
        raise ValueError(f"Expected numpy array, received {type(x)}")

    if x.size == 0:
        return x

    if x.ndim == 1:
        x = x[:, np.newaxis]

    with np.errstate(invalid="ignore"):
        x /= x[-1, :]

    x = x[:-1, :].T
    if nan_to_num:
        x = np.nan_to_num(x)  # np.inf -> large float; np.nan -> 0.0

    return x


def quaternion_to_so3(q) -> np.ndarray:
    """Return a 3x3 rotation matrix from a unit Hamilton quaternion q.
    q can be a numpy array, list, or tuple.

    If q is an Nx4 array of quaternions, an Nx3x3 array of matrices
    will be returned.

    q should follow a real-first component ordering layout.
    """

    Q = qleft(q)
    Qi = qright(q)

    if Q.shape == (4, 4) and Qi.shape == (4, 4):
        M = np.matmul(Q, Qi.T)
        return M[1:4, 1:4]
    elif q.ndim == 2:
        M = np.matmul(Q, Qi.transpose(0, 2, 1))
        return M[:, 1:4, 1:4]
    else:
        raise ValueError("Invalid shape: {}".format(q.shape))


def so3_to_quaternion(R):
    """Convert 3x3 orthogonal rotation matrix to quaternion. Calculation is
    based on the trace method as implemented in pyquaternion:

    https://github.com/KieranWynn/pyquaternion/blob/
    446c31cba66b708e8480871e70b06415c3cb3b0f/pyquaternion/quaternion.py#L205

    Not yet vectorized to handle (N,3,3) arrays of rotation matrices (TODO)

    Parameters
    ----------
    R : ndarray - shape (3, 3)

    Returns
    -------
    q : ndarray - shape (4,)
    """
    R = np.asarray(R)

    if R.ndim != 2:
        raise NotImplementedError(
            "Vectorized R -> quaternion conversion not yet supported."
        )

    R = R.T

    if R[2, 2] < 0:
        if R[0, 0] > R[1, 1]:
            t = 1 + R[0, 0] - R[1, 1] - R[2, 2]
            q = [R[1, 2] - R[2, 1], t, R[0, 1] + R[1, 0], R[2, 0] + R[0, 2]]
        else:
            t = 1 - R[0, 0] + R[1, 1] - R[2, 2]
            q = [R[2, 0] - R[0, 2], R[0, 1] + R[1, 0], t, R[1, 2] + R[2, 1]]
    else:
        if R[0, 0] < -R[1, 1]:
            t = 1 - R[0, 0] - R[1, 1] + R[2, 2]
            q = [R[0, 1] - R[1, 0], R[2, 0] + R[0, 2], R[1, 2] + R[2, 1], t]
        else:
            t = 1 + R[0, 0] + R[1, 1] + R[2, 2]
            q = [t, R[1, 2] - R[2, 1], R[2, 0] - R[0, 2], R[0, 1] - R[1, 0]]

    return 0.5 / np.sqrt(t) * np.array(q)


def rvec_to_quaternion(v):
    """Convert a rotation vector to a unit quaternion.

    Parameters
    ----------
    v : ndarray - shape (3,) or (N, 3)
        Rodrigues/axis-angle/rotation vector(s) whose norm encodes the rotation
        about its direction.

    Returns
    -------
    q : ndarray - shape (4,) or (N, 4)
        Unit quaternion(s)
    """
    v = np.asarray(v)
    if v.ndim == 1:
        assert v.size == 3, f"Invalid input: {v}"
    elif v.ndim == 2:
        assert v.size == 3 or v.shape[1] == 3, f"Invalid input: {v}"
    else:
        raise ValueError(f"Input rank must be 1 or 2: {v.ndim}")
    return expq(0.5 * v)


def quaternion_to_rvec(q):
    """Convert a unit quaternion to a rotation vector.

    Parameters
    ----------
    q : ndarray - shape (4,) or (N, 4)
        Unit quaternion(s)

    Returns
    -------
    v : ndarray - shape (3,) or (N, 3)
        Rodrigues/axis-angle/rotation vector(s) whose norm encodes the rotation
        about its direction.
    """
    q = np.asarray(q)
    if q.ndim == 1:
        assert q.size == 4, f"Invalid input: {q}"
    elif q.ndim == 2:
        assert q.size == 4 or q.shape[1] == 4, f"Invalid input: {q}"
    else:
        raise ValueError(f"Input rank must be 1 or 2: {q.ndim}")
    return q_angle(q) * q_axis(q)


def ypr_to_quaternion(ypr):
    """Convert array of (yaw, pitch, roll) values, either 1D for a single
    triplet or 2D for multiple triplets, to quaternion(s).

    Parameters
    ----------
    ypr : ndarray - shape (3,) or (N, 3)

    Returns
    -------
    ndarray - shape (4,) or (N, 4)
    """
    ypr = np.atleast_2d(ypr)
    assert ypr.shape[1] == 3
    y, p, r = ypr[:, 0], ypr[:, 1], ypr[:, 2]

    cy = np.cos(y / 2)
    cp = np.cos(p / 2)
    cr = np.cos(r / 2)
    sy = np.sin(y / 2)
    sp = np.sin(p / 2)
    sr = np.sin(r / 2)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return np.array([qw, qx, qy, qz]).T.squeeze()


def quaternion_to_ypr(q):
    q = normalize(q)
    q = np.atleast_2d(q)

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    rx = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2[t2 > 1.0] = 1.0
    t2[t2 < -1.0] = -1.0
    ry = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    rz = np.arctan2(t3, t4)

    return np.array([rz, ry, rx]).T.squeeze()


def rvec_to_so3(r: np.ndarray) -> np.ndarray:
    """Get rotation matrix from an axis-angle vector whose norm encodes rotation.

    Parameters
    ----------
    r : ndarray - shape (3,), (3, 1), or (1, 3)
        Rodrigues vector

    Returns
    -------
    R : ndarray - shape (3, 3)
        Rotation matrix
    """
    r = np.asarray(r)
    if r.size != 3:
        raise ValueError("Rotation vector has incorrect shape: {}".format(r.shape))
    r = r.ravel()
    theta = np.linalg.norm(r)
    if theta == 0:
        return np.eye(3)
    n = r / theta
    c, s = np.cos(theta), np.sin(theta)
    R = c * np.eye(3) + (1 - c) * np.outer(n, n) + s * cross_product_matrix(*n)
    return R


def so3_to_rvec(R: np.ndarray) -> np.ndarray:
    """Get an axis-angle vector from a rotation matrix.

    Parameters
    ----------
    R : ndarray - shape (3, 3)
        Rotation matrix

    Returns
    -------
    r : ndarray - shape (3,)
        Rodrigues vector

    Reference
    ---------
    "Vector Representation of Rotations" by Carlo Tomasi
    https://courses.cs.duke.edu/fall13/compsci527/notes/rodrigues.pdf
    """

    # Handle special case of symmetric R
    if np.allclose(R - R.T, np.zeros((3, 3))):
        c = (np.trace(R) - 1) / 2  # this is cos(theta)
        if c == 1:
            # theta is zero or even*pi
            return np.zeros(3)
        if c == -1:
            # theta is odd*pi
            # One way to find a nonzero column
            col = np.argmax(np.sum(R + np.eye(3), axis=0))
            v = R[:, col]
            return keep_north(np.pi * v / np.linalg.norm(v))

    a, b, c = R[0]
    d, e, f = R[1]
    g, h, i = R[2]

    # use tr(R) = 1 + 2*cos(theta)
    arg = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    theta = np.arccos(arg)

    u = np.array([h - f, c - g, d - b])
    u_norm = np.linalg.norm(u)
    if isclose(u_norm, 0.0):
        return np.zeros(3)

    return theta * u / u_norm


def ypr_to_so3(ypr):
    """Convert array of (yaw, pitch, roll) values to rotation matrices.

    Parameters
    ----------
    ypr : ndarray - shape (3,) or (N, 3)

    Returns
    -------
    ndarray - shape (3, 3) or (N, 3, 3)

    Examples
    --------
    # Single (y, p, r) array:
    ypr = quaternion_to_ypr(qrand())
    array([ 2.97870637, -1.11091078, -0.2134833 ])

    ypr_to_matrix(ypr)

    array([[-0.43797052, -0.34582584,  0.82981101],
           [ 0.07197708, -0.93357475, -0.35108046],
           [ 0.8961033 , -0.09403551,  0.43376975]])

    # Two yprs in a 2x3 array:
    ypr = quaternion_to_ypr(qrand(2))
    Rs = ypr_to_matrix(ypr); Rs.shape
    (2, 3, 3)

    References
    ----------
    https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
    See the table entry for "Z1 Y2 X3" in the column for Tait-Bryan angles.
    """
    ypr = np.atleast_2d(ypr)
    assert ypr.shape[1] == 3
    y, p, r = ypr[:, 0], ypr[:, 1], ypr[:, 2]

    c1 = np.cos(y)
    c2 = np.cos(p)
    c3 = np.cos(r)
    s1 = np.sin(y)
    s2 = np.sin(p)
    s3 = np.sin(r)

    R = np.array(
        [
            [c1 * c2, c1 * s2 * s3 - c3 * s1, s1 * s3 + c1 * c3 * s2],
            [c2 * s1, c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3],
            [-s2, c2 * s3, c2 * c3],
        ]
    )
    R = R.transpose((2, 0, 1))
    return R.squeeze()


def so3_to_ypr(R):
    """Convert rotation matrix or array of N matrices to array of yaw, pitch,
    roll Euler angles.

    Parameters
    ----------
    R: ndarray - shape (3, 3) or (N, 3, 3)

    Returns
    -------
    ypr : ndarray - shape (3,) or (N, 3)
    """
    if R.ndim == 2:
        R = R[np.newaxis, :, :]

    N = R.shape[0]
    ypr = np.empty([N, 3])

    s = np.sqrt(R[:, 0, 0] ** 2 + R[:, 1, 0] ** 2)

    singular = s < 1e-6
    ypr[singular, 0] = 0.0
    ypr[singular, 1] = np.arctan2(-R[singular, 2, 0], s[singular])
    ypr[singular, 2] = np.arctan2(-R[singular, 1, 2], R[singular, 1, 1])

    ns = ~singular
    ypr[ns, 0] = np.arctan2(R[ns, 1, 0], R[ns, 0, 0])
    ypr[ns, 1] = np.arctan2(-R[ns, 2, 0], s[ns])
    ypr[ns, 2] = np.arctan2(R[ns, 2, 1], R[ns, 2, 2])

    return ypr.squeeze()


def rodrigues(vector_or_matrix):
    """Computes a rotation matrix from a Rodrigues vector or vice versa. This is a
    convenience function similar to `cv2.Rodrigues`, but does not return a Jacobian.

    Parameters
    ----------
    vector_or_matrix : ndarray - size 3 or shape (3, 3)
        Rotation vector or matrix

    Returns
    -------
    ndarray - shape (3, 3) or (3,)
        Rotation matrix or vector, depending on input shape
    """
    vector_or_matrix = np.asarray(vector_or_matrix)

    if vector_or_matrix.size == 3:
        return rvec_to_so3(vector_or_matrix)
    elif vector_or_matrix.shape == (3, 3):
        return so3_to_rvec(vector_or_matrix)
    else:
        raise ValueError("Invalid input shape: {}".format(vector_or_matrix.shape))
