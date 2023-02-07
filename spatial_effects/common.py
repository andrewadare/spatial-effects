from enum import Enum, auto
from math import pi

import numpy as np


class Basis(Enum):
    """Enumeration constants for interpreting a coordinate triplet (x, y, z) in
    terms of egocentric or cardinal directions. The names refer to (x, y); their
    cross product defines the +z direction according to the right-hand rule.
    """

    FWD_LEFT = auto()  # z up (ROS REP 103)
    EAST_NORTH = auto()  # z up
    FWD_RIGHT = auto()  # z down (aeronautics)
    NORTH_EAST = auto()  # z down
    RIGHT_DOWN = auto()  # z fwd (camera/optical)
    RIGHT_UP = auto()  # z back (OpenGL)

    def to(self, other: "Basis"):
        """Returns a rotation matrix used to reinterpret coordinates."""
        x, y, z = np.eye(3)
        bases = {
            Basis.FWD_LEFT: (x, y, z),
            Basis.EAST_NORTH: (x, y, z),
            Basis.FWD_RIGHT: (x, -y, -z),
            Basis.NORTH_EAST: (x, -y, -z),
            Basis.RIGHT_DOWN: (z, -x, -y),
            Basis.RIGHT_UP: (-z, -x, y),
        }

        return np.array(bases[self]).T @ np.array(bases[other])


def reshape_nx3(x):
    """Standardize array of 3D points/vectors x to have shape (N, 3) whether x is (3,)
    (3, 1), or already (N, 3) where N >= 1.

    The 3D points/vectors in x are expected to be in rows.

    Parameters
    ----------
    x : array-like sequence

    Returns
    -------
    x : ndarray - shape (N, 3)
    is_id : boolean
        If True, x was provided as a 1d sequence.

    """
    is_1d = False

    x = np.asarray(x)

    if x.ndim == 1:
        is_1d = True
        x = np.atleast_2d(x)
    elif x.shape == (3, 1):
        x = x.T

    if x.shape[1] != 3:
        raise ValueError(f"invalid input shape: {x.shape}. 3 columns required.")

    return x, is_1d


def wrap(x, a, b):
    """Return x wrapped into the interval [a, b)

    Parameters
    ----------
    x : scalar or ndarray
    a, b : floats

    Returns
    -------
    scalar or ndarray
    """
    return a + (x - a) % (b - a)


def in_0_2pi(phi):
    """Wrap phi into [0, 2*pi)."""
    return wrap(phi, 0, 2 * pi)


def in_mpi_pi(phi):
    """Wrap phi into [-pi, pi)"""
    return wrap(phi, -pi, pi)


def cross_product_matrix(a, b, c):
    """Return a 3x3 skew-symmetric cross-product matrix from the vector
    components a, b, and c."""

    # Handle 1x1 arrays. Explicit casting to scalars also serves as input validation,
    # since numpy will raise a ValueError if this doesn't work.
    a = a.item() if isinstance(a, np.ndarray) else a
    b = b.item() if isinstance(b, np.ndarray) else b
    c = c.item() if isinstance(c, np.ndarray) else c

    return np.array([[0.0, -c, b], [c, 0.0, -a], [-b, a, 0.0]])
