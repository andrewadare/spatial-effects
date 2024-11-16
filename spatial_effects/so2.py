from math import sin, cos, atan2

import numpy as np

from .common import in_so2


def so2_matrix(angle: float):
    """Returns a rotation matrix in SO(2) from `angle` provided in radians."""
    c = cos(angle)
    s = sin(angle)

    return np.array([[c, -s], [s, c]])


def so2_angle(R: np.ndarray):
    """Get rotation angle from SO(2) matrix, accounting for sign of rotation."""
    if not in_so2(R):
        raise ValueError("Not a valid rotation matrix:", R)
    return atan2(R[1, 0], R[0, 0])
