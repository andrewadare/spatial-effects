import numpy as np

from .common import in_so2
from .so2 import so2_matrix, so2_angle


class SE2:
    def __init__(self, *args):
        self._R = np.eye(2)
        self._t = np.zeros(2)

        if not args:
            return

        if len(args) == 1:  # Either homogeneous matrix or SE2 object
            if isinstance(args[0], np.ndarray) and args[0].shape == (3, 3):
                T = args[0]
                self._R = T[:2, :2]
                self._t = T[:2, 2]
            elif isinstance(args[0], SE2):
                self._R = args[0].R
                self._t = args[0].t
            else:
                raise ValueError("Positional arg must be a 3x3 array or SE(2) object")
        elif len(args) == 2:  # Either ([x, y], phi) or ([x, y], R)
            trans, rot = np.asarray(args[0]), np.asarray(args[1])

            assert trans.size == 2, f"Translation must be a 2-vector: {trans}"

            self._t = trans.ravel()

            if rot.shape == (2, 2):
                if not in_so2(rot):
                    raise ValueError(f"Invalid rotation matrix:\n{rot}")
                self._R = rot
            elif np.isscalar(rot):
                self._R = so2_matrix(rot)
            elif rot.size == 1:
                self._R = so2_matrix(rot.item())
            else:
                raise ValueError(
                    f"Cannot identify rotation by array dimensions: {rot.shape}"
                )
        else:
            raise ValueError(
                "SE2 constructor expects 0, 1, or 2 positional arguments. "
                f"Received {len(args)}"
            )

    def __repr__(self):
        return str(self.matrix)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply this SE(2) transformation to x"""
        return self.R @ x + self.t

    def __eq__(self, other):
        if not isinstance(other, SE2):
            raise ValueError("Comparison with a non-SE2 object")
        return np.allclose(self.matrix, other.matrix)

    def __add__(self, vec: np.ndarray):
        """Returns an SE2 object updated by a 3 DOF vector [x, y, phi]."""
        if not isinstance(vec, np.ndarray) or vec.size != 3:
            raise ValueError(
                "Addition must be performed with a 6 DOF Euclidean vector"
                " [x, y, phi]"
            )
        vec = vec.ravel()
        return SE2(self.matrix @ SE2(vec[:2], vec[2:]).matrix)

    def __sub__(self, other):
        """Returns a 3 DOF Euclidean vector [dx, dy, dphi]
        as the difference between two SE2 objects.
        """
        if not isinstance(other, SE2):
            raise ValueError("Subtraction with a non-SE2 object")
        return SE2(other.inverse.matrix @ self.matrix).vec

    def __mul__(self, other):
        """Matrix multiplication operator: if `other` is an SE2 object, returns
        a composed SE2 object. If `other` is an array of one or more points,
        the transformed point(s) will be returned.
        """
        if isinstance(other, SE2):
            return SE2(self.matrix @ other.matrix)
        elif isinstance(other, np.ndarray):
            return self(other)
        else:
            raise ValueError("* operator must be followed by an SE2 or array object.")

    @property
    def inverse(self):
        return SE2(-self.R.T @ self.t, self.R.T)

    @property
    def matrix(self):
        M = np.eye(3)
        M[:2, :2] = self._R
        M[:2, 2] = self._t
        return M

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, t_):
        self._t = t_

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, R_):
        self._R = R_

    @property
    def vec(self):
        return np.hstack([self._t, so2_angle(self._R)])
