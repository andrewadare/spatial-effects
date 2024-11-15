import numpy as np

from .conversions import (
    quaternion_to_so3,
    rvec_to_so3,
    so3_to_rvec,
    so3_to_quaternion,
)
from .common import in_so3


class SE3:
    """Class for representing rigid transformations (translations and rotations) in
    three dimensional space that form the special Euclidean group SE(3).

    It implements the "boxplus" manifold operators ⊞ and ⊟ by overloading +
    and -. The + operator enables updating a manifold with a 6 DOF Euclidean vector
    [x,y,z,r1,r2,r3], where r is an Euler/Rodrigues rotation vector. The - operator
    returns the residual between two SE(3) objects as a 6 DOF vector.

    Example
    -------
    x = [1, 0, 0]              # Define a point/translation
    r = [0, 0, pi/2]           # Rodrigues rotation vector
    T1 = SE3()                 # Create identity transformation
    T2 = SE3(x, r)             # Lists, tuples, or ndarrays ok here
    callable(T1)               # True
    T2(x)                      # array([1., 1., 0.])
    T2.R @ x + T2.t            # same
    T2.inverse(T2(x))          # x
    T1.matrix                  # np.eye(4)
    T2.vec                     # array([1, 0, 0, 0, 0, 1.571])
    (T1 + T2.vec).vec          # same (in this case)
    T2 - T1                    # same (in this case)
    T1 + T2                    # ValueError! Cannot ⊞ two manifolds
    T1 + (T2 - T1) == T2       # always True
    (T1 + v) - T1 == v         # always True (v is any 6 DOF vector)
    """

    def __init__(self, *args, timestamp=None):
        """Construct an SE(3) object using any of the following:

        - no positional args (identity transformation)
        - a 4x4 matrix [[R,t],[0s,1]]
        - another SE(3) object
        - a translation 3-vector and a rotation vector
        - a translation 3-vector and a 3x3 rotation matrix
        - a translation 3-vector and a Hamilton unit quaternion (w,x,y,z)
        """
        self._R = np.eye(3)
        self._t = np.zeros(3)
        self.timestamp = timestamp

        if not args:
            return

        if len(args) == 1:
            if isinstance(args[0], np.ndarray) and args[0].shape == (4, 4):
                T = args[0]
                self._R = T[:3, :3]
                self._t = T[:3, 3]
            elif isinstance(args[0], SE3):
                self._R = args[0].R
                self._t = args[0].t
            else:
                raise ValueError("Positional arg must be a 4x4 array or SE(3) object")
        elif len(args) == 2:
            trans, rot = np.asarray(args[0]), np.asarray(args[1])

            assert trans.size == 3, f"Translation must be a 3-vector: {trans.size}"

            self._t = trans.ravel()

            if rot.size == 3:
                self._R = rvec_to_so3(rot)
            elif rot.size == 4:
                self._R = quaternion_to_so3(rot)
            elif rot.shape == (3, 3):
                if not in_so3(rot):
                    raise ValueError(f"Invalid rotation matrix:\n{rot}")
                self._R = rot
            else:
                raise ValueError(
                    f"Cannot identify rotation by array dimensions: {rot.shape}"
                )
        else:
            raise ValueError(
                "SE3 constructor expects 0, 1, or 2 positional arguments. "
                f"Received {len(args)}"
            )

    def __repr__(self) -> str:
        return str(self.matrix)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply this SE(3) transformation to x"""
        return self.R @ x + self.t

    def __eq__(self, other):
        if not isinstance(other, SE3):
            raise ValueError("Comparison with a non-SE3 object")
        return np.allclose(self.matrix, other.matrix)

    def __add__(self, vec: np.ndarray):
        """Returns an SE(3) object updated by a 6 DOF vector [x, r]."""
        if not isinstance(vec, np.ndarray) or vec.size != 6:
            raise ValueError(
                "Addition must be performed with a 6 DOF Euclidean vector"
                " [x, y, z, r1, r2, r3]"
            )
        vec = vec.ravel()
        return SE3(self.matrix @ SE3(vec[:3], vec[3:]).matrix)

    def __sub__(self, other):
        """Returns a 6 DOF Euclidean vector [dx, dy, dz, dr1, dr2, dr3]
        as the difference between two SE(3) objects.
        """
        if not isinstance(other, SE3):
            raise ValueError("Subtraction with a non-SE3 object")
        return SE3(other.inverse.matrix @ self.matrix).vec

    def __mul__(self, other):
        """Matrix multiplication operator: if `other` is an SE3 object, returns
        a composed SE3 object. If `other` is an array of one or more points,
        the transformed point(s) will be returned.
        """
        if isinstance(other, SE3):
            return SE3(self.matrix @ other.matrix)
        elif isinstance(other, np.ndarray):
            return self(other)
        else:
            raise ValueError("* operator must be followed by an SE3 or array object.")

    @property
    def inverse(self):
        return SE3(-self.R.T @ self.t, self.R.T)

    @property
    def matrix(self) -> np.ndarray:
        """Returns 4x4 homogeneous transform matrix"""
        M = np.eye(4)
        M[:3, :3] = self._R
        M[:3, 3] = self._t
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
    def r(self):
        return so3_to_rvec(self.R)

    @r.setter
    def r(self, r_):
        self._R = rvec_to_so3(r_)

    @property
    def q(self):
        return so3_to_quaternion(self.R)

    @q.setter
    def q(self, q_):
        self._R = quaternion_to_so3(q_)

    @property
    def vec(self):
        return np.hstack([self._t, so3_to_rvec(self._R)])
