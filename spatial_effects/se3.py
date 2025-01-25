import numpy as np

from .conversions import (
    quaternion_to_so3,
    rvec_to_so3,
    so3_to_rvec,
    so3_to_quaternion,
)
from .common import in_so3
from .so3 import so3_jacobian, inv_so3_jacobian


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
        - another SE(3) object
        - a 4x4 matrix [[R, t], [0s, 1]]
        - a 6 dof vector [rho, phi] where R = exp(phi^) and t = J @ rho
        - a translation 3-vector and a 3x3 rotation matrix
        - a translation 3-vector and a Hamilton unit quaternion (w,x,y,z)
        """
        self._R = np.eye(3)
        self._t = np.zeros(3)
        self.timestamp = timestamp

        if not args:
            return

        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, SE3):
                self._R = arg.R
                self._t = arg.t
            else:
                arg = np.asarray(arg)
                if arg.shape == (4, 4):
                    self._R = arg[:3, :3]
                    self._t = arg[:3, 3]
                elif arg.shape in [(6,), (6, 1), (1, 6)]:
                    pose = SE3.exp(arg)
                    self._R = pose.R
                    self._t = pose.t
                else:
                    raise ValueError(
                        f"Cannot interpret arg as a 4x4 matrix or 6-vector: {arg=}"
                    )
        elif len(args) == 2:
            trans, rot = np.asarray(args[0]), np.asarray(args[1])

            assert trans.size == 3, f"Translation must be a 3-vector: {trans.size=}"

            self._t = trans.ravel()

            if rot.size == 4:
                self._R = quaternion_to_so3(rot.ravel())
            elif rot.shape == (3, 3):
                if not in_so3(rot):
                    raise ValueError(f"Invalid rotation matrix:\n{rot=}")
                self._R = rot
            else:
                raise ValueError(
                    f"Rotation (2nd arg) must be either a 3x3 matrix or a quaternion: {rot=}"
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

    def __add__(self, vec: np.ndarray) -> "SE3":
        """Right (+) operator for SE(3): self + vec.

        See https://arxiv.org/abs/1812.01537, eq 25

        """
        if not isinstance(vec, np.ndarray) or vec.size != 6:
            raise ValueError(
                "Addition must be performed with a 6 DOF Euclidean vector"
                " [x, y, z, r1, r2, r3]"
            )
        return self * SE3.exp(vec)

    def __sub__(self, other) -> np.ndarray:
        """Right (-) operator for SE(3): self - other.

        See https://arxiv.org/abs/1812.01537, eq 26
        """
        if not isinstance(other, SE3):
            raise ValueError("Subtraction with a non-SE3 object")
        return SE3.log(other.inverse * self)

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

    @staticmethod
    def exp(tau: np.ndarray) -> "SE3":
        assert tau.shape in [(6,), (6, 1)], f"{tau=}"
        rho, phi = tau[:3], tau[3:]
        J = so3_jacobian(phi)
        R = rvec_to_so3(phi)
        t = J @ rho
        return SE3(t, R)

    @staticmethod
    def log(pose: "SE3") -> np.ndarray:
        phi = so3_to_rvec(pose.R)
        rho = inv_so3_jacobian(phi) @ pose.t
        return np.hstack([rho, phi])

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
        return SE3.log(self)
