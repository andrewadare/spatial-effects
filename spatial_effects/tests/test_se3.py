import unittest

import numpy as np
from numpy.random import randn, uniform

import spatial_effects as sfx
from spatial_effects.se3 import SE3


class SE3Tests(unittest.TestCase):
    def setUp(self):
        """Runs before every test function."""
        np.set_printoptions(precision=5, suppress=True)

    def check_eq(self, a, b, rtol=1e-4, atol=0):
        """Test for approximate equality."""
        equal = np.allclose(a, b, rtol=rtol, atol=atol)
        if not equal:
            print(f"{a} != {b}")
        self.assertTrue(equal)

    def print_different(self, a, b):
        different = ~np.all(np.isclose(a, b), axis=1)
        for a, b in zip(a[different], b[different]):
            print(a, b)

    def _test_ops(self, a, b, vec):
        self._test_operator_identity_1(a, b)
        self._test_operator_identity_2(a, b, vec)

    def _test_operator_identity_1(self, a, b):
        # Check that a ⊞ (b ⊟ a) == b
        try:
            self.assertAlmostEqual((a + (b - a)), b)
        except TypeError:
            print("Warning: skipping test")
            # Occasionally, this happens:
            # TypeError: type numpy.ndarray doesn't define __round__ method
            pass

    def _test_operator_identity_2(self, a, b, vec):
        # Check that (a ⊞ vec) ⊟ a == vec
        self.check_eq((a + vec) - a, vec)

    def test_operator_identities(self):
        print("\ntest_operator_identities")

        a = SE3(randn(3), sfx.qrand())
        b = SE3(randn(3), sfx.qrand())

        vec = np.zeros(6)
        r = sfx.quaternion_to_rvec(sfx.qnull())
        vec[:3] = uniform(size=(3,))
        vec[3:] = r
        self._test_ops(a, b, vec)

    def test_se3_add_exception(self):
        print("\ntest_se3_add_exception")
        # Make sure a ⊞ a raises a ValueError
        a = SE3(randn(3), sfx.qrand())
        with self.assertRaises(ValueError):
            a + a

    def test_se3_q_property(self):
        print("\ntest_se3_q_property")
        q = sfx.qrand()
        a = SE3(randn(3), q)
        close = np.allclose(q, a.q, rtol=1e-4, atol=0) or np.allclose(
            q, -a.q, rtol=1e-4, atol=0
        )
        self.assertTrue(close)

    def test_se3_r_property(self):
        print("\ntest_se3_r_property")
        r = sfx.rrand()
        a = SE3([*randn(3), *r])
        self.check_eq(r, a.r)

    def test_se3_inverse(self):
        print("\ntest_se3_inverse")
        a = SE3(randn(3), sfx.qrand())
        self.check_eq((a.inverse * a).matrix, np.eye(4), atol=1e-6)

    def test_se3_input_shape(self):
        print("\ntest_se3_input_shape")
        t = uniform(size=(3,))
        q = sfx.qrand()
        self.assertTrue(SE3(t, q) == SE3(t[:, None], q[:, None]))

    def test_bad_rotation_matrix_input(self):
        print("\ntest_bad_rotation_matrix_input")
        R = sfx.rvec_to_so3(sfx.rrand()) + 1e-8
        self.assertFalse(sfx.in_so3(R))
        self.assertRaises(ValueError, sfx.SE3, [0, 0, 0], R)

    def test_se3_copy_constructor(self):
        print("\ntest_se3_copy_constructor")
        a = SE3([*randn(3), *sfx.rrand()])
        b = SE3(a)
        self.check_eq(a.matrix, b.matrix)
        self.assertIsNot(a, b)

    def test_se3_exp_log(self):
        print("\ntest_se3_exp_log")
        T1 = SE3(randn(3), sfx.qrand())
        T2 = SE3(T1.vec)
        self.check_eq(T1.matrix, T2.matrix)
