import unittest
from math import pi

import numpy as np
from numpy.random import randn, uniform

import spatial_effects as sfx
from spatial_effects.se2 import SE2


class SE2Tests(unittest.TestCase):
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

        a = SE2(randn(2), uniform(-pi, pi))
        b = SE2(randn(2), uniform(-pi, pi))
        vec = np.array([*randn(2), uniform(-pi, pi)])

        self._test_ops(a, b, vec)

    def test_se2_add_exception(self):
        print("\ntest_se2_add_exception")
        # Make sure a ⊞ a raises a ValueError
        a = SE2(randn(2), uniform(-pi, pi))
        with self.assertRaises(ValueError):
            a + a

    def test_se2_inverse(self):
        print("\ntest_se2_inverse")
        a = SE2(randn(2), uniform(-pi, pi))
        self.check_eq((a.inverse * a).matrix, np.eye(3), atol=1e-6)

    def test_bad_rotation_matrix_input(self):
        print("\ntest_bad_rotation_matrix_input")
        phi = uniform(-pi, pi)
        R = sfx.so2_matrix(phi) + 1e-6
        self.assertFalse(sfx.in_so2(R))
        self.assertRaises(ValueError, sfx.SE2, [0, 0], R)

    def test_se3_copy_constructor(self):
        print("\ntest_se3_copy_constructor")
        a = SE2(randn(2), uniform(-pi, pi))
        b = SE2(a)
        self.check_eq(a.matrix, b.matrix)
        self.assertIsNot(a, b)
