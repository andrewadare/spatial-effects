import unittest

import numpy as np

import spatial_effects as sfx


class RotationTests(unittest.TestCase):
    def setUp(self):
        """Runs before every test function."""
        pass

    def check_eq(self, a, b, atol=1e-8):
        """Test for approximate equality."""
        self.assertTrue(np.allclose(a, b, atol=atol))

    def print_different(self, a, b):
        different = ~np.all(np.isclose(a, b), axis=1)
        for a, b in zip(a[different], b[different]):
            print(a, b)

    def test_rotate_axis_angle(self):
        print("\ntest_rotate_axis_angle")

        q = sfx.qrand()
        r = sfx.quaternion_to_vector(q)
        v = np.random.random((100, 3))

        self.check_eq(sfx.qrotate(v, q), sfx.rotate_axis_angle(v, r))

    def test_rotation_composition(self):
        print("\ntest_rotation_composition")
        x = np.random.random(3)
        q1, q2 = sfx.qrand(2)
        R1, R2 = sfx.quaternion_to_so3(q1), sfx.quaternion_to_so3(q2)

        xprime_q = sfx.qrotate(sfx.qrotate(x, q1), q2)
        xprime_R = R2 @ R1 @ x

        self.check_eq(xprime_R, xprime_q)
