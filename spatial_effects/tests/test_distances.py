import unittest
from math import pi

import numpy as np

import spatial_effects as sfx


class DistanceTests(unittest.TestCase):
    def setUp(self):
        """Runs before every test function."""
        pass

    def check_eq(self, a, b, atol=1e-8):
        """Test for approximate equality."""
        self.assertTrue(np.allclose(a, b, atol=atol))

    def test_quaternion_angular_distance(self):
        print("\ntest_quaternion_angular_distance")
        phi = pi / 4
        q1 = sfx.qnull()
        q2 = sfx.rvec_to_quaternion([0, 0, phi])
        self.check_eq(phi, sfx.angle_between_quaternions(q1, q2))

    def test_so3_angular_distance(self):
        print("\ntest_so3_angular_distance")
        phi = pi / 4
        R1 = np.eye(3)
        R2 = sfx.rvec_to_so3([0, 0, phi])
        self.check_eq(phi, sfx.so3_angular_distance(R1, R2))

    def test_so3_chordal_distance(self):
        print("\ntest_so3_chordal_distance")
        phi = pi / 4
        R1 = np.eye(3)
        R2 = sfx.rvec_to_so3([0, 0, phi])
        d = sfx.so3_chordal_distance(R1, R2)
        self.check_eq(d, np.linalg.norm(R1 - R2))
