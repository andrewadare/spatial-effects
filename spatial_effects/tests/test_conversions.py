import unittest
from math import pi

import numpy as np

import spatial_effects as sfx


class ConversionTests(unittest.TestCase):
    def setUp(self):
        """Runs before every test function."""

        # Number of quaternions or vectors to use in tests.
        self.n = 100

    def check_eq(self, a, b, atol=1e-8):
        """Test for approximate equality."""
        self.assertTrue(np.allclose(a, b, atol=atol))

    def print_different(self, a, b):
        different = ~np.all(np.isclose(a, b), axis=1)
        for a, b in zip(a[different], b[different]):
            print(a, b)

    def test_so3_to_rvec_norm(self):
        print("\ntest_so3_to_rvec_norm")
        yprs = np.random.uniform(low=0.0, high=2 * pi, size=(100, 3)) - pi
        for ypr in yprs:
            R = sfx.ypr_to_so3(ypr)
            r = sfx.so3_to_rvec(R)
            theta = np.linalg.norm(r)
            self.assertGreaterEqual(theta, 0)
            self.assertLessEqual(theta, pi)

    def test_rvec_so3_roundtrip(self):
        print("\ntest_rvec_so3_roundtrip")
        # when norm(r) = pi, r == -r.
        for r in sfx.rrand(100):
            self.check_eq(r, sfx.so3_to_rvec(sfx.rvec_to_so3(r)))

    def test_rvec_pi_norm(self):
        print("\ntest_rvec_pi_norm")
        r = np.array([0, 0, pi])
        r_also = sfx.so3_to_rvec(sfx.rvec_to_so3(r))
        self.check_eq(r, r_also)

        r = np.array([0, pi, 0])
        r_also = sfx.so3_to_rvec(sfx.rvec_to_so3(r))
        self.check_eq(r, r_also)

        r = np.array([pi, 0, 0])
        r_also = sfx.so3_to_rvec(sfx.rvec_to_so3(r))
        self.check_eq(r, r_also)

    def _test_rodrigues(self, r):
        R = sfx.rodrigues(r)
        r_also = sfx.rodrigues(R)

        # Roundtrip check
        self.check_eq(r, r_also)

        # Check R against r -> q -> R
        Rq = sfx.quaternion_to_so3(sfx.rvec_to_quaternion(r))
        self.check_eq(R, Rq)

    def test_rodrigues(self):
        print("\ntest_rodrigues")
        for q in sfx.qrand(100):
            r = sfx.quaternion_to_rvec(q)
            self._test_rodrigues(r)

        # When R is I, r is zeros.
        self._test_rodrigues(np.zeros((3,)))
        self._test_rodrigues(np.zeros((3, 1)))
        self._test_rodrigues(np.zeros((1, 3)))
        self._test_rodrigues([0, 0, pi])

    def test_q_to_ypr_roundtrip_single(self):
        print("\ntest_q_to_ypr_roundtrip_single")
        q = sfx.qrand()
        ypr = sfx.quaternion_to_ypr(q)
        q_also = sfx.ypr_to_quaternion(ypr)
        d = sfx.quaternion_distance(q, q_also)
        self.check_eq(d, 0.0)

    def test_q_to_ypr_roundtrip_multi(self):
        print("\ntest_q_to_ypr_roundtrip_multi")
        q = sfx.qrand(self.n)
        ypr = sfx.quaternion_to_ypr(q)
        q_also = sfx.ypr_to_quaternion(ypr)
        d = sfx.quaternion_distance(q, q_also)
        self.check_eq(d, np.zeros(self.n))

    def test_quaternion_rotation_matrix_roundtrip(self):
        print("\ntest_quaternion_rotation_matrix_roundtrip")
        q = sfx.qrand()
        R = sfx.quaternion_to_so3(q)
        q_also = sfx.so3_to_quaternion(R)
        d = sfx.quaternion_distance(q, q_also)
        if np.abs(d) > 1e-8:
            self.print_different(np.atleast_2d(q), np.atleast_2d(q_also))
        self.check_eq(d, 0)

    def test_matrix_to_quaternion_1(self):
        print("\ntest_matrix_to_quaternion_1")
        qs = sfx.qrand(100)
        yprs = sfx.quaternion_to_ypr(qs)
        for q, ypr in zip(qs, yprs):
            R = sfx.ypr_to_so3(ypr)
            q_also = sfx.so3_to_quaternion(R)
            d = sfx.quaternion_distance(q, q_also)
            if np.abs(d) > 1e-8:
                self.print_different(np.atleast_2d(q), np.atleast_2d(q_also))
            self.check_eq(d, 0)

    def test_matrix_to_quaternion_2(self):
        print("\ntest_matrix_to_quaternion_2")
        s = 0.5 * np.sqrt(2)
        R = np.array([[-s, -s, 0], [s, -s, 0], [0, 0, 1]])
        q = sfx.so3_to_quaternion(R)
        R_also = sfx.quaternion_to_so3(q)
        self.check_eq(R, R_also)

    def test_rotation_matrix_quaternion_equivalence(self):
        print("\ntest_rotation_matrix_quaternion_equivalence")
        q = sfx.qrand()
        R = sfx.quaternion_to_so3(q)
        x = np.random.random(3)
        x_prime = R @ x
        x_prime_also = sfx.qrotate(x, q)
        self.check_eq(x_prime, x_prime_also)

    def test_quaternion_to_rotation_matrix(self):
        print("\ntest_quaternion_to_rotation_matrix")
        q = sfx.qrand()
        q0, q1, q2, q3 = q
        R = sfx.quaternion_to_so3(q)
        R_also = np.array(
            [
                [
                    q0**2 + q1**2 - q2**2 - q3**2,
                    2 * (q1 * q2 - q0 * q3),
                    2 * (q1 * q3 + q0 * q2),
                ],
                [
                    2 * (q1 * q2 + q0 * q3),
                    q0**2 - q1**2 + q2**2 - q3**2,
                    2 * (q2 * q3 - q0 * q1),
                ],
                [
                    2 * (q1 * q3 - q0 * q2),
                    2 * (q2 * q3 + q0 * q1),
                    q0**2 - q1**2 - q2**2 + q3**2,
                ],
            ]
        )
        self.check_eq(R, R_also)


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    unittest.main()
