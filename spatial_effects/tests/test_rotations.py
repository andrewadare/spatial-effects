from math import pi
import unittest

import numpy as np

import spatial_effects as sfx


class RotationTests(unittest.TestCase):
    def setUp(self):
        """Runs before every test function."""
        np.set_printoptions(precision=5, suppress=True)

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

    def test_rrand_distribution(self):
        print("\ntest_rrand_distribution")
        num_samples = 100_000
        vecs = sfx.rrand(num_samples)

        # Check that the distribution of azimuthal angles is uniform with
        # a chi squared test comparing observed counts/bin (phi_hist) vs.
        # expected counts/bin (mu).
        num_bins = 10
        vecs /= np.linalg.norm(vecs, axis=1)[:, np.newaxis]
        phis = np.arctan2(vecs[:, 1], vecs[:, 0])
        phi_hist, phi_bin_edges = np.histogram(phis, bins=num_bins, range=[-pi, pi])
        mu = num_samples / num_bins
        chi_squared = np.sum(1 / mu * (phi_hist - mu) ** 2)
        self.assertLess(chi_squared / num_bins, 3.0)

        # TODO: test that the polar distribution is proportional to sin(theta).
        # thetas = np.arccos(vecs[:, 2] / lengths)

    def test_basis_change_in_so3(self):
        print("\ntest_basis_change_in_so3")

        for a in sfx.Basis:
            for b in sfx.Basis:
                R = a.to(b)
                self.check_eq(R.shape, (3, 3))
                self.check_eq(R.T @ R, np.eye(3))
                self.check_eq(np.linalg.det(R), 1)

    def test_basis_change_inverse(self):
        print("\ntest_basis_change_inverse")

        for a in sfx.Basis:
            for b in sfx.Basis:
                R = a.to(b)
                Rinv = b.to(a)
                self.check_eq(Rinv @ R, np.eye(3))
                self.check_eq(R @ Rinv, np.eye(3))
