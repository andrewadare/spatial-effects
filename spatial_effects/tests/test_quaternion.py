import unittest
from math import pi

import numpy as np

import spatial_effects as sfx


class QuaternionTests(unittest.TestCase):
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

    def test_slerp(self):
        print("\ntest_slerp")

        yaw_endpoint = pi
        step_size = 0.1

        qa = sfx.ypr_to_quaternion([0, 0, 0])
        qb = sfx.ypr_to_quaternion([yaw_endpoint, 0, 0])

        for i, alpha in enumerate(np.arange(0.0, 1.0, step_size)):
            q_expected = sfx.ypr_to_quaternion([alpha * yaw_endpoint, 0, 0])
            q_interp = sfx.slerp(qa, qb, alpha)
            self.check_eq(q_interp, q_expected)

    def test_lerp(self):
        """The LERP method is an approximation valid only for small intervals, so
        the yaw_endpoint is smaller than for the SLERP method and the equality
        tolerance is loosened.
        """
        print("\ntest_lerp")

        yaw_endpoint = pi / 12
        step_size = 0.1

        qa = sfx.ypr_to_quaternion([0, 0, 0])
        qb = sfx.ypr_to_quaternion([yaw_endpoint, 0, 0])

        for i, alpha in enumerate(np.arange(0.0, 1.0, step_size)):
            q_expected = sfx.ypr_to_quaternion([alpha * yaw_endpoint, 0, 0])
            q_interp = sfx.lerp(qa, qb, alpha)
            # print(alpha, q_expected, q_interp)
            self.check_eq(q_interp, q_expected, atol=1e-4)

    def test_euclidean_angle(self):
        print("\ntest_euclidean_angle")

        p = sfx.ypr_to_quaternion([0, 0, 0])
        q1 = sfx.ypr_to_quaternion([+pi / 2, 0, 0])
        q2 = sfx.ypr_to_quaternion([-pi / 2, 0, 0])

        d1 = sfx.euclidean_angle(p, q1)
        d2 = sfx.euclidean_angle(p, q2)

        self.check_eq(d1, pi / 2)
        self.check_eq(d2, pi / 2)

    def test_euclidean_angles(self):
        print("\ntest_euclidean_angles")
        n = 10
        angles = np.arange(0, pi, pi / n)
        ps = sfx.ypr_to_quaternion(np.zeros((n, 3)))
        qs = sfx.ypr_to_quaternion([[0, 0, a] for a in angles])
        self.check_eq(angles, sfx.euclidean_angle(ps, qs))

    def test_qleft_qright(self):
        print("\ntest_qleft_qright")
        q1, q2 = sfx.qrand(2)
        q12 = sfx.qmult(q1, q2)
        self.check_eq(sfx.qleft(q1) @ q2, q12)
        self.check_eq(sfx.qright(q2) @ q1, q12)

    def test_expq_logq_roundtrip(self):
        print("\ntest_expq_logq_roundtrip")

        # Generate rotation vectors
        qs = sfx.qrand(self.n)
        xs = sfx.q_angle(qs) * sfx.q_axis(qs)

        xs_also = 2 * sfx.logq(sfx.expq(0.5 * xs))
        self.check_eq(xs, xs_also)

    def test_qnorm_single(self):
        print("\ntest_qnorm_single")
        for _ in range(self.n):
            # q = Quaternion.random().elements
            q = sfx.qrand()
            self.check_eq(sfx.qnorm(q), 1.0)

    def test_normalize_1(self):
        print("\ntest_normalize_1")
        q = np.array([1.0, 0.5, 0.0, 0.0])
        q = sfx.normalize(q)
        self.check_eq(sfx.qnorm(q), 1.0)

    def test_normalize_N(self):
        print("\ntest_normalize_N")
        q = np.random.random([self.n, 4])
        q = sfx.normalize(q)
        self.check_eq(sfx.qnorm(q), np.ones(self.n))

    def test_qnorm_multi(self):
        print("\ntest_qnorm_multi")
        qs = sfx.qrand(self.n)
        self.check_eq(sfx.qnorm(qs), np.ones(self.n))

    def test_logq_vs_axis_angle(self):
        print("\ntest_logq_vs_axis_angle")
        qs = sfx.qrand(self.n)
        axes, angles = sfx.q_axis(qs), sfx.q_angle(qs)
        self.check_eq(axes * angles, 2 * sfx.logq(qs))

    def test_expq_logq_with_null_vectors(self):
        print("\ntest_expq_logq_with_null_vectors")
        xs = np.random.random([self.n, 3])
        nz = np.random.binomial(1, 0.5, self.n)  # 1's for nonzero rows
        xs[~nz, :] = 0.0
        xs_also = 2 * sfx.logq(sfx.expq(0.5 * xs))
        self.check_eq(xs, xs_also)

    def test_logq_qnull_1(self):
        print("\ntest_logq_qnull_1")
        q = sfx.qnull()
        v = sfx.logq(q)
        self.check_eq(v, np.zeros(3))

    def test_logq_qnull_N(self):
        print("\ntest_logq_qnull_N")
        q = sfx.qnull(self.n)
        v = sfx.logq(q)
        self.check_eq(v, np.zeros([self.n, 3]))

    def test_quaternion_mean(self):
        print("\ntest_quaternion_mean")
        yprs = np.array(
            [[pi / 6, 0, 0], [2 * pi / 6, 0, 0], [4 * pi / 6, 0, 0], [5 * pi / 6, 0, 0]]
        )
        row_indices = np.arange(yprs.shape[0])

        # Try the unit test 10x, with the order of the rows in yprs shuffled randomly
        # each time.
        for _ in range(10):
            np.random.shuffle(row_indices)
            yprs = yprs[row_indices, :]
            qs = sfx.ypr_to_quaternion(yprs)
            q_mean = sfx.quaternion_mean(qs)
            mean = sfx.quaternion_to_ypr(q_mean)
            self.check_eq(mean, np.array([pi / 2, 0, 0]))

    def test_qplus_qminus_consistency(self):
        """Show that qplus and qdiff work as mutual inverses."""
        print("\ntest_qplus_qminus_consistency")
        q, p = sfx.qrand(2)
        w = sfx.quaternion_to_vector(p)

        # check (q ⊞ w) ⊟ q == w
        self.check_eq(sfx.qdiff(sfx.qplus(q, w), q), w)

        # check q ⊟ (q ⊞ w) == -w
        self.check_eq(sfx.qdiff(q, sfx.qplus(q, w)), -w)

    def test_vectorized_qplus_qminus_consistency(self):
        print("\ntest_vectorized_qplus_qminus_consistency")
        q = sfx.qrand(5)
        p = sfx.qrand(5)
        w = sfx.quaternion_to_vector(p)

        # check (q ⊞ w) ⊟ q == w
        self.check_eq(sfx.qdiff(sfx.qplus(q, w), q), w)

        # check q ⊟ (q ⊞ w) == -w
        self.check_eq(sfx.qdiff(q, sfx.qplus(q, w)), -w)
