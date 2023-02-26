import unittest
from math import pi

import numpy as np

import spatial_effects as sfx


class AveragingTests(unittest.TestCase):
    def setUp(self):
        """Runs before every test function."""
        pass

    def check_eq(self, a, b, atol=1e-8):
        """Test for approximate equality."""
        self.assertTrue(np.allclose(a, b, atol=atol))

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
