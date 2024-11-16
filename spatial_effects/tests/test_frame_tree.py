from math import pi
import unittest

import numpy as np

from spatial_effects import SE3, FrameTree, FrameTreeValidationError


class FrameTreeTests(unittest.TestCase):
    def setUp(self):
        """Runs before every test function."""
        np.set_printoptions(precision=5, suppress=True)

        # Humanoid transform tree
        self.tree = FrameTree()
        self.tree.add(SE3([1, 2, 3], [0, 0, 0]), "body", "origin")
        self.tree.add(SE3([0, 0, 0.5], [0, 0, -pi / 3]), "head", "body")
        self.tree.add(SE3([0, 1, 0], [0, 0, 0]), "l_shoulder", "body")
        self.tree.add(SE3([0, -1, 0], [0, 0, 0]), "r_shoulder", "body")
        self.tree.add(SE3([0, 0, -0.7], [0, -pi / 6, 0]), "l_elbow", "l_shoulder")
        self.tree.add(SE3([0, 0, -0.7], [0, -pi / 3, 0]), "r_elbow", "r_shoulder")
        self.tree.add(SE3([0, 0, -0.6], [0, 0, 0]), "l_wrist", "l_elbow")
        self.tree.add(SE3([0, 0, -0.6], [0, 0, 0]), "r_wrist", "r_elbow")
        self.tree.add(SE3([0, 0, -1], [0, 0, pi / 8]), "waist", "body")
        self.tree.add(SE3([0, 0.4, 0], [0, -pi / 4, 0]), "l_hip", "waist")
        self.tree.add(SE3([0, -0.4, 0], [0, 0, 0]), "r_hip", "waist")
        self.tree.add(SE3([0, 0, -1.2], [0, pi / 2, 0]), "l_knee", "l_hip")
        self.tree.add(SE3([0, 0, -1.2], [0, 0, 0]), "r_knee", "r_hip")
        self.tree.add(SE3([0, 0, -0.9], [0, -pi / 8, 0]), "l_ankle", "l_knee")
        self.tree.add(SE3([0, 0, -0.9], [0, 0, 0]), "r_ankle", "r_knee")
        self.tree.add(SE3([0.2, 0, 0], [0, 0, 0]), "l_foot", "l_ankle")
        self.tree.add(SE3([0.2, 0, 0], [0, 0, 0]), "r_foot", "r_ankle")

    def test_str(self):
        print("\ntest_frame_tree_str_method")
        print(self.tree)

    def test_validate(self):
        print("\ntest_frame_tree_validate_method")
        self.assertTrue(self.tree.validate())

    def test_tree_traversal_1(self):
        """Identity"""
        print("\ntest_tree_traversal_1")
        self.assertEqual(self.tree.get_transform("l_knee", "l_knee"), SE3())

    def test_tree_traversal_2(self):
        """Symmetry"""
        print("\ntest_tree_traversal_2")
        self.assertEqual(
            self.tree.get_transform("l_knee", "r_knee"),
            self.tree.get_transform("r_knee", "l_knee").inverse,
        )

    def test_tree_traversal_3(self):
        """Invalid frame name"""
        print("\ntest_tree_traversal_3")
        self.assertRaises(LookupError, self.tree.get_transform, "bogus_frame", "origin")

    def test_disjoint_validation(self):
        """Adding an orphaned coordinate frame"""
        print("\ntest_disjoint_validation")
        self.tree.add(
            SE3([1, 0, 0], [0, 0, 0]), "disconnected_child", "disconnected_parent"
        )
        self.assertFalse(self.tree.validate())

    def test_disjoint_root_search_exception(self):
        print("\ntest_disjoint_root_search_exception")
        self.tree.add(
            SE3([1, 0, 0], [0, 0, 0]), "disconnected_child", "disconnected_parent"
        )
        self.assertRaises(FrameTreeValidationError, self.tree.root_frame)

    def test_disjoint_traversal_exception(self):
        print("\ntest_disjoint_traversal_exception")
        self.tree.add(
            SE3([1, 0, 0], [0, 0, 0]), "disconnected_child", "disconnected_parent"
        )
        self.assertRaises(
            FrameTreeValidationError, self.tree.get_transform_to_root, "l_foot"
        )

    def test_serdes(self):
        print("\ntest_serdes")
        serialized_tree = self.tree.to_dict()
        deserialized_tree = FrameTree.from_dict(serialized_tree)
        self.assertEqual(self.tree, deserialized_tree)
