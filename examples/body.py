#!/usr/bin/env python

from math import pi

import numpy as np
import open3d as o3d

from spatial_effects import SE3, Transform, TransformForest


def draw_tree(tf: TransformForest):
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

    assert tf.size == 1  # one tree

    root_frame = tf.trees[0].root  # Name of root node

    # Compute transforms with respect to root frame
    global_transforms: dict[str, Transform] = dict()  # key on child frame
    for child_frame, transform in tf.transforms.items():
        t = Transform(tf.get_se3(child_frame, root_frame), child_frame, root_frame)
        global_transforms[child_frame] = t

    # Draw lines between frames
    path = tf.trees[0].frame_map.path  # child frame => parent frame dict
    lines = []
    for child_frame in path:
        parent_frame = path[child_frame]
        if parent_frame:
            # line endpoints
            child_point = global_transforms[child_frame].se3.t
            if parent_frame == root_frame:
                parent_point = np.zeros(3)
            else:
                parent_point = global_transforms[parent_frame].se3.t

            # Open3D LineSet from points
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(
                    np.array([child_point, parent_point])
                ),
                lines=o3d.utility.Vector2iVector([[0, 1]]),
            )
            lines.append(line_set)
    o3d.visualization.draw_geometries([origin, *lines])


def main():

    # Head, shoulders, knees and toes....
    transforms = [
        Transform(SE3([1, 2, 3], [0, 0, 0]), "body", "origin"),
        Transform(SE3([0, 0, 0.5], [0, 0, -pi / 3]), "head", "body"),
        Transform(SE3([0, 1, 0], [0, 0, 0]), "l_shoulder", "body"),
        Transform(SE3([0, -1, 0], [0, 0, 0]), "r_shoulder", "body"),
        Transform(SE3([0, 0, -0.7], [0, -pi / 6, 0]), "l_elbow", "l_shoulder"),
        Transform(SE3([0, 0, -0.7], [0, -pi / 3, 0]), "r_elbow", "r_shoulder"),
        Transform(SE3([0, 0, -0.6], [0, 0, 0]), "l_wrist", "l_elbow"),
        Transform(SE3([0, 0, -0.6], [0, 0, 0]), "r_wrist", "r_elbow"),
        Transform(SE3([0, 0, -1], [0, 0, pi / 8]), "waist", "body"),
        Transform(SE3([0, 0.4, 0], [0, -pi / 4, 0]), "l_hip", "waist"),
        Transform(SE3([0, -0.4, 0], [0, 0, 0]), "r_hip", "waist"),
        Transform(SE3([0, 0, -1.2], [0, pi / 2, 0]), "l_knee", "l_hip"),
        Transform(SE3([0, 0, -1.2], [0, 0, 0]), "r_knee", "r_hip"),
        Transform(SE3([0, 0, -0.9], [0, -pi / 8, 0]), "l_ankle", "l_knee"),
        Transform(SE3([0, 0, -0.9], [0, 0, 0]), "r_ankle", "r_knee"),
        Transform(SE3([0.2, 0, 0], [0, 0, 0]), "l_foot", "l_ankle"),
        Transform(SE3([0.2, 0, 0], [0, 0, 0]), "r_foot", "r_ankle"),
    ]

    tf = TransformForest(transforms)
    draw_tree(tf)


if __name__ == "__main__":
    main()
