from collections import deque
from dataclasses import dataclass
from typing import Any, Optional, Union

from .se3 import SE3


@dataclass
class Transform:
    se3: SE3
    child_frame: str
    parent_frame: str
    timestamp: Any = None


class TransformTree:
    def __init__(self, transforms: list[Transform] = []):
        # Maps child frame ID => Transform
        self._transforms: dict[str, Transform] = dict()
        self._parents_to_children: dict[str, set[str]] = dict()
        self._children_to_parents: list[dict[str, Union[str, None]]] = []
        if transforms:
            self.update(transforms)

    @property
    def transforms(self):
        return self._transforms

    @property
    def root_nodes(self):
        # By construction, the root node is always the first
        # key in the child => parent dictionary and its parent is
        # always None.
        return [next(iter(d)) for d in self._children_to_parents]

    @property
    def tree(self):
        return self._parents_to_children

    @property
    def paths(self):
        return self._children_to_parents

    def update(self, t: Union[Transform, list[Transform]]):
        if isinstance(t, Transform):
            self._transforms[t.child_frame] = t
        elif isinstance(t, list):
            for x in t:
                if not isinstance(x, Transform):
                    raise ValueError(f"Must be a Transform: {type(x)}")
                self._transforms[x.child_frame] = x
        else:
            raise ValueError(f"Unsupported type {type(t)}")

        self._parents_to_children = map_parents_to_children(self._transforms)
        self._children_to_parents = map_children_to_parents(self._parents_to_children)

    def get_se3(self, frame_a: str, frame_b: str) -> SE3:
        """Compute the SE(3) transformation from frame_a to frame_b.

        By convention, upward (downward) traversals are forward (inverse)
        transformations.

        Parameters
        ==========
        frame_a, frame_b: frame identifiers

        Returns
        =======
        SE(3) from frame frame_a to frame_b

        Raises
        ======
        ValueError if no path is found from frame_a to frame_b
        """

        paths = self._children_to_parents

        assert isinstance(paths, list)
        assert len(paths) > 0
        for path in paths:
            assert isinstance(path, dict)

        for path in paths:
            if frame_a not in path or frame_b not in path:
                continue

            # Find paths from a and b up to the root
            a_up: list[str] = traverse_up(path, frame_a)
            b_up: list[str] = traverse_up(path, frame_b)

            # Create transform chains
            a, b = SE3(), SE3()
            for frame in a_up[:-1]:
                a = self._transforms[frame].se3 * a
            for frame in b_up[:-1]:
                b = self._transforms[frame].se3 * b
            return b.inverse * a
        raise ValueError(f"No path from {frame_a} to {frame_b}")


def bfs(g: dict, root: Any):
    """Textbook breadth-first search of a graph g from `root` node.

    Parameters
    ==========
    g: dictionary mapping parent nodes to a sequence of child nodes
    root: starting point for search

    Returns
    =======
    visited nodes in a dictionary mapping children to parents
    """
    parents = {root: None}  # child => parent
    q = deque([root])  # nodes to visit
    while q:
        u = q.popleft()
        if u in g:
            for v in g[u]:
                if v not in parents:
                    parents[v] = u
                    q.append(v)
    return parents


def traverse_up(
    parents: dict[str, Optional[str]], a: str, b: Optional[str] = None
) -> list[str]:
    path: list[str] = []  # frame names
    if b and b not in parents:
        return path
    try:
        parent = parents[a]
    except KeyError:
        return path
    while parent:
        if not path:
            path.append(a)
        path.append(parent)
        if parent == b:
            break
        parent = parents[parent]
    return path


def map_parents_to_children(transforms: dict[str, Transform]) -> dict[str, set[str]]:
    """Returns a dict mapping parent nodes to sets of child nodes."""
    frames: dict[str, set[str]] = dict()
    for child_frame, transform in transforms.items():
        if transform.parent_frame in frames:
            frames[transform.parent_frame].add(child_frame)
        else:
            frames[transform.parent_frame] = set([child_frame])
    return frames


def map_children_to_parents(
    parents_to_children: dict[str, set[str]]
) -> list[dict[str, Union[str, None]]]:
    """Search the transform tree and return a list of graphs, one for each
    connected component.
    If the tree is fully connected, the list will contain one dictionary.
    Otherwise, the list contains one entry for each disconnected sub-tree.
    Each dictionary maps child coordinate frames to their parents (or None
    for root nodes).
    """
    # Traverse all subgraphs using BFS, resulting in a list of
    # dictionaries
    all_paths: list[dict[str, Union[str, None]]] = []
    for parent in parents_to_children:
        all_paths.append(bfs(parents_to_children, parent))

    # Select paths that are not subsets of others.
    # If a root node is not any other node's child,
    # the path is considered an independent graph.
    paths = []
    children = set().union(*parents_to_children.values())
    for parents in all_paths:
        root = next(iter(parents))
        assert parents[root] is None

        if root not in children:
            paths.append(parents)

    return paths
