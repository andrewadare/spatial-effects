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
        self.transforms: dict[str, Transform] = dict()
        if transforms:
            self.update(transforms)

    def update(self, t: Union[Transform, list[Transform]]):
        if isinstance(t, Transform):
            self.transforms[t.child_frame] = t
        elif isinstance(t, list):
            for x in t:
                if not isinstance(x, Transform):
                    raise ValueError(f"Must be a Transform: {type(x)}")
                self.transforms[x.child_frame] = x
        else:
            raise ValueError(f"Unsupported type {type(t)}")

    @property
    def tree(self) -> dict[str, set[str]]:
        """Returns a dict mapping parent nodes to sets of child nodes."""
        frames: dict[str, set[str]] = dict()
        for child_frame, transform in self.transforms.items():
            if transform.parent_frame in frames:
                frames[transform.parent_frame].add(child_frame)
            else:
                frames[transform.parent_frame] = set([child_frame])
        return frames

    @property
    def paths(self) -> list[dict[str, Union[str, None]]]:
        """Search the transform tree and return a list of graphs, one for each
        connected component.
        If the tree is fully connected, the list will contain one dictionary.
        Otherwise, the list contains one entry for each disconnected sub-tree.
        Each dictionary maps child coordinate frames to their parents (or None
        for root nodes).
        """
        tree = self.tree  # cache

        # Traverse all subgraphs using BFS, resulting in a list of
        # dictionaries
        all_paths: list[dict[str, Union[str, None]]] = []
        for parent in tree:
            all_paths.append(bfs(tree, parent))

        # Select paths that are not subsets of others.
        # If a root node is not any other node's child,
        # the path is considered an independent graph.
        paths = []
        children = set().union(*tree.values())
        for parents in all_paths:
            # By construction, the root node is always the first
            # key in the `parents` dictionary and its parent is
            # always None.
            root = next(iter(parents))
            assert parents[root] is None

            if root not in children:
                paths.append(parents)

        return paths

    def get_se3(self, frame_a: str, frame_b: str, paths=None) -> SE3:
        """Compute the SE(3) transformation from frame_a to frame_b.

        By convention, upward (downward) traversals are forward (inverse)
        transformations.

        Parameters
        ==========
        frame_a, frame_b: frame identifiers
        paths: The paths arg allows the user to call TransformTree.paths on
        their own schedule and provide it to this function, which may
        be more efficient in use cases where the tree does not change
        frequently and the cost of recomputing self.paths is noticeable.

        Returns
        =======
        SE(3) from frame frame_a to frame_b

        Raises
        ======
        ValueError if no path is found from frame_a to frame_b
        """

        if paths is None:
            paths = self.paths

        for parents in paths:
            if frame_a not in parents or frame_b not in parents:
                continue

            # Find paths from a and b up to the root
            a_up: list[str] = search_up(parents, frame_a)
            b_up: list[str] = search_up(parents, frame_b)

            # Create transform chains
            a, b = SE3(), SE3()
            for frame in a_up[:-1]:
                a = self.transforms[frame].se3 * a
            for frame in b_up[:-1]:
                b = self.transforms[frame].se3 * b
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


def search_up(
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
