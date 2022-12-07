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


@dataclass
class TreePath:
    a_up: list[str]
    b_up: list[str]


@dataclass
class FrameTree:
    """Maintains a tree of coordinate frame names and supports
    querying the tree for paths. No geometry info here.
    """

    # child frames => parent frames. parent is None if child is a root node
    _path: dict[str, Union[str, None]]

    @property
    def path(self):
        return self._path

    @property
    def root(self):
        # By construction, the root node is always the first
        # key in the child => parent dictionary and its parent is
        # always None.
        return next(iter(self._path))

    def __contains__(self, k):
        return k in self._path

    def get_path(self, frame_a: str, frame_b: str) -> Union[TreePath, None]:
        if frame_a not in self._path or frame_b not in self._path:
            return None

        # Find paths from a and b up to the root
        # TODO: find and remove any common nodes (set intersection)
        a_up: list[str] = traverse_up(self._path, frame_a)
        b_up: list[str] = traverse_up(self._path, frame_b)

        return TreePath(a_up, b_up)


class TransformTree:
    """Data structure for a collection of named 6 DOF coordinate frames
    linked by parent-child relationships. The `tree` and `transforms` data
    members maintain the topological and geometrical information
    (respectively).
    The tree can be dynamically grown using the `update` method, but an error
    will be raised if a coordinate frame is added with no link to the rest of
    the tree.
    """

    def __init__(self, transforms: list[Transform] = []):
        self._transforms: dict[str, Transform] = dict()
        self._tree = FrameTree({})
        self.graph: dict[str, set[str]] = dict()
        if transforms:
            self.update(transforms)

    def render(self, node: str, _lines="", _prefix="") -> str:
        """Create a multiline string representation similar to
        the `tree` filesystem visualization command in linux.

        _lines and _prefix serve as static variables and are not
        meant to be assigned.
        """
        # Add root node
        if not _lines:
            _lines = self.root + "\n"

        # Add all others
        for i, child in enumerate(self.graph[node]):
            if i < len(self.graph[node]) - 1:
                _lines += f"{_prefix}├── {child}\n"
                if child in self.graph:
                    _lines = self.render(child, _lines, _prefix + "│   ")
            else:
                _lines += f"{_prefix}└── {child}\n"
                if child in self.graph:
                    _lines = self.render(child, _lines, _prefix + "    ")
        return _lines

    def __str__(self):
        return self.render(self.root)

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

        self.graph = map_parents_to_children(self._transforms)
        paths_up = map_children_to_parents(self.graph)

        if len(paths_up) > 1:
            raise ValueError(
                f"Multiple trees found. If you require support for disjoint "
                "transform tree collections, use TransformForest instead."
            )
        elif len(paths_up) == 0:
            raise ValueError(f"Empty TransformTree")

        (p,) = paths_up
        self._tree = FrameTree(p)

    @property
    def transforms(self):
        return self._transforms

    @property
    def tree(self):
        return self._tree

    @property
    def root(self):
        return self._tree.root

    def get_se3(self, frame_a: str, frame_b: str) -> SE3:
        """Compute the SE(3) transformation from frame_a to frame_b.

        By convention, upward (downward) traversals are forward (inverse)
        transformations.

        Parameters
        ==========
        frame_a, frame_b: coordinate frame identifiers

        Returns
        =======
        SE(3) from frame frame_a to frame_b

        Raises
        ======
        ValueError if no path is found from frame_a to frame_b
        """

        # Find the path from a -> b
        tree_path = self._tree.get_path(frame_a, frame_b)
        if tree_path is None:
            raise ValueError(f"No path found for {frame_a} -> {frame_b}")

        return get_se3(tree_path, self._transforms)


class TransformForest:
    """Transform manager class that handles dynamic updating and disjoint
    transform trees.
    """

    def __init__(self, transforms: list[Transform] = []):
        # Maps child frame ID => Transform
        self._transforms: dict[str, Transform] = dict()
        self._trees: list[FrameTree] = []
        self.graph: dict[str, set[str]] = dict()
        if transforms:
            self.update(transforms)

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

        self.graph = map_parents_to_children(self._transforms)
        paths_up = map_children_to_parents(self.graph)
        self._trees = [FrameTree(p) for p in paths_up]

    @property
    def transforms(self):
        return self._transforms

    @property
    def size(self):
        return len(self._trees)

    @property
    def trees(self):
        return self._trees

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

        # Find the tree containing a and b
        tree = None
        for t in self._trees:
            if frame_a in t and frame_b in t:
                tree = t
                break
        if tree is None:
            raise ValueError(f"No tree found for {frame_a}, {frame_b}")

        # Find the path from a -> b
        tree_path = tree.get_path(frame_a, frame_b)
        if tree_path is None:
            raise ValueError(f"No path found for {frame_a} -> {frame_b}")

        return get_se3(tree_path, self._transforms)


def get_se3(tree_path: TreePath, transforms: dict[str, Transform]):
    """Compute the chain of transforms specified in `tree_path`."""
    a, b = SE3(), SE3()
    for frame in tree_path.a_up[:-1]:
        a = transforms[frame].se3 * a
    for frame in tree_path.b_up[:-1]:
        b = transforms[frame].se3 * b
    return b.inverse * a


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
