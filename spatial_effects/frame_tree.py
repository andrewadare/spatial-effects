from dataclasses import dataclass, field

import numpy as np

from .se3 import SE3


class FrameTreeValidationError(Exception):
    """Failure to pass validation check"""


@dataclass
class TransformToParent:
    """A rigid transform from a child frame to a named parent frame."""

    parent_name: str
    T: SE3

    def to_dict(self) -> dict:
        return {
            "parent_name": self.parent_name,
            "transform": self.T.matrix.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TransformToParent":
        return cls(data["parent_name"], SE3(np.array(data["transform"])))


@dataclass
class FrameTree:
    """A tree data structure of coordinate frame names linked by child-to-parent
    SE(3) transforms.
    """

    # Key on child frame name
    transforms: dict[str, TransformToParent] = field(default_factory=dict)

    def __str__(self):
        return self.render(self.root_frame())

    def add(self, transform: SE3, child_frame: str, parent_frame: str) -> None:
        """Add a child-to-parent transform to the tree."""
        self.transforms[child_frame] = TransformToParent(parent_frame, transform)

    def validate(self) -> bool:
        """Check that exactly one node in the tree has no parents."""
        num_top_nodes = 0
        for t in self.transforms.values():
            if t.parent_name not in self.transforms:
                num_top_nodes += 1
        return num_top_nodes == 1

    def traverse_up(self, frame_name: str) -> list[SE3]:
        """Returns ordered list of SE3 transforms from frame_name up to the root frame."""
        if not self.validate():
            raise FrameTreeValidationError()

        chain: list[SE3] = []
        try:
            tform: TransformToParent | None = self.transforms[frame_name]
        except KeyError:
            return chain
        while tform:
            chain.append(tform.T)
            tform = self.transforms.get(tform.parent_name)
        return chain

    def root_frame(self) -> str:
        """Returns name of root coordinate frame."""
        if not self.validate():
            raise FrameTreeValidationError()

        for child_name, transform in self.transforms.items():
            if transform.parent_name not in self.transforms:
                return transform.parent_name
        return ""

    def get_transform_to_root(self, frame_name: str) -> SE3:
        """Get transform from frame_name to root of FrameTree."""
        T = SE3()
        for parent_T_child in self.traverse_up(frame_name):
            T = parent_T_child * T
        return T

    def get_transform(self, frame_a: str, frame_b: str) -> SE3:
        """Compute the SE(3) transformation from frame_a to frame_b."""

        for frame in (frame_a, frame_b):
            if frame not in self.transforms:
                raise LookupError(f"Not found:", frame)

        # Not optimized
        a = self.get_transform_to_root(frame_a)
        b = self.get_transform_to_root(frame_b)

        return b.inverse * a

    def map_parents_to_children(self) -> dict[str, set[str]]:
        """Returns a dict mapping parent nodes to sets of child nodes."""
        p2c: dict[str, set[str]] = dict()
        for child_frame, transform in self.transforms.items():
            if transform.parent_name in p2c:
                p2c[transform.parent_name].add(child_frame)
            else:
                p2c[transform.parent_name] = set([child_frame])
        return p2c

    def render(self, node: str, _lines="", _prefix="") -> str:
        """Create a multiline string representation similar to
        the `tree` filesystem visualization command in linux.

        _lines and _prefix serve as static variables and are not
        meant to be assigned.
        """
        p2c: dict[str, set[str]] = self.map_parents_to_children()

        # Add root node
        if not _lines:
            _lines = self.root_frame() + "\n"

        # Add all others
        for i, child in enumerate(p2c[node]):
            if i < len(p2c[node]) - 1:
                _lines += f"{_prefix}├── {child}\n"
                if child in p2c:
                    _lines = self.render(child, _lines, _prefix + "│   ")
            else:
                _lines += f"{_prefix}└── {child}\n"
                if child in p2c:
                    _lines = self.render(child, _lines, _prefix + "    ")
        return _lines

    def to_dict(self) -> dict:
        return {k: v.to_dict() for k, v in self.transforms.items()}

    @classmethod
    def from_dict(cls, d: dict[str, dict[str, list]]) -> "FrameTree":
        tfs: dict[str, TransformToParent] = dict()

        for child_name, t2p_dict in d.items():
            tfs[child_name] = TransformToParent.from_dict(t2p_dict)

        return cls(tfs)
