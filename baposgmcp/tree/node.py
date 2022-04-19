"""A node in the search tree."""
from typing import Optional, Any, List

from baposgmcp import parts
import baposgmcp.tree.belief as B


class Node:
    """A node in the search tree."""

    # class variable
    node_count = 0

    def __init__(self,
                 h: Any,
                 parent: Optional['Node'],
                 belief: B.BaseParticleBelief,
                 policy: parts.ActionDist,
                 v_init: float = 0.0,
                 n_init: float = 0.0):
        self.nid = Node.node_count
        Node.node_count += 1
        self.parent: 'Node' = NullNode() if parent is None else parent
        self.h = h
        self.belief = belief
        self.policy = policy
        self.v = v_init
        self.n = n_init
        self.children: List['Node'] = []

    def get_child(self, target_h: Any):
        """Get child node with given history value."""
        for child in self.children:
            if child.h == target_h:
                return child
        raise AssertionError(f"Child with {target_h=} not in {str(self)}")

    def has_child(self, target_h: Any) -> bool:
        """Check if node has a child node matching history."""
        for child in self.children:
            if child.h == target_h:
                return True
        return False

    def value_str(self) -> str:
        """Get value in nice str format."""
        return f"{self.v:.3f}"

    def policy_str(self) -> str:
        """Get policy in nice str format."""
        action_probs = [f"{a}: {prob:.3f}" for a, prob in self.policy.items()]
        return "{" + ",".join(action_probs) + "}"

    def clear_belief(self):
        """Delete all particles in belief of node."""
        self.belief.clear()

    def is_root(self) -> bool:
        """Return true if this node is a root node."""
        return isinstance(self.parent, NullNode)

    def __str__(self):
        return (
            f"N{self.nid}"
            f"\nh={self.h}"
            f"\nv={self.value_str()}"
            f"\nn={self.n}"
            f"\n|B|={self.belief.size()}"
        )

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}: N{self.nid} h={self.h} "
            f"v={self.value_str()} n={self.n}>"
        )

    def __hash__(self):
        return self.nid

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.nid == other.nid


class NullNode(Node):
    """The Null Node which is the parent of the root node of the tree.

    This class is mainly defined for typechecking convenience...
    """

    def __init__(self):
        super().__init__(None, self, B.ParticleBelief(), {})
