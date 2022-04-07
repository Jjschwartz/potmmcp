"""A node in the search tree."""
from typing import Optional, Any, List

import baposgmcp.tree.belief as B


class Node:
    """A node in the search tree."""

    # class variable
    node_count = 0

    def __init__(self,
                 h: Any,
                 parent: Optional['Node'],
                 belief: B.BaseParticleBelief,
                 v_init: float = 0.0,
                 n_init: float = 0.0):
        self.nid = Node.node_count
        Node.node_count += 1
        self.parent: 'Node' = NullNode() if parent is None else parent
        # pylint: disable=[invalid-name]
        self.h = h
        self.v = v_init
        self.n = n_init
        self.belief = belief
        self.children: List['Node'] = []

    def get_child(self, target_h: Any, **kwargs):
        """Get child node with given history value.

        Adds child node if it doesn't exist.
        """
        for child in self.children:
            if child.h == target_h:
                return child

        child_node = self.get_new_node(
            target_h,
            self,
            B.ParticleBelief(),
            n_init=kwargs.get("n_init", 0.0)
        )
        self.children.append(child_node)
        return child_node

    @classmethod
    def get_new_node(cls,
                     h: Any,
                     parent: 'Node',
                     belief: B.BaseParticleBelief,
                     v_init: float = 0.0,
                     n_init: float = 0.0) -> 'Node':
        """Get a new Node instance."""
        return cls(h, parent, belief, v_init, n_init)

    def has_child(self, target_h: Any) -> bool:
        """Check if node has a child node matching history."""
        for child in self.children:
            if child.h == target_h:
                return True
        return False

    def value_str(self):
        """Get value array in nice str format."""
        return f"{self.v:.3f}"

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

    This class is mainly defined for typechecking convinience...
    """

    def __init__(self):
        super().__init__(None, self, B.ParticleBelief())
