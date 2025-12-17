from dataclasses import dataclass, field
from typing import List
import numpy as np
import hashlib

# -------------------------
# Constants
# -------------------------

ATTR_DIM = 4


# -------------------------
# IR primitives
# -------------------------

@dataclass
class Node:
    op: int
    attrs: np.ndarray

    def copy(self):
        return Node(
            op=self.op,
            attrs=self.attrs.copy()
        )


@dataclass
class Edge:
    src: int
    dst: int
    etype: int

    def copy(self):
        return Edge(
            src=self.src,
            dst=self.dst,
            etype=self.etype
        )


@dataclass
class GraphIR:
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)

    def copy(self):
        return GraphIR(
            nodes=[n.copy() for n in self.nodes],
            edges=[e.copy() for e in self.edges],
        )

    def hash(self) -> str:
        """
        Structural + param hash
        (used for diversity / novelty)
        """
        h = hashlib.sha256()
        for n in self.nodes:
            h.update(bytes([n.op]))
            h.update(n.attrs.tobytes())
        for e in self.edges:
            h.update(bytes([e.src, e.dst, e.etype]))
        return h.hexdigest()
