import random
import numpy as np
from graph_ir import GraphIR, Node, Edge, ATTR_DIM
import torch
from pyg_adapter import graphir_to_pyg
from model import SimpleGNN


_model = None


def learning_fitness(g):
    global _model

    if _model is None:
        _model = SimpleGNN(in_dim=len(g.nodes[0].attrs))

    data = graphir_to_pyg(g)

    pred = _model(data)
    loss = pred.pow(2).mean()  # dummy target = 0

    return -loss.item()


MAX_NODES = 10


# -------------------------
# Initialization
# -------------------------

def random_graph() -> GraphIR:
    g = GraphIR()
    n = random.randint(1, 3)
    for _ in range(n):
        g.nodes.append(
            Node(
                op=random.randint(0, 2),
                attrs=np.random.randn(ATTR_DIM),
            )
        )
    return g


# -------------------------
# Fitness (toy, deterministic)
# -------------------------

def fitness(g: GraphIR) -> float:
    if not g.nodes:
        return -1e9

    attr_penalty = sum(np.linalg.norm(n.attrs) for n in g.nodes)
    return len(g.nodes) * 5.0 - attr_penalty


# -------------------------
# Mutation operators
# -------------------------

def mutate_add_node(g: GraphIR):
    if len(g.nodes) >= MAX_NODES:
        return
    g.nodes.append(
        Node(
            op=random.randint(0, 2),
            attrs=np.random.randn(ATTR_DIM),
        )
    )


def mutate_add_edge(g: GraphIR):
    if len(g.nodes) < 2:
        return
    a, b = random.sample(range(len(g.nodes)), 2)
    g.edges.append(
        Edge(
            src=a,
            dst=b,
            etype=random.randint(0, 1)
        )
    )


def mutate_attr_noise(g: GraphIR, scale=0.1):
    if not g.nodes:
        return
    n = random.choice(g.nodes)
    n.attrs += np.random.randn(ATTR_DIM) * scale


def mutate(g: GraphIR) -> GraphIR:
    g = g.copy()
    r = random.random()
    if r < 0.4:
        mutate_add_node(g)
    elif r < 0.7:
        mutate_attr_noise(g)
    else:
        mutate_add_edge(g)
    return g
