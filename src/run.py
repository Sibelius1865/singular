import random
import copy
from typing import List, Dict

import torch
from torch_geometric.data import Data
import networkx as nx

try:
    from .node_type import NodeType
    from .visualize import visualize_graph
except ImportError:
    from node_type import NodeType
    from visualize import visualize_graph


# =========================
# Graph IR definition
# =========================

class GraphIR:
    """
    Minimal machine-friendly Graph IR
    """

    def __init__(
        self,
        node_features: torch.Tensor,  # [N, F]
        node_types: torch.Tensor,     # [N]  (NodeType)
        edge_index: torch.Tensor,      # [2, E]
        edge_attr: torch.Tensor = None # [E, A]
    ):
        self.node_features = node_features
        self.node_types = node_types
        self.edge_index = edge_index
        self.edge_attr = edge_attr

    @property
    def num_nodes(self):
        return self.node_features.size(0)


# =========================
# IR <-> PyG conversion
# =========================

def ir_to_pyg(ir: GraphIR) -> Data:
    """
    Convert GraphIR to PyTorch Geometric Data
    """
    return Data(
        x=ir.node_features,
        edge_index=ir.edge_index,
        edge_attr=ir.edge_attr,
        node_type=ir.node_types  # include node types
    )


# =========================
# Semantic path checking
# =========================

def has_semantic_path(ir: GraphIR) -> bool:
    """
    Check if there exists at least one INPUT -> OUTPUT path
    """
    g = nx.DiGraph()
    g.add_nodes_from(range(ir.num_nodes))
    g.add_edges_from(ir.edge_index.t().tolist())

    inputs = (ir.node_types == NodeType.INPUT).nonzero(as_tuple=True)[0]
    outputs = (ir.node_types == NodeType.OUTPUT).nonzero(as_tuple=True)[0]

    for i in inputs.tolist():
        for o in outputs.tolist():
            if nx.has_path(g, i, o):
                return True

    return False


# =========================
# Fitness function (toy)
# =========================

def fitness(ir: GraphIR) -> float:
    """
    Simple fitness:
    - reward dense connectivity
    - penalize too many nodes
    """
    num_nodes = ir.num_nodes
    num_edges = ir.edge_index.size(1)

    return num_edges - 0.1 * num_nodes


# =========================
# Semantic-preserving mutation operators
# =========================

def mutate_insert_compute(ir: GraphIR, feature_dim: int) -> GraphIR:
    """
    Insert a COMPUTE node in the middle of an existing edge
    A -> B  =>  A -> C -> B
    """
    if ir.edge_index.size(1) == 0:
        return ir

    new_ir = copy.deepcopy(ir)

    # pick an existing edge
    idx = random.randint(0, new_ir.edge_index.size(1) - 1)
    src, dst = new_ir.edge_index[:, idx].tolist()

    # new compute node
    new_node_id = new_ir.num_nodes

    new_ir.node_features = torch.cat(
        [new_ir.node_features, torch.randn(1, feature_dim)], dim=0
    )
    new_ir.node_types = torch.cat(
        [new_ir.node_types, torch.tensor([NodeType.COMPUTE])], dim=0
    )

    # remove old edge
    mask = torch.ones(new_ir.edge_index.size(1), dtype=torch.bool)
    mask[idx] = False
    new_ir.edge_index = new_ir.edge_index[:, mask]

    # add new edges
    new_edges = torch.tensor(
        [[src, new_node_id], [new_node_id, dst]],
        dtype=torch.long
    ).t()

    new_ir.edge_index = torch.cat(
        [new_ir.edge_index, new_edges], dim=1
    )

    return new_ir


def mutate_parallel_compute(ir: GraphIR, feature_dim: int) -> GraphIR:
    """
    Add a parallel COMPUTE path
    A -> B  =>  A -> C -> B (parallel to A -> B)
    """
    if ir.edge_index.size(1) == 0:
        return ir

    new_ir = copy.deepcopy(ir)

    idx = random.randint(0, new_ir.edge_index.size(1) - 1)
    src, dst = new_ir.edge_index[:, idx].tolist()

    new_node_id = new_ir.num_nodes

    new_ir.node_features = torch.cat(
        [new_ir.node_features, torch.randn(1, feature_dim)], dim=0
    )
    new_ir.node_types = torch.cat(
        [new_ir.node_types, torch.tensor([NodeType.COMPUTE])], dim=0
    )

    new_edges = torch.tensor(
        [[src, new_node_id], [new_node_id, dst]],
        dtype=torch.long
    ).t()

    new_ir.edge_index = torch.cat(
        [new_ir.edge_index, new_edges], dim=1
    )

    return new_ir


def semantic_mutate(ir: GraphIR, feature_dim: int) -> GraphIR:
    """
    Semantic-preserving mutation with retry mechanism
    """
    for _ in range(5):  # safety retry
        if random.random() < 0.5:
            candidate = mutate_insert_compute(ir, feature_dim)
        else:
            candidate = mutate_parallel_compute(ir, feature_dim)

        if has_semantic_path(candidate):
            return candidate

    return ir  # fallback (no-op)


# =========================
# Genetic Algorithm loop
# =========================

def evolve(
    population: List[GraphIR],
    generations: int,
    feature_dim: int,
    elite_ratio: float = 0.3
) -> List[GraphIR]:

    for gen in range(generations):
        scored = [(fitness(ir), ir) for ir in population]
        scored.sort(key=lambda x: x[0], reverse=True)

        elites = [ir for _, ir in scored[: int(len(scored) * elite_ratio)]]

        print(
            f"Gen {gen}: best fitness = {scored[0][0]:.2f}, "
            f"nodes = {scored[0][1].num_nodes}, "
            f"edges = {scored[0][1].edge_index.size(1)}"
        )

        # Reproduce
        new_population = elites.copy()
        while len(new_population) < len(population):
            parent = random.choice(elites)
            child = semantic_mutate(parent, feature_dim)
            new_population.append(child)

        population = new_population

    return population


# =========================
# Initialization
# =========================

def initial_graph(feature_dim: int) -> GraphIR:
    """
    Create an initial graph with semantic meaning:
    INPUT -> COMPUTE -> OUTPUT
    """
    node_features = torch.randn(3, feature_dim)
    node_types = torch.tensor([
        NodeType.INPUT,
        NodeType.COMPUTE,
        NodeType.OUTPUT
    ])

    edge_index = torch.tensor([
        [0, 1],
        [1, 2]
    ], dtype=torch.long).t()

    return GraphIR(node_features, node_types, edge_index)


# =========================
# Entry point
# =========================

def main():
    random.seed(0)
    torch.manual_seed(0)

    feature_dim = 8
    population_size = 10

    # Initialize population
    population = [
        initial_graph(feature_dim)
        for _ in range(population_size)
    ]

    # Run GA
    evolved = evolve(
        population,
        generations=10,
        feature_dim=feature_dim
    )

    # Select best individual by fitness
    best_ir = max(evolved, key=fitness)

    # Visualize best graph
    visualize_graph(best_ir, title="Best Graph")

    # Convert best individual to PyG Data
    pyg_data = ir_to_pyg(best_ir)

    print("\nBest PyG Data:")
    print(pyg_data)


if __name__ == "__main__":
    main()