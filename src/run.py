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

def _ir_to_nx(ir: GraphIR) -> nx.DiGraph:
    """
    Convert GraphIR to NetworkX DiGraph
    """
    g = nx.DiGraph()
    g.add_nodes_from(range(ir.num_nodes))
    g.add_edges_from(ir.edge_index.t().tolist())
    return g


def has_semantic_path(ir: GraphIR) -> bool:
    """
    Check if there exists at least one INPUT -> OUTPUT path
    """
    g = _ir_to_nx(ir)
    inputs = (ir.node_types == NodeType.INPUT).nonzero(as_tuple=True)[0]
    outputs = (ir.node_types == NodeType.OUTPUT).nonzero(as_tuple=True)[0]

    for i in inputs.tolist():
        for o in outputs.tolist():
            if nx.has_path(g, i, o):
                return True

    return False


def semantic_paths(ir: GraphIR) -> List[List[int]]:
    """
    Get all simple paths from INPUT nodes to OUTPUT nodes
    Returns list of paths (each path is a list of node indices)
    """
    g = _ir_to_nx(ir)
    inputs = (ir.node_types == NodeType.INPUT).nonzero(as_tuple=True)[0].tolist()
    outputs = (ir.node_types == NodeType.OUTPUT).nonzero(as_tuple=True)[0].tolist()

    all_paths = []
    for i in inputs:
        for o in outputs:
            try:
                paths = list(nx.all_simple_paths(g, i, o))
                all_paths.extend(paths)
            except nx.NetworkXNoPath:
                continue

    return all_paths


def count_compute_in_path(path: List[int], ir: GraphIR) -> int:
    """
    Count COMPUTE nodes in a path
    """
    return sum(1 for n in path if ir.node_types[n] == NodeType.COMPUTE)


def get_node_degrees(ir: GraphIR) -> tuple:
    """
    Get in-degree and out-degree for each node
    Returns (in_degrees, out_degrees) as dicts
    """
    g = _ir_to_nx(ir)
    in_degrees = dict(g.in_degree())
    out_degrees = dict(g.out_degree())
    return in_degrees, out_degrees


def count_dead_compute(ir: GraphIR) -> int:
    """
    Count COMPUTE nodes that cannot reach any OUTPUT
    """
    g = _ir_to_nx(ir)
    outputs = (ir.node_types == NodeType.OUTPUT).nonzero(as_tuple=True)[0].tolist()
    
    dead_count = 0
    for node_id in range(ir.num_nodes):
        if ir.node_types[node_id] == NodeType.COMPUTE:
            can_reach_output = any(
                nx.has_path(g, node_id, o) for o in outputs
            )
            if not can_reach_output:
                dead_count += 1
    
    return dead_count


# =========================
# Semantic fitness function
# =========================

def fitness(ir: GraphIR) -> float:
    """
    Semantic fitness function that evaluates:
    (A) Number of semantic paths (diversity)
    (B) Average compute depth per path
    (C) Branch/merge structure (nodes with indegree>=2 or outdegree>=2)
    (D) Dead compute penalty (COMPUTE nodes that don't reach OUTPUT)
    (E) Regularization (size penalty)
    """
    paths = semantic_paths(ir)
    
    if not paths:
        return -1e9  # Insurance (should not happen with semantic_mutate)
    
    # (A) Path count score (diversity)
    # Cap at reasonable number to avoid explosion
    # Use log scale for very large path counts to prevent dominance
    path_count = len(paths)
    if path_count <= 20:
        path_score = path_count
    else:
        # Logarithmic scaling for paths > 20
        path_score = 20 + 2 * (path_count - 20) ** 0.5
    
    # (B) Average compute depth per path
    total_compute_in_paths = sum(count_compute_in_path(p, ir) for p in paths)
    depth_score = total_compute_in_paths / len(paths)
    
    # (C) Branch/merge structure score
    in_degrees, out_degrees = get_node_degrees(ir)
    branch_score = sum(
        1 for n in range(ir.num_nodes)
        if ir.node_types[n] == NodeType.COMPUTE
        and (in_degrees.get(n, 0) >= 2 or out_degrees.get(n, 0) >= 2)
    )
    
    # (D) Dead compute penalty
    dead_nodes = count_dead_compute(ir)
    
    # (E) Size regularization
    size_penalty = 0.01 * ir.num_nodes + 0.005 * ir.edge_index.size(1)
    
    # Weighted combination
    return (
        1.0 * path_score +
        0.5 * depth_score +
        0.3 * branch_score -
        0.2 * dead_nodes -
        size_penalty
    )


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

        best_ir = scored[0][1]
        paths = semantic_paths(best_ir)
        print(
            f"Gen {gen}: best fitness = {scored[0][0]:.2f}, "
            f"nodes = {best_ir.num_nodes}, "
            f"edges = {best_ir.edge_index.size(1)}, "
            f"semantic_paths = {len(paths)}"
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
    best_fitness = fitness(best_ir)

    # Visualize best graph with fitness and path information
    visualize_graph(
        best_ir,
        title="Best Graph (Semantic Evolution)",
        fitness_value=best_fitness,
        show_paths=True
    )

    # Convert best individual to PyG Data
    pyg_data = ir_to_pyg(best_ir)

    print("\nBest PyG Data:")
    print(pyg_data)


if __name__ == "__main__":
    main()