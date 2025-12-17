import random
import copy
from typing import List, Dict

import torch
from torch_geometric.data import Data


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
        edge_index: torch.Tensor,      # [2, E]
        edge_attr: torch.Tensor = None # [E, A]
    ):
        self.node_features = node_features
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
        edge_attr=ir.edge_attr
    )


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
# Mutation operators
# =========================

def mutate_add_edge(ir: GraphIR) -> GraphIR:
    """
    Randomly add an edge
    """
    new_ir = copy.deepcopy(ir)

    src = random.randint(0, new_ir.num_nodes - 1)
    dst = random.randint(0, new_ir.num_nodes - 1)

    new_edge = torch.tensor([[src], [dst]], dtype=torch.long)
    new_ir.edge_index = torch.cat([new_ir.edge_index, new_edge], dim=1)

    return new_ir


def mutate_add_node(ir: GraphIR, feature_dim: int) -> GraphIR:
    """
    Add a new node with random features
    """
    new_ir = copy.deepcopy(ir)

    new_feature = torch.randn(1, feature_dim)
    new_ir.node_features = torch.cat(
        [new_ir.node_features, new_feature], dim=0
    )

    return new_ir


def mutate(ir: GraphIR, feature_dim: int) -> GraphIR:
    """
    Choose a mutation randomly
    """
    if random.random() < 0.5:
        return mutate_add_edge(ir)
    else:
        return mutate_add_node(ir, feature_dim)


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
            child = mutate(parent, feature_dim)
            new_population.append(child)

        population = new_population

    return population


# =========================
# Initialization
# =========================

def random_graph_ir(
    num_nodes: int,
    feature_dim: int,
    num_edges: int
) -> GraphIR:

    node_features = torch.randn(num_nodes, feature_dim)

    edge_index = torch.randint(
        0, num_nodes, (2, num_edges), dtype=torch.long
    )

    return GraphIR(node_features, edge_index)


# =========================
# Entry point
# =========================

def main():
    random.seed(0)
    torch.manual_seed(0)

    feature_dim = 8
    population_size = 10

    population = [
        random_graph_ir(
            num_nodes=4,
            feature_dim=feature_dim,
            num_edges=3
        )
        for _ in range(population_size)
    ]

    evolved = evolve(
        population,
        generations=10,
        feature_dim=feature_dim
    )

    # Convert best individual to PyG Data
    best_ir = max(evolved, key=fitness)
    pyg_data = ir_to_pyg(best_ir)

    print("\nBest PyG Data:")
    print(pyg_data)


if __name__ == "__main__":
    main()
