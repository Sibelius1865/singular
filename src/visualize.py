import matplotlib.pyplot as plt
import networkx as nx
import torch


def graph_ir_to_nx(ir):
    """
    Convert GraphIR to NetworkX DiGraph
    """
    g = nx.DiGraph()

    num_nodes = ir.node_features.size(0)

    # add nodes
    for i in range(num_nodes):
        g.add_node(
            i,
            feature=ir.node_features[i].detach().cpu().numpy()
        )

    # add edges
    edge_index = ir.edge_index.detach().cpu()

    for src, dst in edge_index.t().tolist():
        g.add_edge(src, dst)

    return g


def visualize_graph(ir, title="Graph IR"):
    """
    Visualize GraphIR using NetworkX
    """
    g = graph_ir_to_nx(ir)

    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(g, seed=42)

    nx.draw(
        g,
        pos,
        with_labels=True,
        node_size=800,
        node_color="lightblue",
        edge_color="gray",
        arrows=True
    )

    plt.title(title)
    plt.show()
