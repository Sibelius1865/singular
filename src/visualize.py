import matplotlib.pyplot as plt
import networkx as nx
import torch


def graph_ir_to_nx(ir):
    """
    Convert GraphIR to NetworkX DiGraph
    """
    g = nx.DiGraph()

    num_nodes = ir.node_features.size(0)

    # add nodes with node types
    for i in range(num_nodes):
        node_type = ir.node_types[i].item() if hasattr(ir, 'node_types') else 1
        g.add_node(
            i,
            feature=ir.node_features[i].detach().cpu().numpy(),
            node_type=node_type
        )

    # add edges
    edge_index = ir.edge_index.detach().cpu()

    for src, dst in edge_index.t().tolist():
        g.add_edge(src, dst)

    return g


def visualize_graph(ir, title="Graph IR"):
    """
    Visualize GraphIR using NetworkX with node type coloring
    """
    try:
        from .node_type import NodeType
    except ImportError:
        from node_type import NodeType
    
    g = graph_ir_to_nx(ir)

    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(g, seed=42)

    # Color nodes by type
    node_colors = []
    for i in range(ir.num_nodes):
        node_type = ir.node_types[i].item() if hasattr(ir, 'node_types') else NodeType.COMPUTE
        if node_type == NodeType.INPUT:
            node_colors.append("lightgreen")
        elif node_type == NodeType.OUTPUT:
            node_colors.append("lightcoral")
        else:
            node_colors.append("lightblue")

    # Add labels with node types
    labels = {}
    for i in range(ir.num_nodes):
        node_type = ir.node_types[i].item() if hasattr(ir, 'node_types') else NodeType.COMPUTE
        if node_type == NodeType.INPUT:
            labels[i] = f"I{i}"
        elif node_type == NodeType.OUTPUT:
            labels[i] = f"O{i}"
        else:
            labels[i] = f"C{i}"

    nx.draw(
        g,
        pos,
        with_labels=True,
        labels=labels,
        node_size=1000,
        node_color=node_colors,
        edge_color="gray",
        arrows=True,
        font_size=10,
        font_weight="bold"
    )

    plt.title(title)
    plt.show()
