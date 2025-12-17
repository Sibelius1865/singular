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


def visualize_graph(ir, title="Graph IR", fitness_value=None, show_paths=False):
    """
    Visualize GraphIR using NetworkX with node type coloring
    Optionally show fitness value and highlight semantic paths
    """
    try:
        from .node_type import NodeType
        from .run import semantic_paths, get_node_degrees
    except ImportError:
        from node_type import NodeType
        from run import semantic_paths, get_node_degrees
    
    g = graph_ir_to_nx(ir)

    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(g, seed=42, k=1.5, iterations=50)

    # Color nodes by type
    node_colors = []
    node_sizes = []
    for i in range(ir.num_nodes):
        node_type = ir.node_types[i].item() if hasattr(ir, 'node_types') else NodeType.COMPUTE
        if node_type == NodeType.INPUT:
            node_colors.append("lightgreen")
            node_sizes.append(1200)
        elif node_type == NodeType.OUTPUT:
            node_colors.append("lightcoral")
            node_sizes.append(1200)
        else:
            node_colors.append("lightblue")
            # Size based on degree (branch nodes are larger)
            in_degrees, out_degrees = get_node_degrees(ir)
            degree = in_degrees.get(i, 0) + out_degrees.get(i, 0)
            node_sizes.append(800 + degree * 100)

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

    # Calculate edge widths based on path frequency
    edge_widths = {}
    if show_paths:
        paths = semantic_paths(ir)
        for path in paths:
            for j in range(len(path) - 1):
                edge = (path[j], path[j + 1])
                edge_widths[edge] = edge_widths.get(edge, 0) + 1
    else:
        # Default: all edges same width
        for edge in g.edges():
            edge_widths[edge] = 1

    # Draw edges with varying widths
    for edge, width in edge_widths.items():
        nx.draw_networkx_edges(
            g,
            pos,
            edgelist=[edge],
            width=1 + width * 0.5,
            alpha=0.6,
            edge_color="gray",
            arrows=True,
            arrowsize=20
        )

    # Draw nodes
    nx.draw_networkx_nodes(
        g,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9
    )

    # Draw labels
    nx.draw_networkx_labels(
        g,
        pos,
        labels,
        font_size=10,
        font_weight="bold"
    )

    # Build title with fitness info
    title_text = title
    if fitness_value is not None:
        title_text += f" (fitness: {fitness_value:.2f})"
    if show_paths:
        paths = semantic_paths(ir)
        title_text += f" | {len(paths)} semantic paths"

    plt.title(title_text, fontsize=12, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
