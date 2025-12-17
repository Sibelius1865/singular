import torch
from torch_geometric.data import Data
from graph_ir import GraphIR


def graphir_to_pyg(g: GraphIR) -> Data:
    """
    Node attrs -> x
    Edge list -> edge_index
    """
    # Node features
    x = torch.tensor(
        [n.attrs for n in g.nodes],
        dtype=torch.float,
    )

    # Edges
    if g.edges:
        edge_index = torch.tensor(
            [[e.src, e.dst] for e in g.edges],
            dtype=torch.long,
        ).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)
