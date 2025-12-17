import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


class SimpleGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=16):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()

        # Single graph → fake batch
        batch = x.new_zeros(x.size(0), dtype=torch.long)
        x = global_mean_pool(x, batch)

        return self.lin(x)
