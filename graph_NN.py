import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv


class SimpleGCN(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)



def create_example_graph():
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2]
    ], dtype=torch.long)

    x = torch.randn(4, 10)

    model = SimpleGCN(num_features=10, hidden_channels=16, num_classes=3)

    output = model(x, edge_index)
    return output


result = create_example_graph()
print("GCN Output Shape:", result.shape)