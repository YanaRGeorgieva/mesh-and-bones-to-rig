import torch.nn as nn
from torch_geometric.nn import SplineConv

class MeshEncoder(nn.Module):
    """
    Mesh Encoder based on SplineConv.
    
    Args:
        in_channels (int): Dimensionality of the input features (typically 3 for (x, y, z)).
        hidden_channels (int): Hidden dimension in SplineConv layers.
        out_channels (int): Output feature size.
        kernel_size (int): Kernel size for SplineConv.
        num_layers (int): Number of SplineConv layers to stack.
        dim (int): Dimension of the edge attributes, typically 3.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, num_layers, dim):
        super(MeshEncoder, self).__init__()
        self.convs = nn.ModuleList()
        dims = [in_channels] + [hidden_channels]*(num_layers - 1) + [out_channels]
        for i in range(num_layers):
            self.convs.append(SplineConv(dims[i], dims[i + 1], dim=dim, kernel_size=kernel_size))
        self.activation = nn.ReLU()
    
    def forward(self, x, edge_index, edge_attr):
        """
        Parameters:
            x: (N, in_channels) tensor of vertex features.
            edge_index: (2, N*k) LongTensor of edge indices.
            edge_attr: (N*k, 1) tensor of edge attributes.
        Returns:
            x: (N, out_channels) tensor of vertex features.
        """
        for conv in self.convs:
            x = self.activation(conv(x, edge_index, edge_attr))
        return x