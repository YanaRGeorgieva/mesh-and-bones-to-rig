import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse

class BoneEncoder(nn.Module):
    """
    Processes the bone graph using GCNConv layers.
    Converts a dense bone adjacency matrix to an edge_index internally.
    
    Parameters:
        in_channels: Input feature dimension for bones.
        hidden_channels: Hidden dimension in GCN layers.
        out_channels: Output bone embedding dimension.
        num_layers: Number of GCN layers.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(BoneEncoder, self).__init__()
        self.convs = nn.ModuleList()
        dims = [in_channels] + [hidden_channels]*(num_layers - 1) + [out_channels]
        for i in range(num_layers):
            self.convs.append(GCNConv(dims[i], dims[i + 1]))
        self.activation = nn.ReLU()
    
    def forward(self, bone_features, bone_adj):
        """
        Parameters:
            bone_features: (B, in_channels) tensor.
            bone_adj: (B, B) dense normalized adjacency matrix.
        Returns:
            bone_embeddings: (B, out_channels) tensor.
        """
        # Convert dense adjacency to edge_index.
        edge_index, _ = dense_to_sparse(bone_adj)  # edge_index: (2, num_edges)
        x = bone_features  # (B, in_channels)
        for i, conv in enumerate(self.convs):
            x = self.activation(conv(x, edge_index))
        return x