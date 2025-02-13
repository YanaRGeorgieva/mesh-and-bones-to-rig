import torch.nn as nn

from .mesh_encoder import MeshEncoder
from .bone_encoder import BoneEncoder
from .fusion_attention import FusionModule
from .refinement import RefinementModule

class MeshBonesToRigNet(nn.Module):
    """
    The overall network that predicts per-vertex skin weights.
    Inputs:
        - vertices: (N, 3) tensor.
        - edge_index_geodesic: (2, N*K) LongTensor.
        - edge_attr_geodesic: (N*K, 1) tensor.
        - vertex_neighbors: (N, K) LongTensor.
        - vertex_adj: (N, N) normalized vertex adjacency matrix.
        - vertex_normals: (N, 3) tensor of vertex normals.
        - bone_features: (B, bone_in_dim) tensor.
        - bone_adj: (B, B) normalized bone adjacency matrix.
        - vol_geo: (N, B) tensor of pre-calculated volumetric geodesic distances.
        - surface_geodesic: (N, N) dense surface geodesic distance matrix.
    Outputs:
        - refined_skin_weights: (N, B) tensor of predicted skin weights
    """
    def __init__(self,
                 mesh_encoder_in_channels = 3,
                 mesh_encoder_hidden_channels = 128,
                 mesh_encoder_out_channels = 256,
                 mesh_encoder_kernel_size = 5,
                 mesh_encoder_num_layers = 3,
                 mesh_encoder_dim = 1,
                 bone_encoder_in_channels = 8,
                 bone_encoder_hidden_channels = 64,
                 bone_encoder_out_channels = 64,
                 bone_encoder_num_layers = 2,
                 fusion_common_dim = 128,
                 fusion_top_k = 4,
                 fusion_alpha = 1.0,
                 fusion_alpha_learnable = True,
                 refinement_gamma = 0.5,
                 with_refinement = False,
                 ):
        """This network consists of:
        - A Mesh Encoder (built with DevConv layers) that processes raw vertex positions.
        - A Bone Encoder (using GCNConv layers via torch_geometric) that processes bone features.
        - A Fusion Module with sparse attention that computes attention scores (and thus skin weights)
            between vertex features and bone embeddings. This module integrates pre-calculated
            volumetric geodesic distances to bias the attention. The Fusion Module produces an initial
            per-vertex skin weight distribution of shape (N, B) (B = number of bones, which may vary between samples).
        - A Refinement Module is then applied to refine the distribution without changing B.

        Parameters:
            mesh_encoder_in_channels (int, default=3):
                Dimensionality of input vertex positions (usually 3 for x, y, z).
            mesh_encoder_hidden_channels (int, default=128):
                Output dimension of the Mesh Encoder, i.e., the size of the per-vertex feature vectors.
            mesh_encoder_out_channels (int, default=256):
                Output dimension of the Mesh Encoder, i.e., the size of the per-vertex feature vectors.
            mesh_encoder_kernel_size (int, default=5):
                Kernel size for the Mesh Encoder.
            mesh_encoder_num_layers (int, default=3):
                Number of Mesh Encoder layers.
            mesh_encoder_dim (int, default=1):
                Dimension of the edge attributes in the Mesh Encoder.
            bone_encoder_in_channels (int, default=8):
                Input dimension for bone features; joint positions (3D) are padded to a fixed dimension.
            bone_encoder_hidden_channels (int, default=64):
                Hidden dimension for the Bone Encoder GCN layers.
            bone_encoder_out_channels (int, default=64):
                Output dimension of the Bone Encoder, yielding bone embeddings.
            bone_encoder_num_layers (int, default=2):
                Number of Bone Encoder GCN layers.
            fusion_common_dim (int, default=128):
                Projection dimension for the Fusion Module, where vertex and bone embeddings are compared.
            fusion_top_k (int, default=4):
                Number of bones to keep per vertex in the Fusion Module.
            fusion_alpha (float, default=1.0):
                Scaling factor for the volumetric geodesic bias in the Fusion Module.
                It controls how strongly larger geodesic distances penalize the attention scores.
                A value of 1.0 is a common starting point; higher values will reduce the influence of distant bones.
            fusion_alpha_learnable (bool, default=True):
                Whether to make the Fusion Module's alpha parameter learnable.
            refinement_gamma (float, default=0.5):
                Smoothing factor for the Refinement Module.
            with_refinement (bool, default=False):
                Whether to apply the Refinement Module to refine the distribution without changing B.
        """
        super(MeshBonesToRigNet, self).__init__()
        # Mesh Encoder: processes vertex positions.
        self.mesh_encoder = MeshEncoder(in_channels=mesh_encoder_in_channels,
                                           hidden_channels=mesh_encoder_hidden_channels,
                                           out_channels=mesh_encoder_out_channels,
                                           kernel_size=mesh_encoder_kernel_size,
                                           num_layers=mesh_encoder_num_layers,
                                           dim=mesh_encoder_dim)
        # Bone Encoder: processes bone features.
        self.bone_encoder = BoneEncoder(in_channels=bone_encoder_in_channels,
                                        hidden_channels=bone_encoder_hidden_channels,
                                        out_channels=bone_encoder_out_channels,
                                        num_layers=bone_encoder_num_layers)
        # Fusion Module: uses sparse attention with volumetric geodesic bias.
        self.fusion = FusionModule(vertex_feat_dim=mesh_encoder_out_channels,
                                         bone_feat_dim=bone_encoder_out_channels,
                                         common_dim=fusion_common_dim,
                                         top_k=fusion_top_k,
                                         alpha=fusion_alpha,
                                         alpha_learnable=fusion_alpha_learnable)
        # Refinement Module: refines the distribution without changing the number of bones.
        self.refinement = RefinementModule(gamma=refinement_gamma)
        # Whether to apply the Refinement Module to refine the distribution without changing B.
        self.with_refinement = with_refinement

    def forward(self, vertices, edge_index_geodesic, edge_attr_geodesic, vertex_neighbors, vertex_adj, vertex_normals, bone_features, bone_adj, vol_geo, surface_geodesic):
        """
        Parameters:
            vertices: (N, 3) tensor of vertex positions.
            edge_index_geodesic: (2, N*K) LongTensor of geodesic edge indices.
            edge_attr_geodesic: (N*K, 1) tensor of geodesic distances.
            vertex_neighbors: (N, K) LongTensor of neighbor indices.
            vertex_adj: (N, N) normalized vertex adjacency matrix (dense or SparseTensor).
            vertex_normals: (N, 3) tensor of vertex normals.
            bone_features: (B, bone_in_dim) tensor of bone features.
            bone_adj: (B, B) normalized bone adjacency matrix.
            vol_geo: (N, B) tensor of volumetric geodesic distances.
            surface_geodesic: (N, N) dense surface geodesic distance matrix.
        Returns:
            skin_weights: (N, num_bones) tensor of predicted skin weights.
        """
        # Use the geodesic-based edge_index and edge_attr in the mesh encoder.
        vertex_features = self.mesh_encoder(vertices, edge_index_geodesic, edge_attr_geodesic)  # (N, mesh_enc_out)
        # Bone Encoder produces bone embeddings.
        bone_embeddings = self.bone_encoder(bone_features, bone_adj)       # (B, bone_out_dim)
        # Fusion Module uses sparse attention and biases it by vol_geo.
        predicted_weights = self.fusion(vertex_features, bone_embeddings, vol_geo)  # (N, num_bones)
        # Refinement Module refines the weights using vertex adjacency.
        if self.with_refinement:
            refined_skin_weights = self.refinement(predicted_weights, vertex_neighbors, vertex_normals)  # (N, num_bones)
            return refined_skin_weights
        else:
            return predicted_weights