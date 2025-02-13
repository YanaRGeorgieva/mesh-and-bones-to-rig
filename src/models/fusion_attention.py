import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionModule(nn.Module):
    """
    The Fusion Module computes an attention score between each vertex feature
    and each bone embedding, then selects the top-k highest scores for each vertex
    to form a sparse attention distribution, which is normalized via softmax.
    Incorporates pre-calculated volumetric geodesic distances to bias the attention.

    Parameters:
        vertex_feat_dim: Dimension of vertex features from the Mesh Encoder.
        bone_feat_dim: Dimension of bone embeddings from the Bone Encoder.
        common_dim: Projection dimension for computing attention.
        top_k: Number of bones to keep per vertex (sparsity level, default 4).
        alpha: Scaling factor for volumetric geodesic distances.
               Higher values of alpha cause a stronger penalty for larger distances.
        alpha_learnable: Whether to make the alpha parameter learnable.
    """
    def __init__(self, vertex_feat_dim, bone_feat_dim, common_dim, top_k=4, alpha=1.0, alpha_learnable=True):
        super(FusionModule, self).__init__()
        self.proj_vertex = nn.Linear(vertex_feat_dim, common_dim)
        self.proj_bone = nn.Linear(bone_feat_dim, common_dim)
        self.top_k = top_k
        # Make alpha a learnable parameter.
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.alpha_learnable = alpha_learnable
        if self.alpha_learnable:
            self.alpha.requires_grad = True
        else:
            self.alpha.requires_grad = False


    def forward(self, vertex_features, bone_embeddings, vol_geo):
        """
        Parameters:
            vertex_features: (N, vertex_feat_dim)
            bone_embeddings: (B, bone_feat_dim)
            vol_geo: (N, B) tensor of pre-calculated volumetric geodesic distances.
                     Lower values indicate that the vertex is closer to the bone.

        Returns:
            skin_weights: (N, B) tensor representing the predicted skin weight distribution.
        """
        N = vertex_features.size(0)
        B = bone_embeddings.size(0)
        # Project vertex and bone features into a common space.
        v_proj = self.proj_vertex(vertex_features)  # (N, common_dim)
        b_proj = self.proj_bone(bone_embeddings)    # (B, common_dim)
        # Compute raw dot-product attention scores: (N, B)
        scores = torch.matmul(v_proj, b_proj.t())

        # Compute a bias factor from volumetric geodesic distances.
        # We use an exponential decay so that larger distances yield lower factors.
        # Factor: exp(-alpha * distance)
        geo_factor = torch.exp(-self.alpha * vol_geo)  # (N, B)
        print(self.alpha)

        # Multiply raw scores by the geometric factor.
        biased_scores = scores * geo_factor  # (N, B)

        # For each vertex, select the top_k highest biased scores.
        topk_scores, topk_indices = torch.topk(biased_scores, self.top_k, dim=1)
        # Create a mask for the top_k elements.
        mask = torch.zeros_like(biased_scores)
        mask.scatter_(1, topk_indices, 1)
        # Set scores not in the top_k to a very negative value so they contribute nearly zero.
        sparse_scores = biased_scores * mask + (1 - mask) * (-1e9)

        # Softmax over the bone dimension to get a probability distribution.
        skin_weights = F.softmax(sparse_scores, dim=1)
        return skin_weights