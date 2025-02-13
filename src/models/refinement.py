import torch
import torch.nn as nn
import torch.nn.functional as F

class RefinementModule(nn.Module):
    """
    Refinement Module that refines per-vertex skin weight predictions using local neighborhood information
    weighted by vertex normal similarity.

    For each vertex $i$, let $p_i$ be its predicted skinning weight vector (shape: $(B,)$), and let $N(i)$ be the set of
    $k$-nearest neighbors (provided by vertex_neighbors, shape $(k,)$).
    For each neighbor $j$ in $N(i)$, we compute a similarity score:
            s_{ij} = max(0, \dot(n_i, n_j)),
    where $n_i$ and $n_j$ are the normalized vertex normals. We then compute a diffused version:
            diffused_i = (\sum_{j \in N(i)} s_{ij} * p_j) / (\sum_{j \in N(i)} s_{ij} + eps)
    Finally, we let the refined logits be a weighted combination of the original prediction and the diffused one:
            refined logits_i = p_i + \gamma * (diffused_i - p_i)
    and apply softmax to obtain the refined skinning weight distribution.
    The parameter gamma ($\gamma$) is a learnable scalar.
    """
    def __init__(self, gamma=0.5, eps=1e-8):
        super(RefinementModule, self).__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        self.eps = eps

    def forward(self, predicted_weights, vertex_neighbors, vertex_normals):
        """
        Parameters:
            predicted_weights: Tensor of shape (N, B) with initial per-vertex skin weight predictions.
            vertex_neighbors: LongTensor of shape (N, k) with neighbor indices for each vertex.
            vertex_normals: Tensor of shape (N, 3) with normalized vertex normals.

        Returns:
            refined_weights: Tensor of shape (N, B) after refinement.
        """
        N, B = predicted_weights.shape
        k = vertex_neighbors.size(1)

        # For each vertex, gather neighbor predictions and normals.
        # neighbor_preds: (N, k, B)
        neighbor_preds = predicted_weights[vertex_neighbors]
        # neighbor_normals: (N, k, 3)
        neighbor_normals = vertex_normals[vertex_neighbors]

        # Expand vertex normals for similarity computation: (N, 1, 3)
        normals_exp = vertex_normals.unsqueeze(1)
        # Compute cosine similarities: (N, k)
        # Assume normals are already normalized.
        sim = torch.sum(normals_exp * neighbor_normals, dim=2)
        # Clamp to non-negative values.
        sim = F.relu(sim) # (N, k)

        # Compute weighted sum of neighbor predictions.
        # Multiply neighbor_preds (N, k, B) by sim (N, k, 1)
        sim_expanded = sim.unsqueeze(2) # (N, k) -> (N, k, 1)
        weighted_preds = neighbor_preds * sim_expanded  # (N, k, B)
        # Sum over neighbors.
        weighted_sum = weighted_preds.sum(dim=1)  # (N, k, B) -> (N, B)
        # Sum the similarity weights.
        sim_sum = sim.sum(dim=1).unsqueeze(1)  # (N, k) -> (N, 1)
        diffused = weighted_sum / (sim_sum + self.eps)  # (N, B)

        # Combine original predictions with the diffused values.
        refined_logits = predicted_weights + self.gamma * (diffused - predicted_weights)
        refined_weights = F.softmax(refined_logits, dim=1)
        return refined_weights
