import torch
import torch.nn.functional as F

def smoothness_loss(predicted_weights, vertex_adj):
    """
    Computes a Laplacian smoothness loss on the predicted skin weights.
    Each vertex's weight vector is encouraged to be close to the average of its neighbors.
    Parameters:
        predicted_weights: (N, B) tensor of predicted skin weights.
        vertex_adj: (N, N) normalized adjacency matrix (rows sum to 1).
    
    Returns:
        Scalar loss value.
    """
    # Compute the average skin weights of each vertex's neighbors.
    avg_weights = torch.matmul(vertex_adj, predicted_weights)  # (N, B)
    loss = F.mse_loss(predicted_weights, avg_weights)
    return loss