import torch
import torch.nn.functional as F

def geodesic_loss(predicted_weights, geodesic_distances, alpha):
    """
    Computes a KL divergence loss between the predicted skinning weights and a target distribution
    derived from the geodesic distances.
    
    Args:
        predicted_weights (Tensor): shape (N, B), the network's output per vertex.
        geodesic_distances (Tensor): shape (N, B), geodesic distances from each vertex to each bone.
        alpha (float): scaling factor for the exponential decay.
    
    Returns:
        A scalar loss value.
    """
    # Compute the target distribution using an exponential decay.
    # Here, lower distance means higher weight.
    target_logits = -alpha * geodesic_distances  # (N, B)
    p_target = F.softmax(target_logits, dim=1)
    
    # Use KL divergence (or another divergence) as loss.
    loss = F.kl_div(torch.log(predicted_weights + 1e-8), p_target, reduction='batchmean')
    return loss