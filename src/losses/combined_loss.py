from .geodesic_loss import geodesic_loss
from .smoothness_loss import smoothness_loss
from .skinning_weight_loss import skinning_weight_loss

def combined_loss(predicted_weights,
                    target_weights,
                    geodesic_distances,
                    vertex_adj,
                    alpha=1.0,
                    lambda_skin=1.0,
                    lambda_geo=1.0,
                    lambda_smooth=1.0):
    """
    Combines the skinning weight loss, geodesic loss, and smoothness loss.
    
    Parameters:
        predicted_weights: (N, B) tensor of predicted skin weights.
        target_weights: (N, B) tensor of ground-truth skin weights.
        geodesic_distances: (N, B) tensor of geodesic distances between vertices and bones.
        vertex_adj: (N, N) normalized adjacency matrix.
        alpha: Scalar for the exponential decay in the geodesic loss.
        lambda_skin, lambda_geo, lambda_smooth: Scalars weighting each loss term.
    Returns:
        loss_skin, loss_geo, loss_smooth, total_loss: Tuple of the individual loss terms and the total loss.
    """
    loss_skin = skinning_weight_loss(predicted_weights, target_weights)
    loss_geo = geodesic_loss(predicted_weights, geodesic_distances, alpha)
    loss_smooth = smoothness_loss(predicted_weights, vertex_adj)
    total_loss = lambda_skin * loss_skin + lambda_geo * loss_geo + lambda_smooth * loss_smooth
    print(f"Loss Skin: {loss_skin.item():.6f}, Loss Geo: {loss_geo.item():.6f}, Loss Smooth: {loss_smooth.item():.6f}, Total loss: {total_loss.item():.6f}")
    return loss_skin, loss_geo, loss_smooth, total_loss