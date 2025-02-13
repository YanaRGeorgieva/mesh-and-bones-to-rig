import torch
import torch.nn.functional as F

def skinning_weight_loss(predicted_weights, target_weights, eps=1e-6):
    """
    Computes the KL divergence loss between predicted and ground-truth skinning weight distributions.
    
    Parameters:
        predicted_weights (torch.Tensor): Predicted skinning weights of shape. 
                          These should be non-negative and normalized (e.g. via Sparse-Softmax).
        target_weights (torch.Tensor): Ground-truth skinning weights, same shape as predicted_weights.
        eps (float): A small constant for numerical stability.
    
    Returns:
        Scalar KL divergence loss value. (torch.Tensor)
    """
    return F.kl_div(torch.log(predicted_weights + eps), target_weights, reduction='batchmean')