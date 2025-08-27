import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, epsilon=1e-7):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        # Ensure predictions are in [epsilon, 1-epsilon] range
        y_pred = torch.clamp(y_pred, self.epsilon, 1.0 - self.epsilon)
        
        # Focal loss calculation
        ce_loss = -y_true * torch.log(y_pred) - (1.0 - y_true) * torch.log(1.0 - y_pred)
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        focal_weight = self.alpha * y_true + (1.0 - self.alpha) * (1.0 - y_true)
        focal_weight = focal_weight * torch.pow(1.0 - p_t, self.gamma)
        
        focal_loss = focal_weight * ce_loss
        
        return torch.mean(focal_loss)


def focal_loss(alpha=0.25, gamma=2.0, epsilon=1e-7):
    """Factory function for focal loss"""
    return FocalLoss(alpha=alpha, gamma=gamma, epsilon=epsilon)


class BinaryCrossEntropy(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(BinaryCrossEntropy, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        # Clip predictions to avoid log(0)
        clipped_y_pred = torch.clamp(y_pred, self.epsilon, 1.0 - self.epsilon)
        
        # Binary cross entropy calculation
        loss = -y_true * torch.log(clipped_y_pred) - (1.0 - y_true) * torch.log(1.0 - clipped_y_pred)
        
        return torch.mean(loss)


def binary_crossentropy(epsilon=1e-7):
    """Factory function for binary cross entropy loss"""
    return BinaryCrossEntropy(epsilon=epsilon)
