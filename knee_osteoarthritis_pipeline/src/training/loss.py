import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    MODERN IMBALANCE HANDLING: Focal Loss
    Focal Loss dynamically scales cross entropy based on prediction confidence.
    It down-weights easy examples and focuses training on hard negatives/minority classes.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha # Alpha acts as the class weights

    def forward(self, inputs, targets):
        # inputs: [Batch_size, Num_classes], targets: [Batch_size]
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        
        # Calculate pt (probabilities of targets)
        pt = torch.exp(-ce_loss)
        
        # Focal loss formula: (1 - pt)^gamma * CE_Loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss