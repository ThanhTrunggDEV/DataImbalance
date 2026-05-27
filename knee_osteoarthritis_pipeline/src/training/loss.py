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
        
        # 1. Tính toán Standard Cross Entropy Loss (KHÔNG dùng weight ở bước này)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 2. Xây dựng xác suất của true class: pt = exp(-CE)
        pt = torch.exp(-ce_loss)
        
        # 3. Tính cấu phần Focal
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # 4. Tính toán trọng số lớp (alpha) thủ công nếu có truyền vào
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss