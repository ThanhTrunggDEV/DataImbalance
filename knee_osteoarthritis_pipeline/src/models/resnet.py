import torch.nn as nn
from torchvision import models


def get_resnet50_model(num_classes: int = 5, pretrained: bool = True) -> nn.Module:
    """
    Load a pretrained ResNet50 and replace the classifier head.

    Architecture changes:
    - Original FC (2048 → 1000) is replaced by:
        Dropout(0.5) → Linear(2048, num_classes)
    This gives a clean baseline that is also used by all experiment versions.
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)

    in_features = model.fc.in_features          # 2048 for ResNet50
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )
    return model


def get_resnet50_supcon_model(
    num_classes: int = 5,
    project_dim: int = 128,
    pretrained: bool = True,
) -> nn.Module:
    """
    ResNet50 backbone + projection head for SupCon pretraining.

    Architecture:
    - Backbone: ResNet50 (without original FC)
    - Projection head: MLP(2048 → 2048 → project_dim) with ReLU

    Args:
        num_classes:  Not used in projection head (kept for API compatibility).
        project_dim:  Output dimension of the projection head.
        pretrained:   Load ImageNet weights.

    Returns:
        model with .forward(x) → [B, project_dim] features.
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)

    backbone_dim = model.fc.in_features  # 2048
    model.fc = nn.Identity()

    model.projector = nn.Sequential(
        nn.Linear(backbone_dim, backbone_dim),
        nn.ReLU(inplace=True),
        nn.Linear(backbone_dim, project_dim),
    )

    return model


def freeze_backbone(model: nn.Module):
    """Freeze all parameters except the classifier head (fc)."""
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = False


def unfreeze_all(model: nn.Module):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True
