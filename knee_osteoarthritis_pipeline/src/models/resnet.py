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

    # Freeze backbone layers, only train the new head (fine-tuning strategy)
    # Comment these two lines out if you want full fine-tuning from the start
    # for param in model.parameters():
    #     param.requires_grad = False

    in_features = model.fc.in_features          # 2048 for ResNet50
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )
    return model
