import torch.nn as nn
import torchvision.models as models

def get_resnet18(num_classes=4):
    model = models.resnet18(weights="DEFAULT")

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model