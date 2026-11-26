"""ResNet-18 for 32x32 inputs, based on torchvision."""

from __future__ import annotations

from torch import nn
from torchvision import models


def resnet18_small(num_classes: int) -> nn.Module:
    """torchvision ResNet-18 with a 3x3 stem and no max pooling."""
    model = models.resnet18()
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, num_classes)
    return model
