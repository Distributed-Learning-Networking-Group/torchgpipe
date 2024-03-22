import torch
import torchvision
import torch.nn as nn


def ReLU_inplace_to_False(module: nn.Module):
    for layer in module.children():
        if isinstance(layer, nn.ReLU):
            layer.inplace = False
        ReLU_inplace_to_False(layer)


def resnet101(num_classes: int, inplace: bool):
    model = torchvision.models.resnet101(num_classes=num_classes)
    model = torch.nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
        model.avgpool,
        torch.nn.Flatten(),
        model.fc,
    )
    if inplace is False:
        ReLU_inplace_to_False(model)
    return model
