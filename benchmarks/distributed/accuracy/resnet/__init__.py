from typing import Iterable
import torch
import torchvision
import torch.nn as nn


def ReLU_inplace_to_False(module: nn.Module):
    for layer in module.children():
        if isinstance(layer, nn.ReLU):
            layer.inplace = False
        ReLU_inplace_to_False(layer)


def _resnet_to_sequential(torch_model: torchvision.models.ResNet) -> nn.Sequential:
    model = torch.nn.Sequential(
        torch_model.conv1,
        torch_model.bn1,
        torch_model.relu,
        torch_model.maxpool,
    )
    append_sequential(model, torch_model.layer1)
    append_sequential(model, torch_model.layer2)
    append_sequential(model, torch_model.layer3)
    append_sequential(model, torch_model.layer4)
    append_sequential(
        model, [
            torch_model.avgpool,
            torch.nn.Flatten(),
            torch_model.fc,
        ]
    )
    return model


def append_sequential(module: nn.Sequential, children: Iterable[nn.Module]):
    for child in children:
        module.append(child)


def resnet101(num_classes: int, inplace: bool):
    torch_model = torchvision.models.resnet101(num_classes=num_classes)
    if inplace is False:
        ReLU_inplace_to_False(torch_model)
    return _resnet_to_sequential(torch_model)


def resnet50(num_classes: int, inplace: bool):
    torch_model = torchvision.models.resnet50(num_classes=num_classes)
    if inplace is False:
        ReLU_inplace_to_False(torch_model)
    return _resnet_to_sequential(torch_model)
