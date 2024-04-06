from typing import Iterable

import torch
import torch.nn as nn
import torchvision


def relu_inplace_to_false(module: nn.Module):
    for layer in module.children():
        if isinstance(layer, nn.ReLU):
            layer.inplace = False
        relu_inplace_to_false(layer)


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
        relu_inplace_to_false(torch_model)
    return _resnet_to_sequential(torch_model)


def resnet50(num_classes: int, inplace: bool):
    torch_model = torchvision.models.resnet50(num_classes=num_classes)
    if inplace is False:
        relu_inplace_to_false(torch_model)
    return _resnet_to_sequential(torch_model)
