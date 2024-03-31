import torch

from torchgpipe.gpipe import TensorOrTensors


def to(device: torch.device, value: TensorOrTensors):
    if value is None:
        return None
    if not isinstance(value, tuple):
        new_value = value.detach().to(device)
        if value.requires_grad:
            new_value.requires_grad_()
        return new_value

    new_values = tuple(None if v is None else v.detach().to(device) for v in value)

    for old_value, new_value in zip(value, new_values):
        if old_value is None:
            continue
        if old_value.requires_grad:
            new_value.requires_grad_()
    return new_values
