import torch

from torchgpipe.gpipe import TensorOrTensors


def to(device: torch.device, value: TensorOrTensors):
    if not isinstance(value, tuple):
        return None if value is None else value.to(device)

    return tuple(None if v is None else v.to(device) for v in value)
