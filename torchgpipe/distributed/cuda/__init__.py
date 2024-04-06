import torch
import torchgpipe.distributed.context as context


def to(device: torch.device, value: context.TensorOrTensors, non_blocking: bool):
    if value is None:
        return None
    if not isinstance(value, tuple):
        new_value = value.detach().to(device, non_blocking=non_blocking)
        if value.requires_grad:
            new_value.requires_grad_()
        return new_value

    new_values = tuple(None if v is None else v.detach().to(
        device, non_blocking=non_blocking) for v in value)

    for old_value, new_value in zip(value, new_values):
        if old_value is None:
            continue
        if old_value.requires_grad:
            new_value.requires_grad_()
    return new_values


class CudaValueCopyProcessor(context.ValueProcessor):

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self._device = device
        self._stream = torch.cuda.Stream(device)

    def DTH(self, value: context.TensorOrTensors):
        with torch.cuda.stream(self._stream):
            value_processed = to(torch.device("cpu"), value, True)
        return value_processed

    def HTD(self, value: context.TensorOrTensors):
        with torch.cuda.stream(self._stream):
            value_processed = to(self._device, value, True)
        return value_processed
