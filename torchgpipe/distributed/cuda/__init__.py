import torch
import torchgpipe.distributed.context as context


def to(
    device: torch.device,
    value: context.TensorOrTensors,
    non_blocking: bool = True
):
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

    def __init__(self, device: torch.device, microbatches: int) -> None:
        super().__init__()
        self._device = device
        self._streams = [torch.cuda.Stream(device)
                         for _ in range(microbatches)]

    def device_to_host(self, value: context.TensorOrTensors, mbatch: int):
        stream_ = None if mbatch is None else self._streams[mbatch]
        with torch.cuda.stream(stream_):
            torch.cuda.current_stream().wait_stream(torch.cuda.default_stream())
            value_processed = to(torch.device("cpu"), value)
        return value_processed

    def host_to_device(self, value: context.TensorOrTensors, mbatch: int):
        stream_ = None if mbatch is None else self._streams[mbatch]
        with torch.cuda.stream(stream_):
            value_processed = to(self._device, value, False)
        return value_processed

    def sync(self, mbatch: int):
        if mbatch is not None:
            self._streams[mbatch].synchronize()
