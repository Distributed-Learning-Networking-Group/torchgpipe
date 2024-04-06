import contextlib
from typing import Callable, List
import torch

from torchgpipe import microbatch
from torchgpipe.distributed.batch import DistributedBatch
from torchgpipe.gpipe import TensorOrTensors

from . import context
from . import cuda


class CpuDetachProcessor(context.ValueProcessor):

    @staticmethod
    def _detach(value: TensorOrTensors):
        if isinstance(value, torch.Tensor):
            value_ = value.detach()
            if value.requires_grad:
                value_.requires_grad_()
            return value_
        if isinstance(value, tuple):
            value_ = tuple(None if t is None else t.detach() for t in value)
            for i, v in enumerate(value):
                if v is not None and v.requires_grad:
                    value_[i].requires_grad_()
            return value_
        raise ValueError(f"invalid value:{value}, expected tensor or tuple of tensor")

    def DTH(self, value: TensorOrTensors):
        return self._detach(value)

    def HTD(self, value: TensorOrTensors):
        return self._detach(value)


@contextlib.contextmanager
def run(
    context_name: str,
    microbatches: int,
    device: torch.device
):
    if device.type == "cuda":
        value_processor = cuda.CudaValueCopyProcessor(device)
    elif device.type == "cpu":
        value_processor = CpuDetachProcessor()
    else:
        raise ValueError("invalid device type")

    ctx = context.DistributedContext(
        context_name=context_name,
        microbatches=microbatches,
        value_processor=value_processor
    )

    context.DistributedContextRegistry.registrate(ctx)

    yield
    # training...

    context.DistributedContextRegistry.deregistrate(ctx.name)


def loss(
    outputs: List[DistributedBatch],
    targets: TensorOrTensors,
    loss_function: Callable,
    mean: bool = True
) -> List[torch.Tensor]:
    chunks = len(outputs)
    targets_ = microbatch.scatter(targets, chunks)
    scale = float(chunks) if mean else 1.0
    return [
        loss_function(output.value, target.value) / scale
        for output, target in zip(outputs, targets_)
    ]
