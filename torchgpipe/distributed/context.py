"""_summary_
"""

from contextlib import contextmanager
from queue import Queue
from typing import Callable, Dict, Tuple, Union

from torch import Tensor

Tensors = Tuple[Tensor, ...]
TensorOrTensors = Union[Tensor, Tensors]
Channel = Queue[TensorOrTensors]


class TrainingContext:

    def __init__(self, name: str) -> None:
        self.forward_channels: Dict[int, Channel] = {}
        self.backward_channels: Dict[int, Channel] = {}
        self.target_channels: Dict[int, Channel] = {}
        self.name = name


class GlobalContext:

    ctxs: Dict[str, TrainingContext] = {}

    @staticmethod
    def get_context(name: str):
        # TODO: error handling
        return GlobalContext.ctxs[name]


@contextmanager
def worker(name: str):
    if name in GlobalContext.ctxs:
        raise RuntimeError(f"worker {name} already exists")
    ctx = TrainingContext(name)
    GlobalContext.ctxs[name] = ctx
    yield
    del GlobalContext.ctxs[name]


def distributed(name: str):
    def decorator(func: Callable):
        def forwarder(*args, **kwargs):
            with worker(name):
                func(*args, **kwargs)
        return forwarder
    return decorator


def put_forward(name: str, id: int, value: TensorOrTensors):
    ctx = GlobalContext.get_context(name)
    return ctx.forward_channels.setdefault(id, Queue()).put(value)


def put_backward(name: str, id: int, value: TensorOrTensors):
    ctx = GlobalContext.get_context(name)
    return ctx.forward_channels.setdefault(id, Queue()).put(value)


def put_target(name: str, id: int, value: TensorOrTensors) -> TensorOrTensors:
    ctx = GlobalContext.get_context(name)
    return ctx.target_channels.setdefault(id, Queue()).put(value)


def get_forward(name: str, id: int) -> TensorOrTensors:
    ctx = GlobalContext.get_context(name)
    return ctx.forward_channels.setdefault(id, Queue()).get()


def get_backward(name: str, id: int) -> TensorOrTensors:
    ctx = GlobalContext.get_context(name)
    return ctx.forward_channels.setdefault(id, Queue()).get()


def get_target(name: str, id: int) -> TensorOrTensors:
    ctx = GlobalContext.get_context(name)
    return ctx.forward_channels.setdefault(id, Queue()).get()
