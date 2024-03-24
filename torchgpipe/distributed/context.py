""" torhgpipe.distributed.context is used for message&data passing among workers
in distributed training setup.

Internally, this module maintains a list of 'Queue[torch.TensorOrTensors]', which we call 'Channel'
for passing activation/gradients during forward/backward pass.

"""
from contextlib import contextmanager
from queue import Queue
from typing import Callable, Dict, Tuple, Union

from torch import Tensor

Tensors = Tuple[Tensor, ...]
TensorOrTensors = Union[Tensor, Tensors]
Channel = Queue[TensorOrTensors]


class TrainingContext:

    def __init__(self, context_name: str, microbatch_chunks: int) -> None:
        self.forward_channels = [Channel() for _ in range(microbatch_chunks)]
        self.backward_channels = [Channel() for _ in range(microbatch_chunks)]
        self.target_channels: Dict[int, Channel] = [Channel() for _ in range(microbatch_chunks)]
        self.name = context_name


class GlobalContext:
    """Instead of Global variables, we use an Class variable to store Training Contexts,
    indexed by their context_names
    """

    ctxs: Dict[str, TrainingContext] = {}

    @staticmethod
    def get_context(context_name: str):
        # TODO: error handling
        return GlobalContext.ctxs[context_name]


@contextmanager
def worker(context_name: str, microbatch_chunks: int):
    """context manager for training context 

    Args:
        context_name (str): context name for the worker, must be globally unique 
        microbatch_chunks (int): number of microbatch chunks in the gpipe training setup.


    Examples:

    context_name = "worker0"

    with worker(context_name):
        train()
        ...

    """
    if context_name in GlobalContext.ctxs:
        raise RuntimeError(f"worker {context_name} already exists")
    ctx = TrainingContext(context_name, microbatch_chunks)
    GlobalContext.ctxs[context_name] = ctx
    yield
    del GlobalContext.ctxs[context_name]


def distributed(context_name: str, microbatch_chunks):
    """Decorator around torchgpipe.context.worker

    Args:
        context_name (str): context name for the worker, must be globally unique 
        microbatch_chunks (int): number of microbatch chunks in the gpipe training setup.

    Examples:
    context_name = "worker0"

    @distributed(context_name)
    def train(...):
        ...

    train(...)

    Equivalent to:

    with worker(context_name):
        train()
    """
    def decorator(func: Callable):
        def forwarder(*args, **kwargs):
            with worker(context_name, microbatch_chunks):
                func(*args, **kwargs)
        return forwarder
    return decorator


def put_forward(context_name: str, id: int, value: TensorOrTensors):
    """Put given tensor or tensors to the given channel in context 'context_name'
    with id 'id'

    Args:
        name (str): context name for the worker, must be globally unique 
        id (int): channel id the 'value' will be put in
        value (TensorOrTensors): tensor or tensors to be put 

    Examples:

    with worker("worker0"):
        ...
        // microbatch id 3,  
        value = compute() 
        rpc.Call(put_forward, args=("worker1", 3, value))
        ...

    """
    ctx = GlobalContext.get_context(context_name)
    ctx.forward_channels[id].put(value)


def get_forward(name: str, id: int) -> TensorOrTensors:
    """Get given tensor or tensors to the given channel in context 'context_name'
    with id 'id'

    Args:
        name (str): context name for the worker, must be globally unique 
        id (int): channel id the 'value' will be put in

    Returns:
        TensorOrTensors: value that was put in the given channel. 

    Examples:

    with worker("worker0"):
        ...
        // microbatch id 3,  
        value = compute() 
        rpc.Call(get_forward, args=("worker1", 3, value))
        ...

    """
    ctx = GlobalContext.get_context(name)
    return ctx.forward_channels[id].get()


def put_backward(name: str, id: int, value: TensorOrTensors):
    """see also 'put_forward'
    Args:
        name (str): context name for the worker, must be globally unique 
        id (int): channel id the 'value' will be put in
        value (TensorOrTensors): tensor or tensors to be put 
    """
    ctx = GlobalContext.get_context(name)
    return ctx.backward_channels[id].put(value)


def get_backward(name: str, id: int) -> TensorOrTensors:
    """see also 'get_forward'

    Args:
        name (str): context name for the worker, must be globally unique 
        id (int): channel id the 'value' will be put in


    Returns:
        TensorOrTensors: value that was put in the given channel. 
    """
    ctx = GlobalContext.get_context(name)
    return ctx.backward_channels[id].get()


def put_target(name: str, id: int, value: TensorOrTensors) -> TensorOrTensors:
    """see also 'put_forward'
    Args:
        name (str): context name for the worker, must be globally unique 
        id (int): channel id the 'value' will be put in
        value (TensorOrTensors): tensor or tensors to be put 
    """
    ctx = GlobalContext.get_context(name)
    return ctx.target_channels[id].put(value)


def get_target(name: str, id: int) -> TensorOrTensors:
    """see also 'get_forward'

    Args:
        name (str): context name for the worker, must be globally unique 
        id (int): channel id the 'value' will be put in


    Returns:
        TensorOrTensors: value that was put in the given channel. 
    """
    ctx = GlobalContext.get_context(name)
    return ctx.target_channels[id].get()
