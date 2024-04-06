""" torhgpipe.distributed.context is used internally for message&data passing among workers
in distributed training setup.
"""
from abc import abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Queue
from typing import Dict, Optional, Tuple, Union
from torch.distributed import rpc

from torch import Tensor


Tensors = Tuple[Tensor, ...]
TensorOrTensors = Union[Tensor, Tensors]
Channel = Queue[TensorOrTensors]


class ValueProcessor:

    @abstractmethod
    def DTH(self, value: TensorOrTensors) -> TensorOrTensors:
        raise NotImplementedError()

    @abstractmethod
    def HTD(self, value: TensorOrTensors) -> TensorOrTensors:
        raise NotImplementedError()


def _on_send(
    context_name: str,
    value: TensorOrTensors,
    target: bool = False,
    backward: Optional[bool] = None,
    microbatch_id: Optional[int] = None,
):
    ctx = DistributedContextRegistry.context(context_name)
    channel = ctx.channel(target, backward, microbatch_id)
    processed_value = ctx.processor.HTD(value)
    print("on send", id(channel))
    channel.put(processed_value)


class DistributedContext:

    def __init__(
        self,
        microbatches: int,
        context_name: str,
        value_processor: ValueProcessor
    ):
        self.processor = value_processor
        self.name = context_name

        self.forward_channels = [Channel() for _ in range(microbatches)]
        self.backward_channels = [Channel() for _ in range(microbatches)]
        self.target_channel = Channel()

        self._executor = ThreadPoolExecutor(8)

    def channel(
        self,
        target: bool,
        backward: Optional[bool],
        microbatch_id: Optional[int]
    ) -> Channel:
        if target:
            channel = self.target_channel
        else:
            channels = self.backward_channels \
                if backward else self.forward_channels
            channel = channels[microbatch_id]
        return channel

    def put_remote(
        self,
        remote_name: str,
        value: TensorOrTensors,
        target: bool = False,
        *,
        backward: Optional[bool] = None,
        microbatch_id: Optional[int] = None,
    ) -> Future[None]:
        def send_task():
            processed_value = self.processor.DTH(value)
            rpc.remote(
                remote_name, _on_send, args=(
                    remote_name, processed_value, target, backward, microbatch_id
                )
            )
        return self._executor.submit(send_task)

    def get_remote(
        self,
        target: bool,
        *,
        backward: Optional[bool] = None,
        microbatch_id: Optional[int] = None,
    ) -> TensorOrTensors:
        channel = self.channel(target, backward, microbatch_id)
        print(id(channel), "get")
        return channel.get()

    def shutdown(self):
        self._executor.shutdown()


class DistributedContextRegistry:

    _ctxs: Dict[str, DistributedContext] = {}

    @staticmethod
    def context(context_name: str):
        # TODO: error handling
        return DistributedContextRegistry._ctxs[context_name]

    @staticmethod
    def registrate(context: DistributedContext):
        context_name = context.name
        if context_name in DistributedContextRegistry._ctxs:
            raise ValueError(f"context {context_name} alread exists.")
        DistributedContextRegistry._ctxs[context_name] = context

    @staticmethod
    def deregistrate(context_name: str):
        ctx = DistributedContextRegistry.context(context_name)
        ctx.shutdown()
        del DistributedContextRegistry._ctxs[context_name]
