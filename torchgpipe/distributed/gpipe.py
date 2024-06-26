""" Distributed GPipe parallelism based on torch.distributed.rpc 
    TODO: add skip support
"""

from collections import OrderedDict
from queue import Queue
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import Tensor, autograd, nn
from torch.distributed import rpc
from torch.utils import data

from torchgpipe import microbatch
from torchgpipe.batchnorm import DeferredBatchNorm
from torchgpipe.distributed import context
from torchgpipe.distributed.utils import to
from torchgpipe.gpipe import BalanceError, NamedModules, recommend_auto_balance, verify_module

Tensors = Tuple[Tensor, ...]
TensorOrTensors = Union[Tensor, Tensors]

__all__: List[str] = []


def get_module_partition(module: nn.Sequential,
                         rank: int,
                         balance: Iterable[int],
                         device: torch.device,
                         ) -> nn.Sequential:
    """ extract module partition of the given stage according to scheme
    described by 'balance' 

    currently only torch.nn.Sequential is supported

    each element in balace specifies how many layers of the original the stage owns

    Args:
        module (nn.Sequential): full module for distributed training
        rank (int): rank of the stage 
        balance (Iterable[int]): specifies the module partition scheme 
        device (torch.device): device the result parition will be transfered to 


    Returns:
        nn.Sequential: the result module partition 
    """
    balance = list(balance)

    if len(module) != sum(balance):
        raise BalanceError('module and sum of balance have different length '
                           f'(module: {len(module)}, sum of balance: {sum(balance)})')
    if any(x <= 0 for x in balance):
        raise BalanceError(f'all balance numbers must be positive integer (balance: {balance})')

    j = 0
    layers: NamedModules = OrderedDict()

    for name, layer in module.named_children():
        layers[name] = layer
        if len(layers) == balance[j]:
            # Group buffered layers as a partition.
            if j == rank:
                partition = nn.Sequential(layers)
                if device is not None:
                    partition.to(device)
                return partition
            # Prepare for the next partition.
            layers.clear()
            j += 1

    raise RuntimeError('module and balance mismatch')


class DistributedGPipe:
    """Module wrapper for GPipe distributed training. Note this module is not 
    subclass of torch.nn.Module.

    Each stage in the distributed training setup has an gloabally unique name.

    the 'workers' argument passed to the initializer is a dict from stage rank
    to the worker name. 

    """

    @staticmethod
    def _get(name: str, id: int, backward=False):
        func = context.get_backward if backward else context.get_forward
        return func(name, id)

    @staticmethod
    def _put(name: str, id: int, values: TensorOrTensors, backward=False):
        func = context.put_backward if backward else context.put_forward
        rpc.remote(
            name, func, args=(name, id, values)
        )

    #: The number of micro-batches.
    chunks: int = 1

    def __init__(self,
                 module: nn.Sequential,
                 rank: int,
                 workers: Dict[int, str],
                 balance: Optional[Iterable[int]] = None,
                 microbatch_chunks: int = chunks,
                 *,
                 device: Optional[torch.device] = None,
                 deferred_batch_norm: bool = False,
                 ) -> None:

        microbatch_chunks = int(microbatch_chunks)

        if balance is None:
            raise ValueError(recommend_auto_balance('balance is required'))
        if microbatch_chunks <= 0:
            raise ValueError('number of chunks must be positive integer')

        verify_module(module)
        module = get_module_partition(module, rank, balance, device)

        self.module = module
        self.rank = rank
        self.world_size = len(workers)
        self.workers = workers
        self.chunks = microbatch_chunks
        self.device = device
        self.name = workers[rank]

        self._outputs: List[Optional[TensorOrTensors]] = [None for _ in range(microbatch_chunks)] 

        self._grad_output = Queue()  # for retriving gradients from self.module

        self._remove_handle = module.register_full_backward_hook(self._retrieve_grad)

        if deferred_batch_norm:
            module = DeferredBatchNorm.convert_deferred_batch_norm(module, microbatch_chunks)

    def _retrieve_grad(self, module, grad_input, grad_output):
        """This function is used as backward hook for retriving the
        gradients generated by the module backward pass
        """
        self._grad_output.put(grad_input)

    def _previous_worker(self) -> Optional[str]:
        if self.rank == 0:
            return None
        return self.workers[self.rank - 1]

    def _next_worker(self) -> Optional[str]:
        if self.rank == self.world_size - 1:
            return None
        return self.workers[self.rank + 1]

    def model(self):
        return self.module

    # type: ignore
    def forward(self, batch: Optional[TensorOrTensors]) -> TensorOrTensors:
        if batch is not None:
            microbatch.check(batch)
            training_datas = microbatch.scatter(batch, self.chunks)
            assert self.rank == 0

        for mbatch in range(self.chunks):
            if self.rank == 0:
                inputs = training_datas[mbatch].value
            else:
                inputs = DistributedGPipe._get(self.name, mbatch)
            inputs = to(self.device, inputs)
            outputs = self.module(inputs)
            self._outputs[mbatch] = outputs

            next_worker = self._next_worker()
            if next_worker is not None:
                outputs_cpu = to(torch.device("cpu"), outputs)
                DistributedGPipe._put(next_worker, mbatch, outputs_cpu)
        return self._outputs 

    def backward(self, losses: Optional[List[Tensor]]):
        for mbatch in range(self.chunks):
            if losses is not None:
                assert self.rank == (self.world_size - 1)
                losses[mbatch].backward()
            else:
                values = DistributedGPipe._get(self.name, mbatch, True)
                values = to(self.device, values)
                autograd.backward(self._outputs[mbatch], values)

            prev_worker = self._previous_worker()
            if prev_worker is not None:
                leaves = self._grad_output.get()
                leaves = to(torch.device("cpu"), leaves)
                DistributedGPipe._put(prev_worker, mbatch, leaves, True)


class DistributedGPipeDataLoader:
    """The DistributedGPipeDataLoader is intended for use together with 'DistributedGPipe'

    For Stage0, the distributed dataloader loads the data and target using underlying
    dataloader provided during the initialization, then it sends the target to the last
    stage, return (data, None) to the user.

    For middle stages, this data loader always return (None, None)  

    For the last stage, this dataloader receive target sent by Stage0, and return (None, target)
    to user.

    """

    @staticmethod
    def _put(name: str, value: TensorOrTensors):
        rpc.remote(
            name, context.put_target, (name, value)
        )

    @staticmethod
    def _get(name: str):
        return context.get_target(name)

    def __init__(self,
                 data_loader: Optional[data.DataLoader],
                 rank: int,
                 chunks: int,
                 num_iterations: int,
                 last_stage: bool,
                 last_stage_name: str,
                 ):
        self._data_loader = data_loader
        self._rank = rank
        self._chunks = chunks
        self._num_iterations = num_iterations
        self._iter_cnt = 0
        self._last_stage = last_stage
        self._last_stage_name = last_stage_name

    def _last_stage_iter(self):
        """iterator for last stage

        Yields:
            (None, target)
        """
        for _ in range(self._num_iterations):
            mtarget = DistributedGPipeDataLoader._get(self._last_stage_name)
            yield (None, mtarget)

    def _first_stage_iter(self):
        """iterator for stage 0, note this function will start sending
        target to the last stage on the fly.

        Yields:
            (data, None)
        """
        for (data, target), _ in zip(self._data_loader, range(self._num_iterations)):
            DistributedGPipeDataLoader._put(self._last_stage_name, target)
            yield (data, None)

    def _middle_stage_iter(self):
        """iterator for middle stages

        Yields:
            (None, None)
        """
        for _ in range(self._num_iterations):
            yield (None, None)

    def __iter__(self):
        if self._last_stage:
            return self._last_stage_iter()
        if self._rank == 0:
            return self._first_stage_iter()
        return self._middle_stage_iter()

    def __len__(self):
        return self._num_iterations
