""" Distributed GPipe parallelism based on torch.distributed.rpc 
    TODO: add skip support
"""

from collections import OrderedDict
from concurrent.futures import Future
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import Tensor, autograd, nn
from torch.utils import data

from torchgpipe import microbatch
from torchgpipe.distributed import context
from torchgpipe.batchnorm import DeferredBatchNorm
from torchgpipe.distributed.batch import DistributedBatch
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

        self._is_last_stage = self.rank == self.world_size - 1

        self._inputs: List[Optional[DistributedBatch]] = [None for _ in range(microbatch_chunks)]
        self._outputs: List[Optional[DistributedBatch]] = [None for _ in range(microbatch_chunks)]

        self._forward_futures: List[Future[None]] = [None for _ in range(microbatch_chunks)]
        self._backward_futures: List[Future[None]] = [None for _ in range(microbatch_chunks)]

        self._context = context.DistributedContextRegistry.context(self.name)

        if deferred_batch_norm:
            module = DeferredBatchNorm.convert_deferred_batch_norm(module, microbatch_chunks)

    def _previous_worker(self) -> Optional[str]:
        if self.rank == 0:
            return None
        return self.workers[self.rank - 1]

    def _next_worker(self) -> Optional[str]:
        if self.rank == self.world_size - 1:
            return None
        return self.workers[self.rank + 1]

    def is_last_stage(self):
        return self.is_last_stage

    def model(self):
        return self.module

    def forward(self, batch: Optional[TensorOrTensors]) -> List[DistributedBatch]:
        if self.rank == 0:
            microbatch.check(batch)
            training_datas_host = microbatch.scatter(batch, self.chunks)
            training_datas = [self._context.processor.HTD(batch.value)
                              for batch in training_datas_host]

        for mbatch in range(self.chunks):

            if self.rank == 0:
                inputs = training_datas[mbatch]
            else:
                inputs = self._context.get_remote(False, backward=False, microbatch_id=mbatch)

            self._inputs[mbatch] = DistributedBatch(inputs)
            outputs = self.module(inputs)
            self._outputs[mbatch] = DistributedBatch(outputs)

            next_worker = self._next_worker()
            if next_worker is not None:
                self._forward_futures[mbatch] = self._context.put_remote(
                    next_worker, outputs, False, backward=False, microbatch_id=mbatch
                )
        if not self.is_last_stage():
            for fut in self._forward_futures:
                # wait for futures explicitly here to propagate errors
                # to training thread
                fut.result()

        return self._outputs

    def backward(self, losses: Optional[List[Tensor]]):
        for mbatch in range(self.chunks):

            if self.rank == (self.world_size - 1):
                losses[mbatch].backward()
            else:
                values = self._context.get_remote(False, backward=True, microbatch_id=mbatch)
                outputs = self._outputs[mbatch].requires_grad_trait()
                autograd.backward(outputs, values)

            prev_worker = self._previous_worker()
            if prev_worker is not None:
                self._backward_futures[mbatch] = self._context.put_remote(
                    prev_worker,
                    self._inputs[mbatch].grad_trait(),
                    False,
                    backward=True,
                    microbatch_id=mbatch
                )

        if self.rank != 0:
            for fut in self._backward_futures:
                # wait for futures explicitly here to propagate errors
                # to training thread
                fut.result()


class DistributedGPipeDataLoader:
    """The DistributedGPipeDataLoader is intended for use together with 'DistributedGPipe'

    For Stage0, the distributed dataloader loads the data and target using underlying
    dataloader provided during the initialization, then it sends the target to the last
    stage, return (data, None) to the user.

    For middle stages, this data loader always return (None, None)  

    For the last stage, this dataloader receive target sent by Stage0, and return (None, target)
    to user.

    """

    def __init__(
        self,
        name: str,
        data_loader: Optional[data.DataLoader],
        chunks: int,
        num_iterations: int,
        first_stage: bool,
        last_stage: bool,
        last_stage_name: str,
    ):
        self._data_loader = data_loader
        self._chunks = chunks
        self._num_iterations = num_iterations
        self._first_stage = first_stage
        self._last_stage = last_stage
        self._last_stage_name = last_stage_name
        self._context = context.DistributedContextRegistry.context(name)

    def _last_stage_iter(self):
        """iterator for last stage

        Yields:
            (None, target)
        """
        for _ in range(self._num_iterations):
            mtarget = self._context.get_remote(True)
            yield (None, mtarget)

    def _first_stage_iter(self):
        """iterator for stage 0, note this function will start sending
        target to the last stage on the fly.

        Yields:
            (data, None)
        """
        for (data, target), _ in zip(self._data_loader, range(self._num_iterations)):
            self._context.put_remote(
                self._last_stage_name, target, True
            )
            yield (data, None)

    def _middle_stage_iter(self):
        """iterator for middle stages

        Yields:
            (None, None)
        """
        for _ in range(self._num_iterations):
            yield (None, None)

    def _single_worker_iter(self):
        """ iterator for only one worker
        """
        for (data, target), _ in zip(self._data_loader, range(self._num_iterations)):
            yield self._context.processor.HTD((data, target))

    def __iter__(self):
        if self._last_stage and self._first_stage:
            return self._single_worker_iter()
        if self._last_stage:
            return self._last_stage_iter()
        if self._first_stage:
            return self._first_stage_iter()
        return self._middle_stage_iter()

    def __len__(self):
        return self._num_iterations
