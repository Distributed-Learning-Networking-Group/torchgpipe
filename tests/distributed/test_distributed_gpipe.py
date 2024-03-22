from queue import Queue
from typing import Any, Dict
from unittest.mock import patch

import pytest
import torch
from torch import nn
import torch.utils
import torchvision
from torch.utils.data import dataloader

from torchgpipe.checkpoint import TensorOrTensors
from torchgpipe.distributed.context import TrainingContext
from torchgpipe.distributed.gpipe import (DistributedGPipe, DistributedGPipeDataLoader,
                                          get_module_partition)

Channels = Dict[int, Queue]


def detach(value: TensorOrTensors):
    if isinstance(value, tuple):
        ret = tuple(None if v is None else v.detach()
                    for v in value)
        for i, v in enumerate(value):
            if v is not None and v.requires_grad:
                ret[i].requires_grad_()
        return ret
    ret = None if value is None else value.detach()
    if ret is not None and value.requires_grad:
        ret.requires_grad_()
    return ret


class FakeTrainingGloablContext:

    def __init__(self) -> None:
        self.ctxs: Dict[str, TrainingContext] = {}

    def fake_get(self, name: str, id: int, backward=False):
        ctx = self.ctxs.setdefault(name, TrainingContext(name))
        if backward:
            channels = ctx.backward_channels
        else:
            channels = ctx.forward_channels
        return channels[id].get()

    def fake_put(self, name: str, id: int, value: Any, backward=False):
        ctx = self.ctxs.setdefault(name, TrainingContext(name))
        if backward:
            channels = ctx.backward_channels
        else:
            channels = ctx.forward_channels
        value = detach(value)
        return channels[id].put(value)


@pytest.fixture
def module():
    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(784, 256),
                        nn.ReLU(),
                        nn.Linear(256, 10))

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weights)
    return net


def _workers(num: int):
    workers = {}
    for i in range(num):
        workers[i] = f"worker{i}"
    return workers


@pytest.fixture
def workers():
    return _workers(3)


@pytest.mark.parametrize('balance', [
    [1, 1, 1, 1],
    [1, 2, 1],
    [3, 1]
])
def test_module_partition(module, balance):
    for rank, b in enumerate(balance):
        part = get_module_partition(module, rank, balance, None)
        assert len(part) == b


@pytest.mark.timeout(10)
@pytest.mark.parametrize('balance', [[2, 1, 1]])
def test_pipeline(module, workers, balance):
    global_ctx = FakeTrainingGloablContext()
    with patch.object(DistributedGPipe, "_get", global_ctx.fake_get), \
            patch.object(DistributedGPipe, "_put", global_ctx.fake_put):

        world_size = len(balance)
        module.train()
        partitions = [
            DistributedGPipe(module, i, workers, balance, 1) for i in range(world_size)
        ]
        fake_data = torch.randn([5, 28, 28])
        fake_target = torch.randn([5, 10])
        for i, part in enumerate(partitions):
            output = part.forward(0, fake_data if i == 0 else None)
        loss = nn.CrossEntropyLoss()
        output = loss(output, fake_target)

        for i in reversed(range(world_size)):
            loss_v = output if (i == world_size - 1) else None
            partitions[i].backward(0, loss_v)


@pytest.mark.timeout(10)
@pytest.mark.parametrize("batch_size,chunks", [[32, 3]])
def test_distributed_data_loader(batch_size: int, chunks: int):
    import torchgpipe.distributed.context as context
    global_ctx = FakeTrainingGloablContext()
    with patch.object(DistributedGPipeDataLoader, "_put", global_ctx.fake_put), \
            patch.object(DistributedGPipeDataLoader, "_get", global_ctx.fake_get):
        trans = torchvision.transforms.ToTensor()
        mnist_train = torchvision.datasets.FashionMNIST(
            root="./data", train=True, transform=trans, download=True)
        train_iter = dataloader.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

        num_iteration = 1000
        loader0 = DistributedGPipeDataLoader(
            train_iter, 0, chunks, num_iteration, False, "worker2")
        loader1 = DistributedGPipeDataLoader(
            train_iter, 1, chunks, num_iteration, False, "worker2")
        loader2 = DistributedGPipeDataLoader(
            train_iter, 2, chunks, num_iteration, True, "worker2")
        cnt = 0

        for d0, d1, d2 in zip(loader0, loader1, loader2):
            assert d0[0] is not None and d0[1] is None
            assert d1[0] is None and d1[1] is None
            assert d2[0] is None and d2[1] is not None
            cnt += 1

        assert cnt == (num_iteration * chunks)
