from queue import Queue
from typing import Any, Dict
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.utils import data
import torchvision

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
        return channels.setdefault(id, Queue()).get()

    def fake_put(self, name: str, id: int, value: Any, backward=False):
        ctx = self.ctxs.setdefault(name, TrainingContext(name))
        if backward:
            channels = ctx.backward_channels
        else:
            channels = ctx.forward_channels
        value = detach(value)
        return channels.setdefault(id, Queue()).put(value)


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


@pytest.mark.parametrize("batch_size,chunks", [[32, 8]])
def test_distributed_data_loader(batch_size: int, chunks: int):
    trans = torchvision.transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True)
    train_iter = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    batch_num = len(train_iter)
    train_iter = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

    loader = DistributedGPipeDataLoader(train_iter, 0, chunks)
    cnt = 0
    for _ in loader:
        cnt += 1
    assert cnt == batch_num * chunks
