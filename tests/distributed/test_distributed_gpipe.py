from typing import Any, List
from unittest import mock

import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import dataloader
import torchvision

from torchgpipe import distributed
from torchgpipe.distributed import gpipe, run


class FakeRpc:

    @staticmethod
    def remote(
        to: Any,  # pylint: disable=unused-argument
        func: Any,
        args: Any | None = None,
        kwargs: Any | None = None,
        timeout: float = None  # pylint: disable=unused-argument
    ):
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        func(*args, **kwargs)


def _names(prefix: str, num_stages: int):
    return {
        i: f"{prefix}_worker{i}" for i in range(num_stages)
    }


def _module():
    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(784, 256),
                        nn.ReLU(),
                        nn.Linear(256, 10))

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weights)
    return net


@pytest.fixture
def module():
    return _module()


@pytest.fixture
def data_loader():
    batch_size = 32
    trans = torchvision.transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True)
    train_iter = dataloader.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    return train_iter


@pytest.mark.timeout(10)
@mock.patch("torchgpipe.distributed.context.rpc", FakeRpc())
def test_distributed_dataloader(data_loader):
    num_stage = 3
    num_microbatchs = 8
    names = _names(
        "test_distributed_dataloader", num_stage
    )

    device = torch.device("cpu")

    with run(names[0], num_microbatchs, device), \
            run(names[1], num_microbatchs, device), \
            run(names[2], num_microbatchs, device):

        loaders = [
            gpipe.DistributedGPipeDataLoader(
                names[i],
                data_loader,
                num_microbatchs,
                len(data_loader),
                i == 0,
                i == (num_stage - 1),
                names[num_stage - 1],
            )
            for i in range(num_stage)
        ]

        for loader in loaders:
            assert (len(loader) == len(data_loader))

        for i, loader in enumerate(loaders):
            data, target = next(iter(loader))
            if i == 0:
                assert (data is not None and target is None)
            if i == 1:
                assert (data is None and target is None)
            if i == 2:
                assert (data is None and target is not None)


@pytest.mark.timeout(10)
@mock.patch("torchgpipe.distributed.context.rpc", FakeRpc())
def test_gpipe_train(module, data_loader):

    num_stage = 3
    num_microbatchs = 8
    names = _names(
        "test_gpipe_train", num_stage
    )
    balance = [1, 2, 1]

    device = torch.device("cpu")

    module_validate = _module()
    module_validate.load_state_dict(module.state_dict())

    with run(names[0], num_microbatchs, device), \
            run(names[1], num_microbatchs, device), \
            run(names[2], num_microbatchs, device):
        loaders = [
            iter(gpipe.DistributedGPipeDataLoader(
                names[i],
                data_loader,
                num_microbatchs,
                len(data_loader),
                i == 0,
                i == (num_stage - 1),
                names[num_stage - 1],
            ))
            for i in range(num_stage)
        ]

        modules = [
            gpipe.DistributedGPipe(
                module,
                i,
                names,
                balance=balance,
                microbatch_chunks=num_microbatchs,
                device=device
            )
            for i in range(num_stage)
        ]

        data = None
        target = None
        outputs = None
        validate_data = None
        validate_target = None
        loss = torch.nn.CrossEntropyLoss()

        # run gpipe module
        for data_iter, stage in zip(loaders, modules):
            stage.model().train()
            data, target = next(data_iter)
            validate_data = data if data is not None else validate_data
            validate_target = target if target is not None else validate_target
            outputs = stage.forward(data)

        losses = distributed.loss(outputs, target, loss)

        for stage in reversed(modules):
            stage.backward(losses if stage.is_last_stage() else None)

        # run validate module
        module_validate.train()
        output = module_validate(validate_data)
        loss_value = loss(output, validate_target)
        loss_value.backward()

        # grab gpipe&validate module paramters for comparision
        gpipe_params: List[Tensor] = []
        for m in modules:
            gpipe_params.extend(m.model().parameters())
        validate_params = [p for p in module_validate.parameters()]

        assert len(validate_params) == len(gpipe_params)

        for param, param_valid in zip(gpipe_params, validate_params):
            assert param.shape == param_valid.shape
            assert torch.allclose(param.grad, param_valid.grad)
