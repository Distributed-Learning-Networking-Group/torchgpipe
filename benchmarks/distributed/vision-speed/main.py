"""ResNet-101 Accuracy Benchmark"""
import os
import platform
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import click
import resnet
import torch
from torch import nn
from torch.distributed import rpc
from torch.optim import SGD
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import tqdm
import vgg

from torchgpipe import distributed
from torchgpipe.balance import balance_by_time
from torchgpipe.distributed.gpipe import DistributedGPipe, DistributedGPipeDataLoader

Stuffs = Tuple[nn.Module, int, List[torch.device]]  # (model, batch_size, devices)
Experiment = Callable[[nn.Module, List[int]], Stuffs]


class Experiments:

    @staticmethod
    def naive128(model: nn.Module, devices: List[int]) -> Stuffs:
        device = devices[0]
        model.to(device)
        return model, [torch.device(device)]


EXPERIMENTS: Dict[str, Experiment] = {
    'naive-128': Experiments.naive128,
}

MODELS: Dict[str, Callable[[int, int], torch.nn.Module]] = {
    'resnet101': resnet.resnet101,
    'resnet50': resnet.resnet50,
    'vgg16': vgg.vgg16,
}


def dataloaders(
        name: str,
        batch_size: int,
        chunks: int,
        first_stage: bool,
        last_stage: bool,
        last_stage_name: str,
        # [train dataset path, test dataset path]
        dataset_path: Tuple[str, str],
) -> Tuple[DataLoader, DataLoader]:

    if dataset_path is not None:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225]),
        ])
        train_dataset = torchvision.datasets.ImageFolder(dataset_path[0], transform)
        test_dataset = torchvision.datasets.ImageFolder(dataset_path[1], transform)
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225]),
        ])
        train_dataset = torchvision.datasets.CIFAR10(
            root="/data", train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.CIFAR10(
            root="/data", train=False, transform=transform, download=True)

    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loader = DistributedGPipeDataLoader(
        name=name,
        data_loader=train_iter,
        chunks=chunks,
        num_iterations=len(train_iter),
        first_stage=first_stage,
        last_stage=last_stage,
        last_stage_name=last_stage_name
    )

    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DistributedGPipeDataLoader(
        name=name,
        data_loader=test_iter,
        chunks=chunks,
        num_iterations=len(test_iter),
        first_stage=first_stage,
        last_stage=last_stage,
        last_stage_name=last_stage_name
    )

    return train_loader, test_loader


def parse_devices(
    ctx: Any,  # pylint: disable=unused-argument
    param: Any,  # pylint: disable=unused-argument
    value: Optional[str]
) -> List[int]:
    if value is None:
        return list(range(torch.cuda.device_count()))
    return [int(x) for x in value.split(',')]


def evaluate(
    model: DistributedGPipe,
    batch_size: int,
    chunks: int,
    last_stage: bool,
    device: torch.device,
    dataloader: DataLoader,
) -> Tuple[float, float]:
    assert (batch_size % chunks) == 0, \
        "undivisible microbatches are not currentyly supported"
    data_tested = 0
    loss_sum = torch.zeros(1, device=device)
    accuracy_sum = torch.zeros(1, device=device)
    model.model().eval()
    loss_function = nn.CrossEntropyLoss()
    with torch.no_grad():
        pbar = tqdm.tqdm(dataloader, unit="sample",
                         unit_scale=float(batch_size), desc='valid |')
        for inputs, targets in pbar:
            data_tested += batch_size
            current_batch = batch_size // chunks
            outputs = model.forward(inputs)

            if last_stage:
                targets = targets.to(device=device)
                losses = distributed.loss(outputs, targets, loss_function)
                for output, loss, target in zip(outputs, losses, targets.chunk(chunks)):
                    loss_sum += loss.detach() * (current_batch)
                    _, predicted = torch.max(output.value, 1)
                    correct = (predicted == target).sum()
                    accuracy_sum += correct

    if last_stage:
        loss = loss_sum / data_tested
        accuracy = accuracy_sum / data_tested
        return loss.item(), accuracy.item()
    return 0.0, 0.0


def run_epoch(
    model: DistributedGPipe,
    epoch: int,
    epochs: int,
    batch_size: int,
    chunks: int,
    last_stage: bool,
    device: torch.device,
    train_dataloader: torch.utils.data.DataLoader,
    valid_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
) -> Tuple[float, float]:
    assert (batch_size % chunks) == 0, "undivisible microbatches are not currentyly supported"
    microbatch_size = batch_size // chunks
    torch.cuda.synchronize(device)
    tick = time.time()

    data_trained = 0
    loss_sum = torch.zeros(1, device=device)
    model.model().train()
    loss_function = nn.CrossEntropyLoss()
    losses = None

    pbar = tqdm.tqdm(train_dataloader, unit="sample", unit_scale=float(batch_size))
    for inputs, targets in pbar:
        optimizer.zero_grad()
        outputs = model.forward(inputs)
        if last_stage:
            losses = distributed.loss(outputs, targets, loss_function)
            for loss in losses:
                loss_sum += loss.detach() * (microbatch_size)
        model.backward(losses)
        optimizer.step()

        data_trained += batch_size
        throughput = data_trained / (time.time()-tick)
        pbar.set_description(
            f'train | {epoch}/{epochs} epoch | loss: {loss_sum.item() / data_trained}')

    torch.cuda.synchronize(device)
    tock = time.time()

    train_loss = loss_sum.item() / data_trained

    valid_loss, valid_accuracy = evaluate(
        model,
        batch_size,
        chunks,
        last_stage,
        device,
        valid_dataloader
    )

    torch.cuda.synchronize(device)

    elapsed_time = tock - tick
    throughput = data_trained / elapsed_time
    print(f'{epoch}/{epochs} epoch | train loss:{train_loss} {throughput}samples/sec | '
          f'valid loss:{valid_loss} accuracy{valid_accuracy}')

    return throughput, elapsed_time


@click.command()
@click.pass_context
@click.argument(
    'experiment',
    type=click.Choice(sorted(EXPERIMENTS.keys())),
)
@click.option(
    '--chunks', '-c',
    type=int,
    default=4,
    help='Number of microbatches (default: 4)',
)
@click.option(
    '--master', '-a',
    type=str,
    default='localhost:11451',
    help='master address',
)
@click.option(
    '--epochs', '-e',
    type=int,
    default=10,
    help='Number of epochs (default: 10)',
)
@click.option(
    '--devices', '-d',
    metavar='0,1,2,3',
    callback=parse_devices,
    help='Device IDs to use (default: all CUDA devices)',
)
@click.option(
    '--rank', '-r',
    type=int,
    help='Worker rank in distributed training setup.',
)
@click.option(
    '--world', '-w',
    type=int,
    help='Total worker number in distributed training setup.',
)
@click.option(
    '--model', '-m',
    type=str,
    help='model to train.',
)
@click.option(
    '--balance', '-b',
    type=str,
    help='model to train.',
)
@click.option(
    '--batch-size', '-s',
    type=int,
    default=128,
    help='mini batch size.',
)
@click.option(
    '--dataset-path', '-p',
    type=str,
    default=None,
    help='path to train/test datasets, sperated by \',\'.',
)
def cli(
    ctx: click.Context,  # pylint: disable=unused-argument
    experiment: str,
    epochs: int,
    master: str,
    devices: List[int],
    rank: int,
    world: int,
    chunks: int,
    model: str,
    balance: str,
    batch_size: int,
    dataset_path: str,
) -> None:
    """vision model training speed benchmark"""
    device = devices[0]
    workers = {rk: f"worker{rk}" for rk in range(world)}

    with distributed.run(workers[rank], chunks, torch.device(device)):

        relu_inplace = False
        model_raw = MODELS[model](num_classes=10, inplace=relu_inplace)

        f = EXPERIMENTS[experiment]

        model_local, _devices = f(model_raw, devices)
        if balance is None:
            balance_ = balance_by_time(world, model_local, torch.empty(128, 3, 224, 224))
        else:
            balance_ = [int(x) for x in balance.split(",")]
        print("balance: ", balance_)
        print("batchsize: ", batch_size)

        # Prepare dataloaders.
        train_dataloader, valid_dataloader = dataloaders(
            workers[rank],
            batch_size,
            chunks,
            rank == 0,
            rank == world - 1,
            workers[world-1],
            None if dataset_path is None else dataset_path.split(","),
        )
        # TODO: distributed balance information
        model = DistributedGPipe(model_local, rank, workers, balance_, chunks, device=devices[0])
        optimizer = SGD(model.model().parameters(), lr=0.01,
                        momentum=0.9)

        # HEADER ==================================================

        title = f'''{experiment}, {len(_devices)} devices, {batch_size} batch, {epochs} epochs'''

        click.echo(title)
        click.echo('python: {}, torch: {}, cudnn: {}, cuda: {}, gpu: {}' .format(  # pylint: disable=C0209
            platform.python_version(),
            torch.__version__,
            torch.backends.cudnn.version(),
            torch.version.cuda,
            torch.cuda.get_device_name(device)))

        # TRAIN ===================================================

        throughputs = []
        elapsed_times = []

        addr, port = master.split(":")
        os.environ["MASTER_ADDR"] = addr
        os.environ["MASTER_PORT"] = port
        print(f"init rpc with rank{rank}, world size {world}, master: {addr}:{port}")
        rpc.init_rpc(workers[rank], None, rank, world)
        last_stage = rank == (world - 1)

        for epoch in range(epochs):
            throughput, elapsed_time = run_epoch(
                model=model,
                epoch=epoch,
                epochs=epochs,
                batch_size=batch_size,
                chunks=chunks,
                last_stage=last_stage,
                device=device,
                train_dataloader=train_dataloader,
                valid_dataloader=valid_dataloader,
                optimizer=optimizer
            )
            throughputs.append(throughput)
            elapsed_times.append(elapsed_time)

        rpc.shutdown()

        # RESULT ==================================================


if __name__ == '__main__':
    cli()  # pylint: disable=no-value-for-parameter
