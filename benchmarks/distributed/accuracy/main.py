"""ResNet-101 Accuracy Benchmark"""
import os
import platform
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import click
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torch.distributed.rpc as rpc
import torchgpipe.distributed.context as gpipe_context
from torchgpipe.balance import balance_by_time
from torchgpipe.distributed.gpipe import DistributedGPipe, DistributedGPipeDataLoader

import resnet
import vgg

Stuffs = Tuple[nn.Module, int, List[torch.device]]  # (model, batch_size, devices)
Experiment = Callable[[nn.Module, List[int]], Stuffs]


class Experiments:

    @staticmethod
    def naive128(model: nn.Module, devices: List[int]) -> Stuffs:
        batch_size = 128
        device = devices[0]
        model.to(device)
        return model, batch_size, [torch.device(device)]


EXPERIMENTS: Dict[str, Experiment] = {
    'naive-128': Experiments.naive128,
}

MODELS : Dict[str, Callable[[int, int], torch.nn.Module]] = {
    'resnet101': resnet.resnet101,
    'resnet50': resnet.resnet50,
    'vgg16': vgg.vgg16,
}



def dataloaders(batch_size: int, rank, chunks, last_stage, last_stage_name) -> Tuple[DataLoader, DataLoader]:

    transform =transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                                ])


    mnist_train = torchvision.datasets.CIFAR10(
        root="./data", train=True, transform=transform, download=True)
    train_iter = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    train_loader = DistributedGPipeDataLoader(
        train_iter, rank, chunks, len(train_iter), last_stage, last_stage_name)

    mnist_test = torchvision.datasets.CIFAR10(
        root="./data", train=True, transform=transform, download=True)
    test_iter = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)
    test_loader = DistributedGPipeDataLoader(
        test_iter, rank, chunks, len(test_iter), last_stage, last_stage_name)

    return train_loader, test_loader


BASE_TIME: float = 0


def hr() -> None:
    """Prints a horizontal line."""
    width, _ = click.get_terminal_size()
    click.echo('-' * width)


def log(msg: str, clear: bool = False, nl: bool = True) -> None:
    """Prints a message with elapsed time."""
    if clear:
        # Clear the output line to overwrite.
        width, _ = click.get_terminal_size()
        click.echo('\b\r', nl=False)
        click.echo(' ' * width, nl=False)
        click.echo('\b\r', nl=False)

    t = time.time() - BASE_TIME
    h = t // 3600
    t %= 3600
    m = t // 60
    t %= 60
    s = t

    click.echo('%02d:%02d:%02d | ' % (h, m, s), nl=False)
    click.echo(msg, nl=nl)


def parse_devices(ctx: Any, param: Any, value: Optional[str]) -> List[int]:
    if value is None:
        return list(range(torch.cuda.device_count()))
    return [int(x) for x in value.split(',')]


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
    help='Number of epochs (default: 4)',
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
    '--skip-epochs', '-k',
    type=int,
    default=1,
    help='Number of epochs to skip in result (default: 1)',
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
def cli(ctx: click.Context,
        experiment: str,
        epochs: int,
        master: str,
        skip_epochs: int,
        devices: List[int],
        rank: int,
        world: int,
        chunks: int,
        model: str,
        ) -> None:
    """ResNet-101 Accuracy Benchmark"""
    if skip_epochs > epochs:
        ctx.fail('--skip-epochs=%d must be less than --epochs=%d' % (skip_epochs, epochs))

    relu_inplace = False
    model_raw = MODELS[model](num_classes=1000, inplace=relu_inplace)

    f = EXPERIMENTS[experiment]
    try:
        workers = {rk: f"worker{rk}" for rk in range(world)}
    except ValueError as exc:
        # Examples:
        #   ValueError: too few devices to hold given partitions (devices: 1, paritions: 2)
        ctx.fail(str(exc))

    model_local, batch_size, _devices = f(model_raw, devices)
    # TODO: distributed balance information
    model = DistributedGPipe(model_local, rank, workers, balance_by_time(
        world, model, torch.empty(128, 3, 28, 28), device=devices[0]), chunks, device=devices[0])


    # Prepare dataloaders.
    train_dataloader, valid_dataloader = dataloaders(
        batch_size,
        rank,
        chunks,
        rank == world - 1,
        workers[world-1]
    )

    # Optimizer with LR scheduler
    steps = len(train_dataloader)
    lr_multiplier = max(1.0, batch_size / 256)
    optimizer = SGD(model.model().parameters(), lr=0.1,
                    momentum=0.9, weight_decay=1e-4, nesterov=True)

    def gradual_warmup_linear_scaling(step: int) -> float:
        epoch = step / steps

        # Gradual warmup
        warmup_ratio = min(4.0, epoch) / 4.0
        multiplier = warmup_ratio * (lr_multiplier - 1.0) + 1.0

        if epoch < 30:
            return 1.0 * multiplier
        elif epoch < 60:
            return 0.1 * multiplier
        elif epoch < 80:
            return 0.01 * multiplier
        return 0.001 * multiplier

    scheduler = LambdaLR(optimizer, lr_lambda=gradual_warmup_linear_scaling)

    # HEADER ======================================================================================

    title = '%s, %d devices, %d batch, %d-%d epochs'\
            '' % (experiment, len(_devices), batch_size, skip_epochs+1, epochs)

    device = devices[0]

    click.echo(title)
    click.echo('python: %s, torch: %s, cudnn: %s, cuda: %s, gpu: %s' % (
        platform.python_version(),
        torch.__version__,
        torch.backends.cudnn.version(),
        torch.version.cuda,
        torch.cuda.get_device_name(device)))

    # TRAIN =======================================================================================

    global BASE_TIME
    BASE_TIME = time.time()

    def evaluate(dataloader: DataLoader, last_stage: bool) -> Tuple[float, float]:
        assert (batch_size % chunks) == 0, "undivisible microbatches are not currentyly supported"
        tick = time.time()
        steps = len(dataloader)
        data_tested = 0
        loss_sum = torch.zeros(1, device=device)
        accuracy_sum = torch.zeros(1, device=device)
        model.model().eval()
        with torch.no_grad():
            for i, (input, target) in enumerate(dataloader):
                current_batch = batch_size // chunks
                data_tested += current_batch
                output = model.forward(i % chunks, input)

                if last_stage:
                    target = target.to(device=device)
                    loss = F.cross_entropy(output, target)
                    loss_sum += loss * current_batch

                    _, predicted = torch.max(output, 1)
                    correct = (predicted == target).sum()
                    accuracy_sum += correct

                    percent = i / steps * 100
                    throughput = data_tested / (time.time() - tick)
                    log('valid | %d%% | %.3f samples/sec (estimated)'
                        '' % (percent, throughput), clear=True, nl=False)
        if last_stage:
            loss = loss_sum / data_tested
            accuracy = accuracy_sum / data_tested
            return loss.item(), accuracy.item()
        return 0.0, 0.0

    def run_epoch(epoch: int, last_stage: bool) -> Tuple[float, float]:
        torch.cuda.synchronize(device)
        tick = time.time()

        steps = len(train_dataloader)
        data_trained = 0
        loss_sum = torch.zeros(1, device=device)
        model.model().train()

        for i, (input, target) in enumerate(train_dataloader):
            data_trained += batch_size
            output = model.forward(i % chunks, input)
            loss = None
            if last_stage:
                target = target.to(device=device, non_blocking=True)
                loss = F.cross_entropy(output, target)

            optimizer.zero_grad()
            model.backward(i % chunks, loss)

            optimizer.step()
            scheduler.step()
            if last_stage:
                loss_sum += loss.detach() * batch_size

            percent = i / steps * 100
            throughput = data_trained / (time.time()-tick)
            log('train | %d/%d epoch (%d%%) | lr:%.5f | %.3f samples/sec (estimated)'
                '' % (epoch+1, epochs, percent, scheduler.get_lr()[0], throughput),
                clear=True, nl=False)

        torch.cuda.synchronize(device)
        tock = time.time()

        train_loss = loss_sum.item() / data_trained
        valid_loss, valid_accuracy = evaluate(valid_dataloader, last_stage)
        torch.cuda.synchronize(device)

        elapsed_time = tock - tick
        throughput = data_trained / elapsed_time
        log('%d/%d epoch | lr:%.5f | train loss:%.3f %.3f samples/sec | '
            'valid loss:%.3f accuracy:%.3f'
            '' % (epoch+1, epochs, scheduler.get_lr()[0], train_loss, throughput,
                  valid_loss, valid_accuracy),
            clear=True)

        return throughput, elapsed_time

    throughputs = []
    elapsed_times = []

    with gpipe_context.worker(workers[rank], chunks):
        addr, port = master.split(":")
        os.environ["MASTER_ADDR"] = addr
        os.environ["MASTER_PORT"] = port 
        log(f"init rpc with rank{rank}, world size {world}, master: {addr}:{port}")
        rpc.init_rpc(workers[rank], None, rank, world)
        last_stage = rank == (world - 1)
        hr()

        for epoch in range(epochs):
            throughput, elapsed_time = run_epoch(epoch, last_stage)

            if epoch < skip_epochs:
                continue

            throughputs.append(throughput)
            elapsed_times.append(elapsed_time)

        _, valid_accuracy = evaluate(valid_dataloader, last_stage)
        hr()

        rpc.shutdown()

    # RESULT ======================================================================================

    # pipeline-4, 2-10 epochs | 200.000 samples/sec, 123.456 sec/epoch (average)
    n = len(throughputs)
    throughput = sum(throughputs) / n if n > 0 else 0.0
    elapsed_time = sum(elapsed_times) / n if n > 0 else 0.0
    click.echo('%s | valid accuracy: %.4f | %.3f samples/sec, %.3f sec/epoch (average)'
               '' % (title, valid_accuracy, throughput, elapsed_time))


if __name__ == '__main__':
    cli()
