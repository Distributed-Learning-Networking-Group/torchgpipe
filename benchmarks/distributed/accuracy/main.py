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
import tqdm
import h5py
import numpy as np

import resnet
import vgg
import bert


class CrossEntropyWrapper(nn.Module):
    def __init__(self, vocab_size):
        super(CrossEntropyWrapper, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size
        # print("vocab")
        # print(vocab_size)

    def forward(self, prediction_scores, masked_lm_labels=None):
        if masked_lm_labels is not None:

            masked_lm_loss = self.cross_entropy(
                prediction_scores.view(-1, self.vocab_size), masked_lm_labels.view(-1))
            return masked_lm_loss
        else:

            return prediction_scores


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
    'bert': bert.bert
}


class pretraining_dataset(torch.utils.data.Dataset):

    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
        keys = ['input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids',
                'next_sentence_labels']
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])

    def __getitem__(self, index):
        [input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels] = [
            torch.from_numpy(input[index].astype(np.int64)) if indice < 5 else torch.from_numpy(
                np.asarray(input[index].astype(np.int64))) for indice, input in enumerate(self.inputs)]

        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        index = self.max_pred_length
        # store number of  masked tokens in index
        padded_mask_indices = (masked_lm_positions == 0).nonzero(as_tuple=False)
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]
        # print("nsp_label",next_sentence_labels)
        return (input_ids, segment_ids, input_mask), masked_lm_labels


def dataloaders(
        batch_size: int,
        rank,
        chunks,
        last_stage,
        last_stage_name,
        # [train dataset path, test dataset path]
        dataset_path: Tuple[str, str],
        is_bert: bool,
) -> Tuple[DataLoader, DataLoader]:

    if is_bert:
        train_dataset = pretraining_dataset(dataset_path[0], 80)
        test_dataset = pretraining_dataset(dataset_path[0], 80)
    elif dataset_path is not None:
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
            root=".data", train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.CIFAR10(
            root=".data", train=False, transform=transform, download=True)

    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loader = DistributedGPipeDataLoader(
        train_iter, rank, chunks, len(train_iter), last_stage, last_stage_name)

    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DistributedGPipeDataLoader(
        test_iter, rank, chunks, len(test_iter), last_stage, last_stage_name)

    return train_loader, test_loader


BASE_TIME: float = 0


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
        balance: str,
        batch_size: int,
        dataset_path: str,
        ) -> None:
    """ResNet-101 Accuracy Benchmark"""
    if skip_epochs > epochs:
        ctx.fail('--skip-epochs=%d must be less than --epochs=%d' % (skip_epochs, epochs))

    model_name = model

    relu_inplace = False
    model_raw = MODELS[model](num_classes=10, inplace=relu_inplace)

    f = EXPERIMENTS[experiment]
    try:
        workers = {rk: f"worker{rk}" for rk in range(world)}
    except ValueError as exc:
        # Examples:
        #   ValueError: too few devices to hold given partitions (devices: 1, paritions: 2)
        ctx.fail(str(exc))

    model_local, _devices = f(model_raw, devices)
    if balance is None:
        balance_ = balance_by_time(world, model_local, torch.empty(128, 3, 224, 224))
    else:
        balance_ = [int(x) for x in balance.split(",")]
    print("balance: ", balance_)
    print("batchsize: ", batch_size)
    # TODO: distributed balance information
    model = DistributedGPipe(model_local, rank, workers, balance_, chunks, device=devices[0])

    # Prepare dataloaders.
    train_dataloader, valid_dataloader = dataloaders(
        batch_size,
        rank,
        chunks,
        rank == world - 1,
        workers[world-1],
        None if dataset_path is None else dataset_path.split(","),
        model_name == "bert"
    )

    # Optimizer with LR scheduler
    # steps = len(train_dataloader)
    # lr_multiplier = max(1.0, batch_size / 256)
    optimizer = SGD(model.model().parameters(), lr=0.001,
                    momentum=0.9)
    # weight_decay=1e-4, nesterov=True)

    # def gradual_warmup_linear_scaling(step: int) -> float:
    #     epoch = step / steps

    #     # Gradual warmup
    #     warmup_ratio = min(4.0, epoch) / 4.0
    #     multiplier = warmup_ratio * (lr_multiplier - 1.0) + 1.0

    #     if epoch < 30:
    #         return 1.0 * multiplier
    #     elif epoch < 60:
    #         return 0.1 * multiplier
    #     elif epoch < 80:
    #         return 0.01 * multiplier
    #     return 0.001 * multiplier

    # scheduler = LambdaLR(optimizer, lr_lambda=gradual_warmup_linear_scaling)

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
        data_tested = 0
        loss_sum = torch.zeros(1, device=device)
        accuracy_sum = torch.zeros(1, device=device)
        model.model().eval()
        with torch.no_grad():
            pbar = tqdm.tqdm(dataloader, unit="sample",
                             unit_scale=float(batch_size), desc='valid |')
            for input, targets in pbar:
                data_tested += batch_size
                current_batch = batch_size // chunks
                outputs = model.forward(input)

                if last_stage:
                    targets = targets.to(device=device)
                    for output, target in zip(outputs, targets.chunk(chunks)):
                        loss = F.cross_entropy(output, target)
                        loss_sum += loss.detach() * (current_batch)
                        _, predicted = torch.max(output, 1)
                        correct = (predicted == target).sum()
                        accuracy_sum += correct

        if last_stage:
            loss = loss_sum / data_tested
            accuracy = accuracy_sum / data_tested
            return loss.item(), accuracy.item()
        return 0.0, 0.0

    def run_epoch(epoch: int, last_stage: bool) -> Tuple[float, float]:

        loss_func = CrossEntropyWrapper(30528)

        assert (batch_size % chunks) == 0, "undivisible microbatches are not currentyly supported"
        microbatch_size = batch_size // chunks
        torch.cuda.synchronize(device)
        tick = time.time()

        steps = len(train_dataloader)
        data_trained = 0
        loss_sum = torch.zeros(1, device=device)
        model.model().train()
        losses = None if not last_stage else [None for _ in range(chunks)]

        pbar = tqdm.tqdm(train_dataloader, unit="sample", unit_scale=float(batch_size))
        for input, targets in pbar:
            optimizer.zero_grad()
            if input is not None:
                input[2] = input[2].unsqueeze(1).unsqueeze(2)
                input[2] = input[2].to(dtype=torch.float32)  # fp16 compatibility
                input[2] = (1.0 - input[2]) * -10000.0
                input = tuple(input)
            outputs = model.forward(input)
            if last_stage:
                targets = targets.to(device=device, non_blocking=True)
                for mbatch, output, target in zip(range(chunks), outputs, targets.chunk(chunks)):
                    loss = loss_func(output, target)
                    loss_sum += loss.detach() * (microbatch_size)
                    losses[mbatch] = loss / chunks
            model.backward(losses)
            optimizer.step()
            # scheduler.step()

            data_trained += batch_size
            throughput = data_trained / (time.time()-tick)
            pbar.set_description('train | %d/%d epoch | loss: %.3f'
                                 '' % (epoch+1, epochs, loss_sum.item() / data_trained))

        torch.cuda.synchronize(device)
        tock = time.time()

        train_loss = loss_sum.item() / data_trained
        valid_loss, valid_accuracy = evaluate(valid_dataloader, last_stage)
        torch.cuda.synchronize(device)

        elapsed_time = tock - tick
        throughput = data_trained / elapsed_time
        print('%d/%d epoch | train loss:%.3f %.3f samples/sec | '
              'valid loss:%.3f accuracy:%.3f'
              '' % (epoch+1, epochs, train_loss, throughput,
                    valid_loss, valid_accuracy))

        return throughput, elapsed_time

    throughputs = []
    elapsed_times = []

    with gpipe_context.worker(workers[rank], chunks):
        addr, port = master.split(":")
        os.environ["MASTER_ADDR"] = addr
        os.environ["MASTER_PORT"] = port
        print(f"init rpc with rank{rank}, world size {world}, master: {addr}:{port}")
        rpc.init_rpc(workers[rank], None, rank, world)
        last_stage = rank == (world - 1)

        for epoch in range(epochs):
            throughput, elapsed_time = run_epoch(epoch, last_stage)

            if epoch < skip_epochs:
                continue

            throughputs.append(throughput)
            elapsed_times.append(elapsed_time)

        _, valid_accuracy = evaluate(valid_dataloader, last_stage)

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
