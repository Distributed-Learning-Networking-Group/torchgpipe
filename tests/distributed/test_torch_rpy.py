from contextlib import ExitStack, contextmanager
import os
import socket
from typing import Tuple

import pytest
from pytest import fixture
import torch
from torch import Tensor
from torch import multiprocessing as mp
from torch.distributed import rpc

from torchgpipe.distributed.context import worker


def next_free_port(port=8793, max_port=65535):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while port <= max_port:
        try:
            sock.bind(('', port))
            sock.close()
            return port
        except OSError:
            port += 1
    raise IOError('no free ports')


@contextmanager
def launch_background(
    name: str,
    addr: Tuple[str, str],
    rank: int,
    world_size: int,
):
    with worker(name):
        master_addr, master_port = addr
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        rpc.init_rpc(name, rank=rank, world_size=world_size)
        yield
        rpc.shutdown()


def send_tensor(
    name: str,
    addr: Tuple[str, str],
    rank: int,
    world_size: int,
    queue: mp.Queue
):
    with launch_background(name, addr, rank, world_size):
        x = torch.tensor([1.0])
        x.requires_grad_()
        queue.put(x)


@fixture
def addrs():
    return ("127.0.0.1", str(next_free_port()))


@pytest.mark.timeout(10)
def test_tensor_transfer(addrs):
    with ExitStack() as stack:
        q = mp.Queue()
        p = mp.Process(target=send_tensor, args=("worker1", addrs, 1, 2, q))
        p.start()
        stack.callback(p.terminate)
        stack.callback(p.join)
        with launch_background("worker0", addrs, 0, 2):
            t: Tensor = q.get()
        assert t.requires_grad
