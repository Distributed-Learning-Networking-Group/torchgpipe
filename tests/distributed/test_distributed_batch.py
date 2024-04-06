import torch

import torchgpipe.distributed.batch as batch


def test_requires_grad_trait():
    t = torch.tensor([1.0]).requires_grad_()
    b = batch.DistributedBatch(t)
    t_ = b.requires_grad_trait()

    assert isinstance(t_, tuple)
    assert len(t_) == 1
    assert t_[0] is t

    a = torch.tensor([1.0]).requires_grad_()
    b = torch.tensor([1.0])
    c = torch.tensor([1.0]).requires_grad_()

    t = batch.DistributedBatch((a, b, c))
    t_ = t.requires_grad_trait()

    assert isinstance(t_, tuple)
    assert len(t_) == 2
    assert t_[0] is a and t[2] is c and a is not c


def test_grad_trait():
    a = torch.tensor([1.0]).requires_grad_()
    b = torch.tensor([1.0])
    c = torch.tensor([1.0]).requires_grad_()
    d = torch.tensor([1.0])
    a.grad = b

    t = batch.DistributedBatch((a, c, d))
    t_ = t.grad_trait()

    assert isinstance(t_, tuple)
    assert len(t_) == 2
    assert (t_[0] is b and t_[1] is None)
