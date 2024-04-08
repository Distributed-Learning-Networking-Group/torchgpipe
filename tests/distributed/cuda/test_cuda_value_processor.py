import pytest
import torch
from torchgpipe.distributed import cuda


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count == 0,
    reason="current pytorch does have cuda extension"
)
def test_device_to_host():
    device = torch.device("cuda")
    value_processor = cuda.CudaValueCopyProcessor(device, 1)

    a = torch.tensor([1.0])
    b = torch.tensor([1.0])
    c = torch.tensor([1.0])
    d = torch.tensor([1.0])

    a_ = value_processor.device_to_host(a, 0)

    assert isinstance(a_, torch.Tensor)
    assert a_.device == torch.device("cpu")

    values = tuple(t.to(device) for t in (b, c, d))
    values_ = value_processor.device_to_host(values, 0)

    assert isinstance(values_, tuple)
    for v in values_:
        assert v.device == torch.device("cpu")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count == 0,
    reason="current pytorch does have cuda extension"
)
def test_host_to_device():
    device = torch.device("cuda:0")
    value_processor = cuda.CudaValueCopyProcessor(device, 1)

    a = torch.tensor([1.0])
    b = torch.tensor([1.0])
    c = torch.tensor([1.0])
    d = torch.tensor([1.0])

    a_ = value_processor.host_to_device(a, 0)

    assert isinstance(a_, torch.Tensor)
    assert a_.device == device

    values = (b, c, d)
    values_ = value_processor.host_to_device(values, 0)

    assert isinstance(values_, tuple)
    for v in values_:
        assert v.device == device
