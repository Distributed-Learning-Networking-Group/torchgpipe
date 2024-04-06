# pylint: disable=W0702
from typing import Any
from unittest import mock
from unittest.mock import MagicMock

import pytest
import torch
import torchgpipe.distributed.context as context

MICROBATCHES = 12


def test_registry():
    ctx_name = "test_registry"
    value_processor = MagicMock()
    ctx = context.DistributedContext(
        MICROBATCHES, ctx_name, value_processor
    )
    context.DistributedContextRegistry.registrate(ctx)

    try:
        context.DistributedContextRegistry.registrate(ctx)
        pytest.fail(f"context {ctx.name} registrate twice")
    except ValueError:
        pass
    except:
        pytest.fail("registrate fail with unknown error")

    ctx_ = context.DistributedContextRegistry.context(ctx_name)
    assert ctx is ctx_

    context.DistributedContextRegistry.deregistrate(ctx_name)

    try:
        ctx_ = context.DistributedContextRegistry.context(ctx_name)
        pytest.fail(f"get deregistrated context {ctx_.name}")
    except:
        pass


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


@pytest.mark.timeout(10)
def test_context_target():
    context_name = "test_context_put"
    with mock.patch("torchgpipe.distributed.context.rpc", FakeRpc()):
        value_processor = MagicMock()
        ctx = context.DistributedContext(
            MICROBATCHES, context_name, value_processor
        )
        context.DistributedContextRegistry.registrate(ctx)
        value = torch.tensor([1.0])
        ctx.put_remote(context_name, value, True).result()
        value_processor.DTH.assert_called_with(value)

        ret = value_processor.DTH.return_value
        value_ = ctx.get_remote(True)
        value_processor.HTD.assert_called_with(ret)
        assert value_ is value_processor.HTD.return_value


@pytest.mark.timeout(10)
def test_context_forward_backward():
    context_name = "test_context_forward_backward"
    with mock.patch("torchgpipe.distributed.context.rpc", FakeRpc()):
        value_processor = MagicMock()
        ctx = context.DistributedContext(
            MICROBATCHES, context_name, value_processor
        )
        context.DistributedContextRegistry.registrate(ctx)

        # test forward, microbatch_id == 7
        value = torch.tensor([1.0])
        ctx.put_remote(context_name, value, False, backward=False, microbatch_id=7).result()
        value_processor.DTH.assert_called_with(value)

        ret = value_processor.DTH.return_value
        value_ = ctx.get_remote(False, backward=False, microbatch_id=7)
        value_processor.HTD.assert_called_with(ret)
        assert value_ is value_processor.HTD.return_value

        # test backward, microbatch_id == 3
        value = torch.tensor([1.0])
        ctx.put_remote(context_name, value, False, backward=True, microbatch_id=3).result()
        value_processor.DTH.assert_called_with(value)

        ret = value_processor.DTH.return_value
        value_ = ctx.get_remote(False, backward=True, microbatch_id=3)
        value_processor.HTD.assert_called_with(ret)
        assert value_ is value_processor.HTD.return_value
