from torchgpipe import microbatch
from torchgpipe.gpipe import TensorOrTensors


class DistributedBatch(microbatch.Batch):

    def requires_grad_trait(self) -> TensorOrTensors:
        return tuple(t for t in self if t.requires_grad)

    def grad_trait(self) -> TensorOrTensors:
        return tuple(t.grad for t in self if t.requires_grad)
