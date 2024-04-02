import torch
from bert.modeling import *


class BERT_HEAD(nn.Module):
    def __init__(self) -> None:
        super(BERT_HEAD, self).__init__()
        self.layer6 = BertEmbeddings(30528, 1024, 512, 2, 0.1)

    def forward(self, out0, out1, out2):
        return self.layer6(out0, out1), out2


class TransformerBlock(nn.Module):

    def __init__(self) -> None:
        super(TransformerBlock, self).__init__()
        self.layer7 = BertSelfAttention(1024, 16, 0.1)
        self.layer8 = torch.nn.Linear(in_features=1024, out_features=1024, bias=True)
        self.layer9 = torch.nn.Dropout(p=0.1)
        self.layer11 = BertLayerNorm(1024)
        self.layer12 = LinearActivation(in_features=1024, out_features=4096, bias=True)
        self.layer13 = torch.nn.Linear(in_features=4096, out_features=1024, bias=True)
        self.layer14 = torch.nn.Dropout(p=0.1)
        self.layer16 = BertLayerNorm(1024)

    def forward(self, out6, out2):
        out7 = self.layer7(out6, out2)
        out8 = self.layer8(out7)
        out9 = self.layer9(out8)
        out9 = out9 + out6
        out11 = self.layer11(out9)
        out12 = self.layer12(out11)
        out13 = self.layer13(out12)
        out14 = self.layer14(out13)
        out14 = out14 + out11
        out16 = self.layer16(out14)
        return out16, out2


class BERT_TAIL(nn.Module):
    def __init__(self) -> None:
        super(BERT_TAIL, self).__init__()
        self.layer247 = LinearActivation(in_features=1024, out_features=1024, bias=True)
        self.layer248 = BertLayerNorm(1024)
        self.layer249 = torch.nn.Linear(in_features=1024, out_features=30528, bias=False)
        self.layer250 = BertAdd(30528)

    def forward(self, out246, _):
        out247 = self.layer247(out246)
        out248 = self.layer248(out247)
        out249 = self.layer249(out248)
        return self.layer250(out249)


def bert(num_classes: int, inplace: bool):
    return nn.Sequential(
        BERT_HEAD(),
        TransformerBlock(),
        TransformerBlock(),
        TransformerBlock(),
        TransformerBlock(),
        TransformerBlock(),
        TransformerBlock(),
        TransformerBlock(),
        TransformerBlock(),
        TransformerBlock(),
        TransformerBlock(),
        TransformerBlock(),
        TransformerBlock(),
        TransformerBlock(),
        TransformerBlock(),
        TransformerBlock(),
        TransformerBlock(),
        TransformerBlock(),
        TransformerBlock(),
        TransformerBlock(),
        TransformerBlock(),
        TransformerBlock(),
        TransformerBlock(),
        TransformerBlock(),
        TransformerBlock(),
        BERT_TAIL,
    )
