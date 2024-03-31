import torch

def vgg16(num_classes: int, inplace: bool) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.ReLU(inplace=inplace),
        torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.ReLU(inplace=inplace),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.ReLU(inplace=inplace),
        torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.ReLU(inplace=inplace),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.ReLU(inplace=inplace),
        torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.ReLU(inplace=inplace),
        torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.ReLU(inplace=inplace),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.ReLU(inplace=inplace),
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.ReLU(inplace=inplace),
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.ReLU(inplace=inplace),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.ReLU(inplace=inplace),
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.ReLU(inplace=inplace),
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.ReLU(inplace=inplace),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        torch.nn.Sequential (
            torch.nn.AdaptiveAvgPool2d((7, 7)),
            torch.nn.Flatten(),
        ),
        torch.nn.Linear(in_features=25088, out_features=4096, bias=True),
        torch.nn.ReLU(inplace=inplace),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
        torch.nn.ReLU(inplace=inplace),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(in_features=4096, out_features=num_classes, bias=True),
    )