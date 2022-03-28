import torch
from torch import nn
from torchvision.models import resnet50

class PTResNet(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.resnet = resnet50(pretrained=True)
        self.freeze_layers()
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()
        self.resnet.fc =nn.Sequential(nn.Linear(in_features=2048, out_features=config['n_classes']))#, nn.ReLU(), nn.Linear(in_features=512, out_features=10))

    def freeze_layers(self):
        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.resnet(x)
        return x
