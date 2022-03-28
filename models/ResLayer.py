from torch import nn

from models.ResBlock import ResBlock

class ResLayer(nn.Module):
    def __init__(self, in_channels, out_channels, config) -> None:
        super().__init__()
        self.resblock1 = ResBlock(in_channels, out_channels, config)
        self.resblock2 = ResBlock(out_channels, out_channels, config)

    def forward(self, x):
        x = self.resblock1(x)
        x = self.resblock2(x)
        return x