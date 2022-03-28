from torch import nn
from torch.nn import functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, config):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=config['res_kernel'], stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=config['res_kernel'], stride=1, padding=1)
            self.shortcut = nn.Identity()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=config['res_kernel'], stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

    @property
    def downsample(self):
        return self.in_channels != self.out_channels

    def forward(self, x):
        shortcut = self.shortcut(x)
        
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.leaky_relu(x)

        x += shortcut
        x = F.leaky_relu(x)
        return x