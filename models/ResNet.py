from turtle import forward
from torch import nn 
from models.ResLayer import ResLayer
from models.Classifier import Classifier

class ResNet(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.layer0 = nn.Sequential(
                nn.Conv2d(in_channels=config['layer0_in'], out_channels=config['layer1_in'], \
                        kernel_size=config['layer0_kernel'], stride=config['layer0_stride'], padding=config['layer0_padding']),
                nn.MaxPool2d(kernel_size=config['layer0_kernel'], stride=config['layer0_stride'], padding=config['layer0_padding']),
                nn.BatchNorm2d(config['layer1_in']),
                nn.ReLU()
                )
        self.layer1 = ResLayer(in_channels=config['layer1_in'], out_channels=config['layer2_in'], config=config)
        self.layer2 = ResLayer(in_channels=config['layer2_in'], out_channels=config['layer3_in'], config=config)
        self.layer3 = ResLayer(in_channels=config['layer3_in'], out_channels=config['layer4_in'], config=config)
        self.layer4 = ResLayer(in_channels=config['layer4_in'], out_channels=config['layer5_in'], config=config)
        self.layer5 = ResLayer(in_channels=config['layer5_in'], out_channels=config['classifier_in'], config=config)
        self.classifier = Classifier(config)
        self.dropout = nn.Dropout(config['dropout'])
    
    def forward(self, x):
        x = self.layer0(x)          # batch x 16 x 32 x 32
        x = self.layer1(x)          # batch x 16 x 32 x 32
        x = self.dropout(x)
        x = self.layer2(x)          # batch x 32 x 16 x 16
        x = self.dropout(x)
        x = self.layer3(x)          # batch x 64 x 8 x 8
        x = self.dropout(x)
        x = self.layer4(x)          # batch x 128 x 4 x 4
        x = self.dropout(x)
        x = self.layer5(x)          # batch x 256 x 2 x 2
        x = self.dropout(x)
        x = self.classifier(x)         # batch x n_classes
        
        return x