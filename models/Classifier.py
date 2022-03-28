import torch
from torch import nn 

class Classifier(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(config['classifier_in'], config['n_classes'])
        
    def forward(self, x):
        x = self.pooling(x)
        x = torch.squeeze(x)
        return self.fc(x)
