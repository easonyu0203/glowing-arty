import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50Wrapper(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        if pretrained:
            weights = ResNet50_Weights.DEFAULT
            self.resnet = resnet50(weights=weights)
        else:
            self.resnet = resnet50()
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x
