import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import transforms as T
from models.base_model_weights import Model_Weights


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


class ResNet50Wrapper_Weights(Model_Weights):
    train_preprocess = T.Compose([
        T.Resize([232, ]),
        T.CenterCrop(176),
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    inference_preprocess = T.Compose([
        T.Resize([232, ]),
        T.CenterCrop(224),
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
