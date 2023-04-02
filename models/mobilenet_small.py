import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.transforms import transforms as T
from models.base_model_weights import Model_Weights


class MobileNetSmallWrapper(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        if pretrained:
            weights = MobileNet_V3_Small_Weights.DEFAULT
            self.mobile_net = mobilenet_v3_small(weights=weights)
        else:
            self.mobile_net = mobilenet_v3_small()
        num_features = self.mobile_net.classifier[3].in_features
        self.mobile_net.classifier[3] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.mobile_net(x)
        return x


class MobileNetSmallWrapper_Weights(Model_Weights):
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
