import torch
import torch.nn as nn
from torchvision.transforms import transforms as T
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from models.base_model_weights import Model_Weights
import torch.nn.functional as F


class STNMobileNetSmall(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.mobile_net = mobilenet_v3_small(weights=weights)
        self.mobile_net.classifier[3] = nn.Linear(self.mobile_net.classifier[3].in_features, num_classes)

        self.localization = mobilenet_v3_small(weights=weights)
        self.localization.classifier[3] = nn.Linear(1024, 3 * 2)

        # Initialize the weights/bias with identity transformation
        self.localization.classifier[3].weight.data.zero_()
        self.localization.classifier[3].bias.data.copy_(torch.eye(2, 3, dtype=torch.float).view(-1))

    # Spatial transformer network forward function
    def stn(self, x):
        theta = self.localization(x)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid, align_corners=False)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)
        x = self.mobile_net(x)
        return x


class STNMobileNetSmall_Weights(Model_Weights):
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
