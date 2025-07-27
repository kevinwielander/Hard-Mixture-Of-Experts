import torch.nn as nn
import torch
from torchvision import models


class Stem(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class ResNetBackbone(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.features = nn.Sequential(*list(resnet.children())[4:-1])  # Remove stem and fc
        self.out_dim = out_dim

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


class Predictor(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class BasicNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.stem = Stem()
        self.backbone = ResNetBackbone()
        self.predictor = Predictor(self.backbone.out_dim, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.backbone(x)
        x = self.predictor(x)
        return x
