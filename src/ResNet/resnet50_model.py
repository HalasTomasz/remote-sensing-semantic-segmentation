"""Implemention of ResNet50 based on original paper with my own decoder"""

import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConvLayerResNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvLayerResNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ResNet_Block(nn.Module):
    """Resnet Block module"""

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1) -> None:

        super(ResNet_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.shortcut = None
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):

        identity_x = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.shortcut is not None:
            x += self.shortcut(identity_x)

        x = F.relu(x)

        return x


class ResNet50(nn.Module):
    """Resnet50 main class"""

    def __init__(self, block, number_of_convolutions_list, num_classes) -> None:

        super(ResNet50, self).__init__()
        self.in_channels = 64
        channel_list = [64, 128, 256, 512]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, number_of_convolutions_list[0], stride=1)
        self.layer2 = self._make_layer(block, 128, number_of_convolutions_list[1], stride=2)
        self.layer3 = self._make_layer(block, 256, number_of_convolutions_list[2], stride=2)
        self.layer4 = self._make_layer(block, 512, number_of_convolutions_list[3], stride=2)

        self.ups = nn.ModuleList()

        for feature in reversed(channel_list):
            self.ups.append(nn.ConvTranspose2d(feature * 4 * 2, feature * 4, kernel_size=2, stride=2))
            self.ups.append(DoubleConvLayerResNet(feature * 4 * 2, feature * 4))

        self.bottleneck = DoubleConvLayerResNet(channel_list[-1] * 4, channel_list[-1] * 4 * 2)

        self.tmp = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def _make_layer(self, block, out_channels, blocks, stride):

        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool(x)

        skips = []
        x = self.layer1(x)
        skips.append(x)
        x = self.layer2(x)
        skips.append(x)
        x = self.layer3(x)
        skips.append(x)
        x = self.layer4(x)
        skips.append(x)
        skips.reverse()
        x = self.bottleneck(x)

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skips[idx // 2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        x = self.tmp(x)
        x = self.final_conv(x)

        return x
