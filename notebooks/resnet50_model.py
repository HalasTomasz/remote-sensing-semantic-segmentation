"""Implemention of ResNet50 based on original paper with my own decoder"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, number_of_convolutions_list[0], stride=1)
        self.layer2 = self._make_layer(block, 128, number_of_convolutions_list[1], stride=2)
        self.layer3 = self._make_layer(block, 256, number_of_convolutions_list[2], stride=2)
        self.layer4 = self._make_layer(block, 512, number_of_convolutions_list[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Adaptive pooling to handle different input sizes
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=1, padding=2, output_padding=0),
            nn.Tanh()
        )

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
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        x = self.decoder(x)

        return x

# def test():

#     num_classes = 2
#     BATCH_SIZE = 4

#     model = ResNet50(ResNet_Block, [3, 4, 6, 3], num_classes)
#     x = torch.randn(BATCH_SIZE, 3, 512, 512)
#     y = model(x).to('cuda')
#     print(y.size())
#     #assert y.size() == torch.Size([BATCH_SIZE, 2])


# if __name__ == "__main__":
#     test()
