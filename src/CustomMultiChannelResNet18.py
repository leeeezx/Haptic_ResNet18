import torch
import torch.nn as nn
import pandas as pd
import numpy as np



# 自定义多通道ResNet-18-1D模型
class CustomMultiChannelResNet18(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(CustomMultiChannelResNet18, self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes

        # 输入层
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet 模块
        self.layer1 = self.make_resnet_layer(in_channels=64, out_channels=64, num_blocks=2)
        self.layer2 = self.make_resnet_layer(in_channels=64, out_channels=128, num_blocks=2, stride=2)
        self.layer3 = self.make_resnet_layer(in_channels=128, out_channels=256, num_blocks=2, stride=2)
        self.layer4 = self.make_resnet_layer(in_channels=256, out_channels=512, num_blocks=2, stride=2)

        # 全局平均池化层
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # 全连接层
        self.fc = nn.Linear(512, num_classes)

    def make_resnet_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(1, num_blocks):
            layers.append(nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 输入数据 x 的形状应该是 [batch_size, num_channels, sequence_length]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

