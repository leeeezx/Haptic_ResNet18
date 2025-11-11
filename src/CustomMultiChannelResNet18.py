import torch
import torch.nn as nn
import pandas as pd
import numpy as np



# 1D-ResNet的基础残差块
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 块内第一个卷积层
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # 块内第二个卷积层
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 下采样层，用于匹配残差连接的维度
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 如果维度不匹配，使用下采样层处理identity
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接
        out += identity
        out = self.relu(out)

        return out


# 自定义多通道ResNet-18-1D模型
class CustomMultiChannelResNet18(nn.Module):
    def __init__(self, num_channels, num_classes, block=BasicBlock, layers=[2, 2, 2, 2]):
        '''
        Args:
            num_channels (int): 输入数据的通道数。
            num_classes (int): 分类任务的类别数
            block: 残差块类型
            layers: 每个ResNet层包含的残差块数量
        '''
        super(CustomMultiChannelResNet18, self).__init__()
        self.in_channels = 64

        # 输入层。一维卷积 -> 批归一化 -> ReLU激活 -> 最大池化
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=self.in_channels, kernel_size=7, stride=2, padding=3, bias=False) 
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet 模块
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 全局平均池化层
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # 全连接层
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        '''
        构建一个 ResNet 层

        Args:
            block: 残差块类型
            out_channels (int): 输出通道数
            num_blocks (int): 残差块数量
            stride (int): 第一个残差块的步长，用于控制下采样
        '''
        downsample = None
        # 当需要下采样或者通道数不匹配时，定义downsample层
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        layers = []
        # 添加第一个残差块，它可能包含下采样
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        # 添加余下的残差块
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

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

