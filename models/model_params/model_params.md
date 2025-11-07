# 模型参数
本文档用于记录各个模型的参数信息

## CustomMultiChannelResNet18
### model = CustomMultiChannelResNet18(num_channels=2, num_classes=8)

模型参数摘要

- 总参数量:          1,581,896
- 可训练参数量:      1,581,896
- 不可训练参数量:    0


各层参数详情:

| 层名称 | 形状 | 参数量 |
|--------|------|--------|
| conv1.weight | [64, 2, 7] | 896 |
| conv1.bias | [64] | 64 |
| bn1.weight | [64] | 64 |
| bn1.bias | [64] | 64 |
| layer1.0.weight | [64, 64, 3] | 12,288 |
| layer1.1.weight | [64] | 64 |
| layer1.1.bias | [64] | 64 |
| layer1.3.weight | [64, 64, 3] | 12,288 |
| layer1.4.weight | [64] | 64 |
| layer1.4.bias | [64] | 64 |
| layer2.0.weight | [128, 64, 3] | 24,576 |
| layer2.1.weight | [128] | 128 |
| layer2.1.bias | [128] | 128 |
| layer2.3.weight | [128, 128, 3] | 49,152 |
| layer2.4.weight | [128] | 128 |
| layer2.4.bias | [128] | 128 |
| layer3.0.weight | [256, 128, 3] | 98,304 |
| layer3.1.weight | [256] | 256 |
| layer3.1.bias | [256] | 256 |
| layer3.3.weight | [256, 256, 3] | 196,608 |
| layer3.4.weight | [256] | 256 |
| layer3.4.bias | [256] | 256 |
| layer4.0.weight | [512, 256, 3] | 393,216 |
| layer4.1.weight | [512] | 512 |
| layer4.1.bias | [512] | 512 |
| layer4.3.weight | [512, 512, 3] | 786,432 |
| layer4.4.weight | [512] | 512 |
| layer4.4.bias | [512] | 512 |
| fc.weight | [8, 512] | 4,096 |
| fc.bias | [8] | 8 |


**模型总参数量: 1,581,896**