# models文件夹命名规范
默认环境为 python3.14.0，环境名称为 `HapticResNet18`
## 缩写含义
**f2**：输入数据的预处理方式。详见输入数据命名规范

**Adam、SGD**：优化器种类
**trueResNet**：优化后的ResNet模型架构

**bmp**：best model params
**bmw**：best model weights
**xxx epochs**: xxx epochs

**maxAB**：输入的训练数据分割方法为，找到峰值，前后固定长度。（仅在train中临时处理，不对原始数据修改）
**stateMax**：

**allscaler**：单一通道全局归一化
**noVal**：显示取消验证集（train_split=None）

**noFullScalerTrain**：不进行最终全量训练

**python3.8.10**：ubuntu系统中ROS2的默认python环境版本。基于此python版本在Windows11中搭建的虚拟环境，环境名称为 `HapticResNet18-python3.8`  
**xxx**（阿拉伯数字）：相同条件的重复训练测试

**linux**：在xiejiapeng的linux系统中训练




