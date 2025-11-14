'''
封装所有与PyTorch模型加载、数据预处理和单次预测相关的逻辑
'''
import torch
import numpy as np
import joblib
import json
import os
from CustomMultiChannelResNet18 import CustomMultiChannelResNet18

class TerrainPredictor:
    """
    地形预测器类。

    负责加载训练好的模型和预处理工具，并提供一个简单的接口  
    来对新的实时接触数据进行地形分类预测。
    """
    def __init__(self, model_path, scaler_path, mapping_path, max_length_path):
        """
        初始化预测器。

        Args:
            model_path (str): 训练好的PyTorch模型权重文件路径 (.pth)。
            scaler_path (str): 训练时使用的MinMaxScaler对象保存文件路径 (.pkl)。
            mapping_path (str): 地形名称与索引的映射文件路径 (.json)。
            max_length_path (str): 训练数据中的最大序列长度记录文件路径 (.json)。
        """
        # 1. 设置设备 (GPU或CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"预测器将使用 {self.device} 设备。")

        # 2. 加载地形映射 和 最大长度
        with open(mapping_path, 'r') as f:
            terrain_mapping = json.load(f)
            self.index_to_terrain = {int(k): v for k, v in terrain_mapping['index_to_terrain'].items()}
        
        with open(max_length_path, 'r') as f:
            self.max_length = json.load(f)['max_length']

        # 3. 加载数据归一化scaler
        self.scalers = joblib.load(scaler_path)

        # 4. 动态实例化并加载模型
        # 从加载的配置中获取模型构建所需的参数
        num_classes = len(self.index_to_terrain)
        num_channels = 2  # 两个通道：位移和力
        
        # 实例化模型结构
        self.model = CustomMultiChannelResNet18(num_channels=num_channels, num_classes=num_classes)
        # 加载训练好的权重
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        # 将模型移动到指定设备
        self.model.to(self.device)
        # 设置为评估模式，这会关闭Dropout和BatchNorm的训练行为
        self.model.eval()
        
        print("地形预测器初始化完成。")

    def _preprocess(self, contact_data):
        """
        对单次接触数据进行预处理，使其符合模型输入要求。
        这个过程必须与 `train.py` 中的预处理流程完全一致。

        Args:
            contact_data (list of tuples): 记录的接触数据，格式为 [(位移1, 力1), (位移2, 力2), ...]。

        Returns:
            torch.Tensor: 预处理后，可直接输入模型的张量。
        """
        # 1. 转换为Numpy数组，形状为 (序列长度, 通道数)
        data_np = np.array(contact_data, dtype=np.float32)

        # 2. 填充 (Padding) 到统一长度
        current_length = data_np.shape[0]
        if current_length < self.max_length:
            padding_size = self.max_length - current_length
            # 在序列的末尾填充0
            padded_data = np.pad(data_np, ((0, padding_size), (0, 0)), 'constant', constant_values=0)
        elif current_length > self.max_length:
            # 如果实时数据超过最大长度，则截断
            padded_data = data_np[:self.max_length, :]
        else:
            padded_data = data_np

        # 3. 交换维度以匹配模型输入 (样本数, 通道数, 序列长度)
        # 当前形状: (序列长度, 通道数) -> 目标形状: (通道数, 序列长度)
        transposed_data = padded_data.transpose(1, 0)

        # 4. 数据归一化 (Min-Max Scaling)
        # 使用与训练时相同的scaler对每个通道进行归一化
        scaled_data = np.zeros_like(transposed_data)
        for i in range(transposed_data.shape[0]): # 遍历每个通道
            # scaler期望的输入是 (样本数, 特征数)，这里我们只有一个样本（因为是实时获取），但特征是序列，所以我们将 (序列长度,) reshape为 (-1, 1) 再 transform
            scaled_data[i, :] = self.scalers[i].transform(transposed_data[i, :].reshape(-1, 1)).flatten()

        # 5. 转换为PyTorch张量并添加Batch维度
        # 当前形状: (通道数, 序列长度) -> 目标形状: (1, 通道数, 序列长度)
        final_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0)

        return final_tensor

    def predict(self, contact_data):
        """
        对一次完整的接触数据进行地形预测。

        Args:
            contact_data (list of tuples): 记录的接触数据。

        Returns:
            tuple: (预测的地形名称, 预测概率)
        """
        with torch.no_grad():
            # 1. 预处理数据
            input_tensor = self._preprocess(contact_data)
            # 2. 将张量移动到模型所在的设备
            input_tensor = input_tensor.to(self.device)

            # 3. 模型前向传播,得到原始输出 (logits)
            output = self.model(input_tensor)

            # 4. 应用softmax获取概率分布
            probabilities = torch.nn.functional.softmax(output, dim=1)

            # 5. 获取最可能的类别索引和对应的概率
            confidence, predicted_idx = torch.max(probabilities, 1)
            predicted_idx = predicted_idx.item()
            confidence = confidence.item()

            # 6. 将索引转换为地形名称
            predicted_terrain = self.index_to_terrain[predicted_idx]

            return predicted_terrain, confidence
