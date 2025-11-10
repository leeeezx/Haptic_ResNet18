import os
import json
import re
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def preprocess_and_save_data(data_root="D:/Dataset/", 
                             save_dir="d:/CodeProject/haptic_ResNet/data",
                             # scalers_path: str = None,  # 注释掉：改为内部生成scalers
                             mode: str = "both",
                             filter_type: str = "suffix",
                             filter_pattern: str = "-p2"):
    """
    从原始数据预处理并保存测试数据，文件形式为.pt，只供评估时读取使用。  
    原理与train.py中的数据预处理类似，但这里直接保存为.pt文件，方便评估脚本读取。
    
    Args:
        data_root: 原始数据根目录
        save_dir: 预处理后数据保存目录
        # scalers_path: 已保存的scalers文件路径，用于数据归一化  # 注释掉
        mode: 数据生成模式，可选值: "train"(只训练), "test"(只测试), "both"(训练+测试)
        filter_type: 文件夹过滤类型，可选值: "prefix"(前缀匹配), "suffix"(后缀匹配), "contains"(包含匹配)
        filter_pattern: 过滤模式字符串，如 "t-" 或 "-p2"
    """
    # 验证mode参数
    if mode not in ["train", "test", "both"]:
        raise ValueError("mode参数必须是 'train', 'test' 或 'both' 之一")
    
    # 验证filter_type参数
    if filter_type not in ["prefix", "suffix", "contains"]:
        raise ValueError("filter_type参数必须是 'prefix', 'suffix' 或 'contains' 之一")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 定义数据路径和类别
    # ==========================================
    all_folders = os.listdir(data_root)
    TERRAINS = sorted([
        folder for folder in all_folders 
        if os.path.isdir(os.path.join(data_root, folder)) and re.match(r'^\d{2}$', folder)
    ])
    NUM_CLASSES = len(TERRAINS)
    
    print(f"发现 {NUM_CLASSES} 种地形: {TERRAINS}")
    print(f"过滤方式: {filter_type}, 过滤模式: '{filter_pattern}'")
    
    # 2. 从CSV文件加载数据和标签
    # ==========================================
    all_data = []
    all_labels = []
    
    # 创建地形名称到连续索引的映射
    terrain_to_index = {terrain: idx for idx, terrain in enumerate(TERRAINS)}
    index_to_terrain = {idx: terrain for terrain, idx in terrain_to_index.items()}
    
    print(f"地形映射关系: {terrain_to_index}")
    
    # 定义文件夹过滤函数
    def match_folder(folder_name, pattern, match_type):
        """根据匹配类型判断文件夹是否符合条件"""
        if match_type == "prefix":
            return folder_name.startswith(pattern)
        elif match_type == "suffix":
            return folder_name.endswith(pattern) or pattern in folder_name
        elif match_type == "contains":
            return pattern in folder_name
        return False
    
    # 遍历每一种地形
    for terrain_name in TERRAINS:
        label_index = terrain_to_index[terrain_name]
        
        terrain_path = os.path.join(data_root, terrain_name)
        subfolders = os.listdir(terrain_path)
        
        # 根据filter_type和filter_pattern过滤子文件夹
        valid_subfolders = [
            subfolder for subfolder in subfolders 
            if os.path.isdir(os.path.join(terrain_path, subfolder)) and 
            match_folder(subfolder, filter_pattern, filter_type)
        ]
        
        print(f"地形 {terrain_name}: 发现 {len(valid_subfolders)} 个匹配的子文件夹")
        
        for subfolder in valid_subfolders:
            subfolder_path = os.path.join(terrain_path, subfolder)
            # 获取该地形文件夹下所有的csv文件
            csv_files = [f for f in os.listdir(subfolder_path) if f.endswith('.csv')]
    
            # 遍历并读取每一个csv文件
            for csv_file in csv_files:
                csv_path = os.path.join(subfolder_path, csv_file)
                try:
                    # 读取CSV文件
                    df = pd.read_csv(csv_path)
                    # 根据列名 '/realtime_robot_poseAndextTau/data.2' 和 '/realtime_robot_poseAndextTau/data.8' 提取数据
                    sample_data = df[['/realtime_robot_poseAndextTau/data.2', '/realtime_robot_poseAndextTau/data.8']].values
                    all_data.append(sample_data)
                    all_labels.append(label_index)
                except Exception as e:
                    print(f"读取文件 {csv_path} 时出错: {e}")
    
    print(f"数据加载完成!共加载了 {len(all_data)} 个样本。")
    
    # 保存映射关系供后续使用
    with open(os.path.join(save_dir, 'terrain_mapping.json'), 'w') as f:
        json.dump({'terrain_to_index': terrain_to_index, 
                   'index_to_terrain': index_to_terrain}, f, indent=2)
    
    # 2.5. 对所有样本进行长度统一(Padding)
    # ==========================================
    # 计算所有样本中的最大序列长度
    max_length = 0
    for sample in all_data:
        if len(sample) > max_length:
            max_length = len(sample)
    print(f"数据中最长的序列长度为: {max_length}")
    
    # 现在,对所有比 max_length 短的样本进行填充
    padded_data = []
    for sample in all_data:
        # sample 的形状是 (序列长度, 2)
        len_sample = len(sample)
        if len_sample < max_length:
            # 计算需要填充的长度
            padding_size = max_length - len_sample
            # 使用 numpy.pad 进行填充
            padded_sample = np.pad(sample, ((0, padding_size), (0, 0)), 'constant', constant_values=0)
        else:
            padded_sample = sample
        padded_data.append(padded_sample)
    
    print(f"Padding完成,所有样本长度已统一为: {max_length}")
    
    # 3. 将数据列表转换为一个大的Numpy数组
    # ==========================================
    X = np.array(padded_data)
    y = np.array(all_labels)
    
    # 转置数组以匹配模型输入格式
    X = X.transpose(0, 2, 1)
    
    print(f"原始数据形状 (X): {X.shape}")
    print(f"标签数据形状 (y): {y.shape}")
    
    # 4. 根据模式进行数据处理
    # ==========================================
    if mode == "test":
        # 只测试模式：不进行数据划分，所有数据作为测试集
        X_test = X
        y_test = y
        print(f"测试集大小: {X_test.shape}")
        
        # 5. 数据归一化 (Min-Max Scaling)
        # ==========================================
        print("开始归一化处理...")
        # scalers = joblib.load(scalers_path)  # 注释掉：改为内部生成
        # 生成scalers
        scalers = []
        for i in range(X_test.shape[1]):
            scaler = MinMaxScaler()
            scaler.fit(X_test[:, i, :])
            scalers.append(scaler)
            X_test[:, i, :] = scaler.transform(X_test[:, i, :])
        print("归一化完成")
        
        # 6. 将Numpy数组转换为PyTorch张量
        # ==========================================
        Test_data_final_tensor = torch.tensor(X_test, dtype=torch.float32)
        Test_data_final_label_tensor = torch.tensor(y_test, dtype=torch.long)
        
        # 保存测试数据集
        torch.save(Test_data_final_tensor, os.path.join(save_dir, 'X_test_independent.pt'))
        torch.save(Test_data_final_label_tensor, os.path.join(save_dir, 'y_test_independent.pt'))
        
        print(f"\n预处理的测试数据已保存到: {save_dir}")
        print("- 测试数据: X_test.pt, y_test.pt")
        
    elif mode == "train":
        # 只训练模式：不进行数据划分，所有数据作为训练集
        X_train = X
        y_train = y
        print(f"训练集大小: {X_train.shape}")
        
        # 5. 数据归一化 (Min-Max Scaling)
        # ==========================================
        print("开始归一化处理...")
        # scalers = joblib.load(scalers_path)  # 注释掉：改为内部生成
        # 生成scalers
        scalers = []
        for i in range(X_train.shape[1]):
            scaler = MinMaxScaler()
            scaler.fit(X_train[:, i, :])
            scalers.append(scaler)
            X_train[:, i, :] = scaler.transform(X_train[:, i, :])
        print("归一化完成")
        
        # 6. 将Numpy数组转换为PyTorch张量
        # ==========================================
        Train_data_final_tensor = torch.tensor(X_train, dtype=torch.float32)
        Train_data_final_label_tensor = torch.tensor(y_train, dtype=torch.long)
        
        # 保存训练数据集
        torch.save(Train_data_final_tensor, os.path.join(save_dir, 'X_train.pt'))
        torch.save(Train_data_final_label_tensor, os.path.join(save_dir, 'y_train.pt'))
        
        print(f"\n预处理的训练数据已保存到: {save_dir}")
        print("- 训练数据: X_train.pt, y_train.pt")
        
    else:  # mode == "both"
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        
        print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
        
        # 5. 数据归一化 (Min-Max Scaling)
        # ==========================================
        print("开始归一化处理...")
        # scalers = joblib.load(scalers_path)  # 注释掉：改为内部生成
        # 生成scalers（基于训练集）
        scalers = []
        for i in range(X_train.shape[1]):
            scaler = MinMaxScaler()
            scaler.fit(X_train[:, i, :])
            scalers.append(scaler)
            X_train[:, i, :] = scaler.transform(X_train[:, i, :])
            X_test[:, i, :] = scaler.transform(X_test[:, i, :])
        print("归一化完成")
        
        # 6. 将Numpy数组转换为PyTorch张量
        # ==========================================
        Train_data_final_tensor = torch.tensor(X_train, dtype=torch.float32)
        Train_data_final_label_tensor = torch.tensor(y_train, dtype=torch.long)
        Test_data_final_tensor = torch.tensor(X_test, dtype=torch.float32)
        Test_data_final_label_tensor = torch.tensor(y_test, dtype=torch.long)
        
        # 保存所有数据集
        torch.save(Train_data_final_tensor, os.path.join(save_dir, 'X_train.pt'))
        torch.save(Train_data_final_label_tensor, os.path.join(save_dir, 'y_train.pt'))
        torch.save(Test_data_final_tensor, os.path.join(save_dir, 'X_test.pt'))
        torch.save(Test_data_final_label_tensor, os.path.join(save_dir, 'y_test.pt'))
        
        print(f"\n所有预处理数据已保存到: {save_dir}")
        print("- 训练数据: X_train.pt, y_train.pt")
        print("- 测试数据: X_test.pt, y_test.pt")
    
    # 保存最大序列长度，用于后续可能的新数据预处理
    with open(os.path.join(save_dir, 'max_length.json'), 'w') as f:
        json.dump({'max_length': max_length}, f)
    
    print("- 地形映射: terrain_mapping.json")
    print("- 最大序列长度: max_length.json")

if __name__ == "__main__":
    # 执行预处理
    # 示例用法：
    
    # # 1. 生成测试数据（t-开头的文件夹）
    # preprocess_and_save_data(
    #     # scalers_path="d:/CodeProject/haptic_ResNet/models/scalers/scalers-100epochs.pkl",  # 注释掉
    #     mode="test",
    #     filter_type="prefix",
    #     filter_pattern="t-"
    # )
    
    # 2. 生成训练数据（-p2后缀的文件夹）
    preprocess_and_save_data(
        # scalers_path="d:/CodeProject/haptic_ResNet/models/scalers/scalers-100epochs.pkl",  # 注释掉
        mode="train",
        filter_type="suffix",
        filter_pattern="-p2"
    )
    
    # 3. 生成训练+测试数据（包含特定字符的文件夹）
    # preprocess_and_save_data(
    #     # scalers_path="d:/CodeProject/haptic_ResNet/models/scalers/scalers-100epochs.pkl",  # 注释掉
    #     mode="both",
    #     filter_type="contains",
    #     filter_pattern="p2"
    # )