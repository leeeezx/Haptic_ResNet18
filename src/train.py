import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from skorch import NeuralNetClassifier
from skorch.callbacks import ProgressBar
import joblib
import os
import re
import json

from CustomMultiChannelResNet18 import CustomMultiChannelResNet18
from evaluation_visualizer import (
    plot_confusion_matrix,
    generate_and_save_classification_report,
    plot_class_accuracies,
    plot_roc_curves,
    plot_prediction_samples,
    save_evaluation_results
)

# 全局配置
DATA_ROOT = "D:/Dataset/"
WEIGHTS_DIR = 'd:/CodeProject/haptic_ResNet/models/weights'
HYPERPARAMS_DIR = 'd:/CodeProject/haptic_ResNet/models/hyperparams'
SCALERS_DIR = 'd:/CodeProject/haptic_ResNet/models/scalers'
DATA_DIR = 'd:/CodeProject/haptic_ResNet/data'
MAPPING_FILE = os.path.join(DATA_DIR, 'terrain_mapping.json')
RESULTS_DIR = 'd:/CodeProject/haptic_ResNet/results/test_eval_resnet18_python3.8.10' 

def load_data(data_root):
    """
    从指定的根目录加载和准备地形数据。

    Args:
        data_root (str): 原始数据根目录

    Returns:
        all_data (list of np.ndarray): 加载的所有样本数据
        all_labels (list of int): 对应的标签列表
        terrain_to_index (dict): 地形名称到索引的映射
        index_to_terrain (dict): 索引到地形名称的映射
        num_classes (int): 地形类别的数量
    """
    # 1. 定义数据路径和类别
    # ==========================================
    all_folders = os.listdir(data_root) # 获取根目录下的所有文件夹名称
    terrains = sorted([
        folder for folder in all_folders 
        if os.path.isdir(os.path.join(data_root, folder)) and re.match(r'^\d{2}$', folder)
    ])                                  
    num_classes = len(terrains) # 地形类别的数量
    print(f"发现 {num_classes} 种地形: {terrains}")

    # 2. 从CSV文件加载数据和标签
    # ==========================================
    all_data = []  # 用来存放所有样本的数据
    all_labels = [] # 用来存放所有样本的标签

    # 创建地形名称到连续索引的映射
    terrain_to_index = {terrain: idx for idx, terrain in enumerate(terrains)}
    index_to_terrain = {idx: terrain for terrain, idx in terrain_to_index.items()}
    print(f"地形映射关系: {terrain_to_index}")

    # 遍历每一种地形
    for terrain_name in terrains:
        label_index = terrain_to_index[terrain_name]  # 使用映射后的连续索引
        
        terrain_path = os.path.join(data_root, terrain_name)
        subfolders = os.listdir(terrain_path) # 获取一级目录地形文件夹下的所有子文件夹
        # 过滤二级目录:只保留包含 '-p2' 后缀的文件夹
        valid_subfolders = [
            subfolder for subfolder in subfolders 
            if os.path.isdir(os.path.join(terrain_path, subfolder)) and '-p2' in subfolder
        ]
        
        print(f"地形 {terrain_name}: 发现 {len(valid_subfolders)} 个有效子文件夹")
        
        for subfolder in valid_subfolders:
            subfolder_path = os.path.join(terrain_path, subfolder)
            # 获取该地形文件夹下所有的csv文件
            csv_files = [f for f in os.listdir(subfolder_path) if f.endswith('.csv')]

            # 遍历并读取每一个csv文件
            for csv_file in csv_files:
                file_path = os.path.join(subfolder_path, csv_file)

                # 使用pandas读取csv
                df = pd.read_csv(file_path)
                
                # ！！！！！！ 假设您的列名是 'displacement' 和 'force'，请根据实际情况修改 ！！！！！！
                # .values 会将数据框转换为Numpy数组
                # 我们希望得到的数组形状是 (序列长度, 2)
                sample_data = df[['/realtime_robot_poseAndextTau/data.2', 
                                  '/realtime_robot_poseAndextTau/data.8']].values 
                
                all_data.append(sample_data)
                all_labels.append(label_index)

    print(f"数据加载完成!共加载了 {len(all_data)} 个样本。")
    return all_data, all_labels, terrain_to_index, index_to_terrain, num_classes

def preprocess_data(all_data, all_labels):
    """
    对加载的数据进行预处理，包括填充、格式转换、划分和归一化。

    Args:   
        all_data (list of np.ndarray): 加载的所有样本数据
        all_labels (list of int): 对应的标签列表
    
    Returns:
        X_train (np.ndarray): 训练集数据
        X_test (np.ndarray): 测试集数据
        y_train (np.ndarray): 训练集标签
        y_test (np.ndarray): 测试集标签
        scalers (list of MinMaxScaler): 用于归一化的scaler列表
    """
    # 2.5. 对所有样本进行长度统一(Padding)
    # ==========================================
    # 计算所有样本中的最大序列长度
    max_length = max(len(sample) for sample in all_data)
    print(f"数据中最长的序列长度为: {max_length}")

    max_length_path = os.path.join(DATA_DIR, 'max_length.json')
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(max_length_path, 'w') as f:
        json.dump({'max_length': max_length}, f)
    print(f"已将max_length保存到: {max_length_path}")

    # 现在,对所有比 max_length 短的样本进行填充
    padded_data = []
    for sample in all_data:
        # sample 的形状是 (序列长度, 2)
        len_sample = len(sample)
        if len_sample < max_length:
            # 计算需要填充的长度
            padding_size = max_length - len_sample
            # 使用 numpy.pad 进行填充
            # ((0, padding_size), (0, 0)) 的意思是:
            # - 在第0维(序列长度维度)上,前面不填充(0),后面填充(padding_size)
            # - 在第1维(通道维度)上,前后都不填充(0)
            padded_sample = np.pad(sample, ((0, padding_size), (0, 0)), 'constant', constant_values=0)
            padded_data.append(padded_sample)
        else:
            padded_data.append(sample)
    print(f"Padding完成,所有样本长度已统一为: {max_length}")

    # 3. 将数据列表转换为一个大的Numpy数组
    # ==========================================
    # np.array(padded_data) 会创建一个形状为 (样本数, 序列长度, 2) 的数组
    # 现在所有样本的序列长度都是 max_length
    X = np.array(padded_data)
    y = np.array(all_labels)

    # 我们的模型需要输入的形状是 (样本数, 通道数, 序列长度)
    # 所以需要交换最后两个维度 (600, 2) -> (2, 600)
    # transpose(0, 2, 1) 的意思是：保持第0维（样本数）不变，将第2维（通道数）和第1维（序列长度）交换
    X = X.transpose(0, 2, 1)

    print(f"原始数据形状 (X): {X.shape}") # 应该打印 (样本数, 2, max_length)
    print(f"标签数据形状 (y): {y.shape}")   # 应该打印 (样本数,)

    # 4. 划分训练集和测试集
    # ==========================================
    # 这是评估模型性能的关键一步。我们用80%的数据训练，20%的数据测试。
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,    # 20%作为测试集
        random_state=42,  # 保证每次划分结果都一样，方便复现
        stratify=y        # 确保训练集和测试集中各类别的比例与原始数据一致
    )
    print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

    # 5. 数据归一化 (Min-Max Scaling)
    # ==========================================
    # 归一化可以加速模型训练，提升性能
    # 重要原则：只能在训练集上 `fit`（学习缩放规则），然后用这个规则去 `transform`（应用规则）训练集和测试集
    # 这样可以防止测试集的信息泄露给训练过程
    print("开始归一化处理...")
    scalers = []  
    # 我们需要对每个通道分别进行归一化
    for i in range(X_train.shape[1]):  # X_train.shape[1] 就是通道数，这里是 2
        scaler = MinMaxScaler()
        X_train[:, i, :] = scaler.fit_transform(X_train[:, i, :]) # scaler学习训练集第i个通道的缩放规则
        X_test[:, i, :] = scaler.transform(X_test[:, i, :]) # scaler应用在测试集第i个通道上
        scalers.append(scaler)
    print("归一化完成")

    # os.makedirs(SCALERS_DIR, exist_ok=True)
    # scaler_path = os.path.join(SCALERS_DIR, 'scalers-100epochs-train-python3.8.10.pkl')
    # joblib.dump(scalers, scaler_path)
    
    return X_train, X_test, y_train, y_test, scalers

def train_and_evaluate(X_train, y_train, X_test, y_test, num_classes, terrain_mapping):
    """
    执行模型训练、超参数搜索和评估。
    
    Args:
        terrain_mapping (dict): 包含 'terrain_to_index' 和 'index_to_terrain' 的字典
    
    Returns:
        best_model: 训练好的最佳模型
        evaluation_results (dict): 评估结果字典
    """
    # 6. 将Numpy数组转换为PyTorch张量
    # ==========================================
    # 这是将数据喂给PyTorch模型前的最后一步
    Train_data_final_tensor = torch.tensor(X_train, dtype=torch.float32)
    Train_data_final_label_tensor = torch.tensor(y_train, dtype=torch.long)

    # 为了后续评估，我们也可以转换测试集
    Test_data_final_tensor = torch.tensor(X_test, dtype=torch.float32)
    Test_data_final_label_tensor = torch.tensor(y_test, dtype=torch.long)

    # 打印最终数据统计信息，确认无误
    print("\n--- 最终数据准备完毕 ---")
    print(f"训练数据张量形状: {Train_data_final_tensor.shape}")
    print(f"训练标签张量形状: {Train_data_final_label_tensor.shape}")
    print(f"测试数据张量形状: {Test_data_final_tensor.shape}")
    print(f"测试标签张量形状: {Test_data_final_label_tensor.shape}")

    # 检查CUDA是否可用
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("使用GPU进行训练")
    else:
        device = torch.device("cpu")
        print("使用CPU进行训练")

    # 定义超参数网格
    param_grid = {
        # 'lr': [0.1, 0.01, 0.001],
        'lr': [0.001, 0.0001, 0.00001],
        'batch_size': [16, 32], # 目前笔记本测试时，发现batch=64会显存溢出，直接卡住
        'max_epochs': [100],
    }

    # 创建Skorch神经网络分类器
    net = NeuralNetClassifier(
        CustomMultiChannelResNet18,  # 直接传入类,不要实例化
        module__num_channels=X_train.shape[1],  # 使用module__前缀传递模型参数
        module__num_classes=num_classes,
        max_epochs=100,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        callbacks=[
            ProgressBar(),  # 添加进度条
        ],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        iterator_train__shuffle=True,
        iterator_valid__shuffle=False,
    )

    # 使用Skorch的GridSearchCV执行超参数搜索
    print("开始超参数搜索...")
    grid_search = GridSearchCV(net, param_grid, cv=5, scoring='accuracy', n_jobs=1, verbose=3) # cv代表 进行x折交叉验证
    grid_search.fit(Train_data_final_tensor, y=Train_data_final_label_tensor)

    # 输出最佳超参数组合和验证集上的性能
    print("Best params:", grid_search.best_params_)
    print("Best validation accuracy:", grid_search.best_score_)

    # 用测试集评估最佳模型
    best_model = grid_search.best_estimator_

    # 在测试集上评估模型
    test_accuracy = best_model.score(Test_data_final_tensor, Test_data_final_label_tensor)
    print(f"测试集准确率: {test_accuracy:.4f}")
    
    # 获取预测结果
    y_pred = best_model.predict(Test_data_final_tensor)
    
    # 获取预测概率(用于ROC曲线)
    y_score = best_model.predict_proba(Test_data_final_tensor)
    
    # 准备可视化所需的数据
    index_to_terrain = terrain_mapping['index_to_terrain']
    unique_labels = sorted(list(set(y_test.tolist())))
    target_names = [index_to_terrain[label] for label in unique_labels]  # 移除 str(label)
    
    # 创建结果保存目录
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. 绘制混淆矩阵
    print("\n生成混淆矩阵...")
    cm = plot_confusion_matrix(
        y_test, y_pred, 
        labels=unique_labels,
        display_labels=target_names,
        save_path=os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    )
    
    # 2. 生成分类报告
    print("\n生成分类报告...")
    report = generate_and_save_classification_report(
        y_test, y_pred,
        labels=unique_labels,
        target_names=target_names,
        save_path=os.path.join(RESULTS_DIR, 'classification_report.txt')
    )
    
    # 3. 绘制各类别准确率
    print("\n生成各类别准确率图...")
    class_accuracies = plot_class_accuracies(
        y_test, y_pred,
        labels=unique_labels,
        target_names=target_names,
        save_path=os.path.join(RESULTS_DIR, 'class_accuracies.png')
    )
    
    # 4. 绘制ROC曲线
    print("\n生成ROC曲线...")
    plot_roc_curves(
        y_test, y_score,
        unique_labels=unique_labels,
        class_names=target_names,
        save_path=os.path.join(RESULTS_DIR, 'roc_curves.png')
    )
    
    # 5. 可视化预测样本
    print("\n生成预测样本可视化...")
    plot_prediction_samples(
        Test_data_final_tensor, Test_data_final_label_tensor, y_pred,
        index_to_terrain=index_to_terrain,  # 直接传递,不需要转换
        save_path=os.path.join(RESULTS_DIR, 'prediction_samples.png'),
        num_samples_to_show=9
    )
    
    # 6. 保存评估结果JSON
    # 从 best_params 中只提取可序列化的参数
    best_params = grid_search.best_params_.copy()
    
    evaluation_results = {
        'test_accuracy': float(test_accuracy),
        'best_params': best_params,  # 只保存网格搜索的最佳参数
        'class_accuracies': class_accuracies,
        'confusion_matrix': cm.tolist()
    }
    
    save_evaluation_results(
        evaluation_results,
        save_path=os.path.join(RESULTS_DIR, 'evaluation_results.json')
    )
    
    print(f"\n所有评估结果已保存到: {RESULTS_DIR}")
    
    return best_model, evaluation_results

def save_artifacts(model, terrain_mapping, scalers):
    """
    保存训练好的模型、参数和地形映射关系。
    """
    print("\n模型训练和保存完成")
    # 保存映射关系供后续使用
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(MAPPING_FILE, 'w') as f:
        json.dump(terrain_mapping, f, indent=2)

    # 保存PyTorch模型的权重
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    torch.save(model.module_.state_dict(), 
               os.path.join(WEIGHTS_DIR, 'Adam-best_model_weights-100epochs-.pth'))

    # 保存Skorch模型的超参数
    os.makedirs(HYPERPARAMS_DIR, exist_ok=True)
    joblib.dump(model, 
                os.path.join(HYPERPARAMS_DIR, 'Adam-best_model_params-100epochs-python3.8.10.pkl'))

    joblib.dump(scalers, 
                os.path.join(SCALERS_DIR, 'scalers-100epochs-train-python3.8.10.pkl'))
    # print(f"Scalers已保存到: {os.path.join(SCALERS_DIR, 'scalers-100epochs.pkl')}")

def main():
    """
    主函数，按顺序执行数据加载、预处理、训练和保存。
    """
    all_data, all_labels, terrain_to_index, index_to_terrain, num_classes = load_data(DATA_ROOT)
    X_train, X_test, y_train, y_test, scalers = preprocess_data(all_data, all_labels)

    terrain_mapping = {
        'terrain_to_index': terrain_to_index,
        'index_to_terrain': index_to_terrain
    }
    
    best_model, evaluation_results = train_and_evaluate(
        X_train, y_train, X_test, y_test, num_classes, terrain_mapping
    )
    
    save_artifacts(best_model, terrain_mapping, scalers)

if __name__ == "__main__":
    main()
