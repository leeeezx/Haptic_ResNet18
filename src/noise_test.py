import torch
import numpy as np
import pandas as pd
import os
import json
import joblib
from sklearn.preprocessing import MinMaxScaler
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
WEIGHTS_DIR = 'd:/CodeProject/haptic_ResNet/models/weights'
SCALERS_DIR = 'd:/CodeProject/haptic_ResNet/models/scalers'
HYPERPARAMS_DIR = 'd:/CodeProject/haptic_ResNet/models/hyperparams'
DATA_DIR = 'd:/CodeProject/haptic_ResNet/data'
MAPPING_FILE = os.path.join(DATA_DIR, 'terrain_mapping.json')
LENGTH_CONFIG_FILE = os.path.join(DATA_DIR, 'f2_max_before_after.json')

def add_noise(data, noise_type='gaussian', noise_level=0.1):
    """
    为数据添加噪声
    
    Args:
        data (np.ndarray): 输入数据, shape: (num_samples, 2, sequence_length)
        noise_type (str): 噪声类型 ('gaussian', 'uniform', 'salt_pepper')
        noise_level (float): 噪声水平
    
    Returns:
        np.ndarray: 添加噪声后的数据
    """
    noisy_data = data.copy()
    
    if noise_type == 'gaussian':
        # 高斯噪声
        noise = np.random.normal(0, noise_level, data.shape)
        noisy_data = data + noise
        
    elif noise_type == 'uniform':
        # 均匀噪声
        noise = np.random.uniform(-noise_level, noise_level, data.shape)
        noisy_data = data + noise
        
    elif noise_type == 'salt_pepper':
        # 椒盐噪声
        mask = np.random.random(data.shape) < noise_level
        noisy_data[mask] = np.random.choice([data.min(), data.max()], size=mask.sum())
    
    return noisy_data

def load_test_data(test_data_folder, terrain_mapping, length_config, folder_suffix=''):
    """
    从指定文件夹加载测试数据
    
    Args:
        test_data_folder (str): 测试数据文件夹路径 (例如: "D:/Dataset/01")
        terrain_mapping (dict): 地形映射关系
        length_config (dict): 序列长度配置
        folder_suffix (str): 指定的文件夹后缀(例如: '_f2')
    
    Returns:
        X_test (np.ndarray): 测试数据
        y_test (np.ndarray): 测试标签
    """
    BEFORE_MAX = length_config['before_max']
    AFTER_MAX = length_config['after_max']
    FIXED_LENGTH = length_config['max_length']
    
    terrain_to_index = terrain_mapping['terrain_to_index']
    
    # 从文件夹路径中提取地形名称(文件夹01的名称)
    terrain_name = os.path.basename(test_data_folder)
    
    # 获取地形标签
    if terrain_name not in terrain_to_index:
        print(f"警告: 未知地形 {terrain_name}, 跳过")
        return None, None
    
    label_index = terrain_to_index[terrain_name]
    print(f"处理地形: {terrain_name} -> 标签: {label_index}")
    
    all_data = []
    all_labels = []
    
    # 遍历地形文件夹下的所有子文件夹(文件夹02)
    for subfolder_name in os.listdir(test_data_folder):
        subfolder_path = os.path.join(test_data_folder, subfolder_name)
        
        # 只处理目录
        if not os.path.isdir(subfolder_path):
            continue
        
        # 如果指定了后缀,只处理符合后缀的文件夹
        if folder_suffix and not subfolder_name.endswith(folder_suffix):
            continue
        
        print(f"  处理子文件夹: {subfolder_name}")
        
        # 读取该子文件夹下的所有CSV文件
        csv_files = [f for f in os.listdir(subfolder_path) if f.endswith('.csv')]
        
        for csv_file in csv_files:
            file_path = os.path.join(subfolder_path, csv_file)
            try:
                df = pd.read_csv(file_path)
                
                # 读取两个通道的数据
                sample_data = df[['/realtime_robot_poseAndextTau/data.2', 
                                 '/realtime_robot_poseAndextTau/data.8']].values
                
                all_data.append(sample_data)
                all_labels.append(label_index)
                print(f"    已加载: {csv_file}")
            except Exception as e:
                print(f"    错误: 无法读取 {csv_file} - {e}")
    
    # 对所有样本进行截取和填充 (与train.py保持一致)
    processed_data = []
    for idx, sample in enumerate(all_data):
        # sample 的形状是 (序列长度, 2)
        # 第0列是 data.2, 第1列是 data.8
        force_column = sample[:, 1]  # data.8 列
        
        # 找到最大值的索引
        max_idx = np.argmax(force_column)
        
        # 计算起始和结束索引
        start_idx = max_idx - BEFORE_MAX
        end_idx = max_idx + AFTER_MAX
        
        # 初始化一个固定长度的数组,用0填充
        processed_sample = np.zeros((FIXED_LENGTH, 2))
        
        # 计算实际可用的数据范围
        actual_start = max(0, start_idx)
        actual_end = min(len(sample), end_idx)
        
        # 计算在processed_sample中的放置位置
        target_start = max(0, -start_idx)
        target_end = target_start + (actual_end - actual_start)
        
        # 将实际数据复制到目标位置
        processed_sample[target_start:target_end, :] = sample[actual_start:actual_end, :]
        
        processed_data.append(processed_sample)
    
    # 转换为numpy数组并调整维度
    X = np.array(processed_data)
    y = np.array(all_labels)
    X = X.transpose(0, 2, 1)  # (samples, 2, sequence_length)
    
    print(f"加载测试数据完成: {X.shape[0]} 个样本")
    print(f"截取和填充完成,所有样本长度已统一为: {FIXED_LENGTH}")
    return X, y

def normalize_data(X_test, scalers):
    """
    使用训练时的scaler对测试数据进行归一化
    
    Args:
        X_test (np.ndarray): 测试数据
        scalers (list): scaler列表
    
    Returns:
        np.ndarray: 归一化后的测试数据
    """
    X_normalized = X_test.copy()
    for i in range(X_test.shape[1]):
        X_normalized[:, i, :] = scalers[i].transform(X_test[:, i, :])
    return X_normalized

def evaluate_with_noise(model, X_test, y_test, terrain_mapping, 
                        noise_type='gaussian', noise_level=0.1, 
                        results_dir='results/noise_test'):
    """
    对添加噪声后的数据进行评估
    
    Args:
        model: 训练好的模型
        X_test (np.ndarray): 测试数据
        y_test (np.ndarray): 测试标签
        terrain_mapping (dict): 地形映射关系
        noise_type (str): 噪声类型
        noise_level (float): 噪声水平
        results_dir (str): 结果保存目录
    """
    # 添加噪声
    print(f"\n添加 {noise_type} 噪声 (level={noise_level})...")
    X_noisy = add_noise(X_test, noise_type=noise_type, noise_level=noise_level)
    
    # 获取模型所在设备
    device = next(model.parameters()).device
    
    # 转换为张量并移到正确的设备
    X_noisy_tensor = torch.tensor(X_noisy, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    
    # 设置模型为评估模式
    model.eval()
    
    # 模型推理(禁用梯度计算)
    print("开始模型推理...")
    with torch.no_grad():
        # 直接调用模型获取输出(logits)
        outputs = model(X_noisy_tensor)
        
        # 应用softmax获取概率分布
        y_score = torch.nn.functional.softmax(outputs, dim=1)
        
        # 获取预测类别
        y_pred = torch.argmax(y_score, dim=1)
    
    # 计算准确率(将结果移回CPU进行后续处理)
    y_pred_cpu = y_pred.cpu()
    y_test_cpu = y_test_tensor.cpu()
    y_score_cpu = y_score.cpu()
    
    correct = (y_pred_cpu == y_test_cpu).sum().item()
    test_accuracy = correct / len(y_test_cpu)
    print(f"噪声测试准确率: {test_accuracy:.4f}")
    
    # 准备可视化数据
    index_to_terrain = terrain_mapping['index_to_terrain']
    unique_labels = sorted(list(set(y_test.tolist())))
    # 修改这一行:将整数标签转换为字符串
    target_names = [index_to_terrain[str(label)] for label in unique_labels]
    
    # 创建结果保存目录
    noise_results_dir = os.path.join(results_dir, f"{noise_type}_noise_{noise_level}")
    os.makedirs(noise_results_dir, exist_ok=True)
    
    # # 1. 混淆矩阵
    # print("\n生成混淆矩阵...")
    # cm = plot_confusion_matrix(
    #     y_test, y_pred_cpu,
    #     labels=unique_labels,
    #     display_labels=target_names,
    #     save_path=os.path.join(noise_results_dir, 'confusion_matrix.png')
    # )
    
    # # 2. 分类报告
    # print("生成分类报告...")
    # report = generate_and_save_classification_report(
    #     y_test, y_pred_cpu,
    #     labels=unique_labels,
    #     target_names=target_names,
    #     save_path=os.path.join(noise_results_dir, 'classification_report.txt')
    # )
    
    # # 3. 各类别准确率
    # print("生成各类别准确率图...")
    # class_accuracies = plot_class_accuracies(
    #     y_test, y_pred_cpu,
    #     labels=unique_labels,
    #     target_names=target_names,
    #     save_path=os.path.join(noise_results_dir, 'class_accuracies.png')
    # )
    
    # # 4. ROC曲线
    # print("生成ROC曲线...")
    # plot_roc_curves(
    #     y_test, y_score_cpu,
    #     unique_labels=unique_labels,
    #     class_names=target_names,
    #     save_path=os.path.join(noise_results_dir, 'roc_curves.png')
    # )
    
    # # 5. 预测样本可视化
    # print("生成预测样本可视化...")
    # plot_prediction_samples(
    #     X_noisy_tensor.cpu(), y_test_tensor.cpu(), y_pred_cpu,
    #     index_to_terrain=index_to_terrain,
    #     save_path=os.path.join(noise_results_dir, 'prediction_samples.png'),
    #     num_samples_to_show=9
    # )
    
    # 6. 保存评估结果
    evaluation_results = {
        'noise_type': noise_type,
        'noise_level': float(noise_level),
        'test_accuracy': float(test_accuracy),
        # 'class_accuracies': class_accuracies,
        # 'confusion_matrix': cm.tolist()
    }
    
    # save_evaluation_results(
    #     evaluation_results,
    #     save_path=os.path.join(noise_results_dir, 'evaluation_results.json')
    # )
    
    print(f"\n噪声测试结果已保存到: {noise_results_dir}")
    return test_accuracy

def main():
    """
    主函数
    """
    # 配置参数
    TEST_DATA_FOLDER = r"D:\Dataset\08"
    MODEL_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, 'f2-trueResNet-best_model_weights-100epochs-python3.8.0.pth')
    MODEL_PARAMS_PATH = os.path.join(HYPERPARAMS_DIR, 'f2-trueResNet-best_model_params-100epochs-python3.8.0.pkl')
    SCALERS_PATH = os.path.join(SCALERS_DIR, 'f2-trueResNet-scalers-100epochs-train-python3.8.10.pkl')
    RESULTS_DIR = 'd:/CodeProject/haptic_ResNet/results/noise_test'
    
    # 噪声测试配置
    NOISE_CONFIGS = [
        {'type': 'gaussian', 'level': 0.01},
        {'type': 'gaussian', 'level': 0.02},
        {'type': 'gaussian', 'level': 0.03},
        {'type': 'gaussian', 'level': 0.04},
        {'type': 'gaussian', 'level': 0.05},
        {'type': 'gaussian', 'level': 0.06},
        {'type': 'gaussian', 'level': 0.07},
        {'type': 'gaussian', 'level': 0.08},
        {'type': 'gaussian', 'level': 0.09},
        {'type': 'gaussian', 'level': 0.1},
        {'type': 'uniform', 'level': 0.01},
        {'type': 'uniform', 'level': 0.02},
        {'type': 'uniform', 'level': 0.03},
        {'type': 'uniform', 'level': 0.04},
        {'type': 'uniform', 'level': 0.05},
        {'type': 'uniform', 'level': 0.06},
        {'type': 'uniform', 'level': 0.07},
        {'type': 'uniform', 'level': 0.08},
        {'type': 'uniform', 'level': 0.09},
        {'type': 'uniform', 'level': 0.10},
        {'type': 'uniform', 'level': 0.15},
        {'type': 'uniform', 'level': 0.25},
        {'type': 'salt_pepper', 'level': 0.008},
        {'type': 'salt_pepper', 'level': 0.009},
        {'type': 'salt_pepper', 'level': 0.01},
        {'type': 'salt_pepper', 'level': 0.02},
        {'type': 'salt_pepper', 'level': 0.03},
        {'type': 'salt_pepper', 'level': 0.04},
        {'type': 'salt_pepper', 'level': 0.05},
        {'type': 'salt_pepper', 'level': 0.06},
        {'type': 'salt_pepper', 'level': 0.07},
    ]
    
    # 1. 加载配置文件
    print("加载配置文件...")
    with open(MAPPING_FILE, 'r') as f:
        terrain_mapping = json.load(f)
    
    with open(LENGTH_CONFIG_FILE, 'r') as f:
        length_config = json.load(f)
    
    # 2. 加载测试数据
    print("\n加载测试数据...")
    X_test, y_test = load_test_data(TEST_DATA_FOLDER, terrain_mapping, length_config, folder_suffix='-f2')
    
    # 3. 加载scaler并归一化
    print("\n加载scaler并归一化数据...")
    scalers = joblib.load(SCALERS_PATH)
    X_test_normalized = normalize_data(X_test, scalers)
    
    # 4. 加载模型
    print("\n加载模型...")
    skorch_model = joblib.load(MODEL_PARAMS_PATH)
    
    # 从 Skorch 包装器中获取 PyTorch 模型
    model = skorch_model.module_
    
    # 加载权重文件
    print("加载模型权重...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    
    # 将模型移到设备并设置为评估模式
    model.to(device)
    model.eval()
    print("模型加载完成")
    
    # 5. 对不同噪声配置进行测试
    results_summary = []
    
    for config in NOISE_CONFIGS:
        print(f"\n{'='*50}")
        print(f"测试配置: {config['type']} 噪声, 水平 = {config['level']}")
        print(f"{'='*50}")
        
        accuracy = evaluate_with_noise(
            model=model,
            X_test=X_test_normalized,
            y_test=y_test,
            terrain_mapping=terrain_mapping,
            noise_type=config['type'],
            noise_level=config['level'],
            results_dir=RESULTS_DIR
        )
        
        results_summary.append({
            'noise_type': config['type'],
            'noise_level': config['level'],
            'accuracy': float(accuracy)
        })
    
    # 6. 保存总结报告
    print("-" * 35)
    for result in results_summary:
        print(f"{result['noise_type']:<15} {result['noise_level']:<10.3f} {result['accuracy']:<10.4f}")

if __name__ == "__main__":
    main()