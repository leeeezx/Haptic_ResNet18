'''
独立运行 T-SNE 可视化的脚本。
'''
import torch
import numpy as np
import joblib
import json
import os

from CustomMultiChannelResNet18 import CustomMultiChannelResNet18
from tsne_visualizer import extract_features, plot_tsne, plot_tsne_with_decision_boundary


def load_processed_data(data_path):
    """
    加载预处理后保存的数据。
    
    Args:
        data_path (str): .npz 文件路径
        
    Returns:
        X_test (np.ndarray): 测试集数据（已归一化）
        y_test (np.ndarray): 测试集标签
    """
    data = np.load(data_path)
    return data['X_test'], data['y_test']


def main():
    # 配置路径
    MODEL_PATH = '/media/xiejiapeng/Work/Backup/CodeProject/haptic_ResNet/models/hyperparams/p-trueResNet-bmp-100epochs-stateMax_linux.pkl'
    MAPPING_PATH = '/media/xiejiapeng/Work/Backup/CodeProject/haptic_ResNet/data/terrain_mapping.json'
    DATA_PATH = '/media/xiejiapeng/Work/Backup/CodeProject/haptic_ResNet/data/processed_data.npz'
    OUTPUT_DIR = '/media/xiejiapeng/Work/Backup/CodeProject/haptic_ResNet/results/tsne'
    
    # 加载模型
    print("加载模型...")
    model = joblib.load(MODEL_PATH)
    
    # 加载地形映射
    print("加载地形映射...")
    with open(MAPPING_PATH, 'r') as f:
        terrain_mapping = json.load(f)
    index_to_terrain = {int(k): v for k, v in terrain_mapping['index_to_terrain'].items()}
    
    # 加载预处理后的测试数据（已归一化）
    print("加载测试数据...")
    X_test, y_test = load_processed_data(DATA_PATH)
    print(f"测试数据形状: {X_test.shape}, 标签数量: {len(y_test)}")
    
    # 提取特征并可视化
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("提取特征...")
    features = extract_features(
        model, 
        torch.tensor(X_test, dtype=torch.float32), 
        device
    )
    print(f"特征形状: {features.shape}")
    
    # 绘制基础 T-SNE 图
    print("\n生成 T-SNE 可视化...")
    plot_tsne(
        features, 
        y_test, 
        index_to_terrain, 
        os.path.join(OUTPUT_DIR, 'tsne_test.png'),
        perplexity=30,
        n_iter=1000
    )
    
    # 可选：绘制带决策边界的 T-SNE 图
    # plot_tsne_with_decision_boundary(
    #     features, 
    #     y_test, 
    #     index_to_terrain, 
    #     os.path.join(OUTPUT_DIR, 'tsne_with_boundary.png')
    # )
    
    print("\nT-SNE 可视化完成！")


if __name__ == "__main__":
    main()