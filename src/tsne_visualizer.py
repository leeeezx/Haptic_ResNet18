'''
此模块提供 T-SNE 特征可视化功能。
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import torch.nn as nn

plt.rcParams['font.sans-serif'] = [    
    'Noto Sans CJK SC',
    'AR PL UMing CN',
    'AR PL UKai CN',
    'WenQuanYi Zen Hei',
    'DejaVu Sans'
]
plt.rcParams['axes.unicode_minus'] = False


def extract_features(model, data_tensor, device='cuda'):
    """
    从模型中提取特征（全连接层之前的特征）。
    
    Args:
        model: 训练好的模型（Skorch 包装器或 PyTorch 模型）
        data_tensor (torch.Tensor): 输入数据
        device (str): 计算设备
        
    Returns:
        np.ndarray: 提取的特征向量
    """
    # 获取底层 PyTorch 模型
    if hasattr(model, 'module_'):
        pytorch_model = model.module_
    else:
        pytorch_model = model
    
    pytorch_model.eval()
    pytorch_model.to(device)
    
    features = []
    
    # 定义 hook 函数来提取 avgpool 层的输出
    def hook_fn(module, input, output):
        features.append(output.detach().cpu())
    
    # 注册 hook 到 avgpool 层
    hook = pytorch_model.avgpool.register_forward_hook(hook_fn)
    
    # 分批处理以避免显存溢出
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(data_tensor), batch_size):
            batch = data_tensor[i:i+batch_size].to(device)
            _ = pytorch_model(batch)
    
    # 移除 hook
    hook.remove()
    
    # 合并所有批次的特征
    all_features = torch.cat(features, dim=0)
    # 展平特征 (batch, channels, 1) -> (batch, channels)
    all_features = all_features.view(all_features.size(0), -1)
    
    return all_features.numpy()


def plot_tsne(features, labels, index_to_terrain, save_path, 
              perplexity=30, n_iter=1000, random_state=42):
    """
    绘制 T-SNE 二维特征可视化图。
    
    Args:
        features (np.ndarray): 特征向量，形状为 (n_samples, n_features)
        labels (np.ndarray): 标签数组
        index_to_terrain (dict): 索引到地形名称的映射
        save_path (str): 图片保存路径
        perplexity (int): T-SNE 困惑度参数
        n_iter (int): 迭代次数
        random_state (int): 随机种子
        
    Returns:
        np.ndarray: T-SNE 降维后的二维坐标
    """
    print(f"正在进行 T-SNE 降维... (perplexity={perplexity}, n_iter={n_iter})")
    
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        init='pca',
        learning_rate='auto'
    )
    
    features_2d = tsne.fit_transform(features)
    
    print("T-SNE 降维完成，开始绘图...")
    
    # 获取唯一标签并排序
    unique_labels = sorted(list(set(labels.tolist())))
    n_classes = len(unique_labels)
    
    # 创建颜色映射
    cmap = plt.cm.get_cmap('tab20' if n_classes > 10 else 'tab10')
    colors = [cmap(i / n_classes) for i in range(n_classes)]
    
    # 绘制散点图
    plt.figure(figsize=(12, 10))
    
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        terrain_name = index_to_terrain[label]
        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[colors[idx]],
            label=terrain_name,
            alpha=0.7,
            s=50,
            edgecolors='white',
            linewidth=0.5
        )
    
    plt.xlabel('T-SNE 维度 1', fontsize=14)
    plt.ylabel('T-SNE 维度 2', fontsize=14)
    plt.title('T-SNE 特征可视化', fontsize=16)
    plt.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=10,
        title='地形类别',
        title_fontsize=12
    )
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"T-SNE 图已保存到: {save_path}")
    
    return features_2d


def plot_tsne_with_decision_boundary(features, labels, index_to_terrain, save_path,
                                      perplexity=30, n_iter=1000, random_state=42):
    """
    绘制带有决策边界背景的 T-SNE 图（可选高级版本）。
    
    Args:
        features (np.ndarray): 特征向量
        labels (np.ndarray): 标签数组
        index_to_terrain (dict): 索引到地形名称的映射
        save_path (str): 图片保存路径
        perplexity (int): T-SNE 困惑度参数
        n_iter (int): 迭代次数
        random_state (int): 随机种子
    """
    from sklearn.neighbors import KNeighborsClassifier
    
    # 先进行 T-SNE 降维
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        init='pca',
        learning_rate='auto'
    )
    features_2d = tsne.fit_transform(features)
    
    # 训练一个简单的 KNN 分类器用于绘制决策边界
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(features_2d, labels)
    
    # 创建网格
    x_min, x_max = features_2d[:, 0].min() - 5, features_2d[:, 0].max() + 5
    y_min, y_max = features_2d[:, 1].min() - 5, features_2d[:, 1].max() + 5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # 预测网格点的类别
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 获取唯一标签
    unique_labels = sorted(list(set(labels.tolist())))
    n_classes = len(unique_labels)
    cmap = plt.cm.get_cmap('tab20' if n_classes > 10 else 'tab10')
    colors = [cmap(i / n_classes) for i in range(n_classes)]
    
    # 绘图
    plt.figure(figsize=(14, 10))
    
    # 绘制决策边界背景
    plt.contourf(xx, yy, Z, alpha=0.2, cmap='tab10')
    
    # 绘制散点
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        terrain_name = index_to_terrain[label]
        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[colors[idx]],
            label=terrain_name,
            alpha=0.8,
            s=60,
            edgecolors='black',
            linewidth=0.5
        )
    
    plt.xlabel('T-SNE 维度 1', fontsize=14)
    plt.ylabel('T-SNE 维度 2', fontsize=14)
    plt.title('T-SNE 特征可视化（带决策边界）', fontsize=16)
    plt.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=10,
        title='地形类别'
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()