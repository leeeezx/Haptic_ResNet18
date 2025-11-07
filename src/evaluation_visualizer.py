
'''
此模块提供用于模型评估结果可视化的函数。
'''
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

def plot_confusion_matrix(y_true, y_pred, labels, display_labels, save_path):
    """
    生成并保存混淆矩阵图。
    
    Args:
        y_true (np.ndarray): 真实标签。
        y_pred (np.ndarray): 预测标签。
        labels (list): 标签索引列表。
        display_labels (list): 用于显示的标签名称列表。
        save_path (str): 图片保存路径。
        
    Returns:
        np.ndarray: 计算出的混淆矩阵。
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    fig, ax = plt.subplots(figsize=(10, 8))  
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    
    for texts in disp.text_:
        for text in texts:
            text.set_fontsize(20)
    
    ax.set_xlabel(ax.get_xlabel(), fontsize=15)
    ax.set_ylabel(ax.get_ylabel(), fontsize=15)
    ax.tick_params(axis='both', labelsize=15)
    
    plt.title('混淆矩阵', fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    return cm

def generate_and_save_classification_report(y_true, y_pred, labels, target_names, save_path):
    """
    生成并保存分类报告。
    
    Args:
        y_true (np.ndarray): 真实标签。
        y_pred (np.ndarray): 预测标签。
        labels (list): 标签索引列表。
        target_names (list): 类别名称列表。
        save_path (str): 文本文件保存路径。
        
    Returns:
        str: 生成的分类报告。
    """
    report = classification_report(y_true, y_pred, 
                                   labels=labels,
                                   target_names=target_names, 
                                   digits=4)
    print("\n分类报告:")
    print(report)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report)
    return report

def plot_class_accuracies(y_true, y_pred, labels, target_names, save_path):
    """
    生成并保存各类别准确率柱状图。
    
    Args:
        y_true (np.ndarray): 真实标签。
        y_pred (np.ndarray): 预测标签。
        labels (list): 标签索引列表。
        target_names (list): 类别名称列表。
        save_path (str): 图片保存路径。
        
    Returns:
        dict: 包含各类别准确率的字典。
    """
    class_accuracies = []
    for label in labels:
        mask = y_true == label
        if mask.sum() > 0:
            class_acc = accuracy_score(y_true[mask], y_pred[mask])
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(target_names)), class_accuracies, color='steelblue', alpha=0.8)
    plt.xlabel('地形类别', fontsize=15)
    plt.ylabel('准确率', fontsize=15)
    plt.title('各地形类别分类准确率', fontsize=15)
    plt.xticks(range(len(target_names)), target_names, rotation=45, fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim([0, 1.1])
    plt.grid(axis='y', alpha=0.3)
    
    for i, (bar, acc) in enumerate(zip(bars, class_accuracies)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{acc:.2%}', ha='center', va='bottom', fontsize=15)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    return {terrain: float(acc) for terrain, acc in zip(target_names, class_accuracies)}

def plot_roc_curves(y_true, y_score, unique_labels, class_names, save_path):
    """
    生成并保存多分类ROC曲线图。
    
    Args:
        y_true (np.ndarray): 真实标签。
        y_score (np.ndarray): 预测得分或概率。
        unique_labels (list): 数据中实际存在的唯一标签索引。
        class_names (list): 类别名称列表。
        save_path (str): 图片保存路径。
    """
    num_classes = len(class_names)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    y_test_remapped = np.array([label_mapping[label] for label in y_true])
    
    y_score_filtered = y_score[:, unique_labels]
    
    y_test_bin = label_binarize(y_test_remapped, classes=range(num_classes))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray'])
    
    for i, color in zip(range(num_classes), colors):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score_filtered[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='随机猜测')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率', fontsize=15)
    plt.ylabel('真阳性率', fontsize=15)
    plt.title('多分类ROC曲线', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc="lower right", fontsize=15)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def plot_prediction_samples(X_test, y_test, y_pred, index_to_terrain, save_path, num_samples_to_show=9):
    """
    可视化预测样本。
    
    Args:
        X_test (torch.Tensor): 测试数据张量。
        y_test (torch.Tensor): 测试标签张量。
        y_pred (np.ndarray): 预测标签。
        index_to_terrain (dict): 索引到地形名称的映射。
        save_path (str): 图片保存路径。
        num_samples_to_show (int): 显示的样本数量。
    """
    num_samples_to_show = min(num_samples_to_show, len(X_test))
    indices = np.random.choice(len(X_test), num_samples_to_show, replace=False)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, sample_idx in enumerate(indices):
        sample = X_test[sample_idx]
        true_label = y_test[sample_idx].item()
        pred_label = y_pred[sample_idx]
        
        ax = axes[idx]
        ax.plot(sample[0].numpy(), label='通道1', alpha=0.7)
        ax.plot(sample[1].numpy(), label='通道2', alpha=0.7)
        
        true_terrain = index_to_terrain[true_label]
        pred_terrain = index_to_terrain[pred_label]
        
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f'真实: {true_terrain}\n预测: {pred_terrain}', 
                     color=color, fontweight='bold', fontsize=15)
        ax.legend(fontsize=15)
        ax.tick_params(axis='both', labelsize=15)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def save_evaluation_results(results, save_path):
    """
    将评估结果字典保存为JSON文件。
    
    Args:
        results (dict): 包含评估结果的字典。
        save_path (str): JSON文件保存路径。
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n评估结果的 JSON 文件已保存到: {save_path}")
