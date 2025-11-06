import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import joblib
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
from CustomMultiChannelResNet18 import CustomMultiChannelResNet18

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

def evaluate_model(model_path, params_path, data_dir, results_dir):
    """
    评估训练好的模型并生成可视化结果
    
    Args:
        model_path: 模型权重文件路径
        params_path: 模型超参数文件路径
        data_dir: 预处理数据目录
        results_dir: 结果保存目录
    """
    # 创建结果保存目录
    os.makedirs(results_dir, exist_ok=True)
    
    # 加载地形映射
    with open(os.path.join(data_dir, 'terrain_mapping.json'), 'r') as f:
        mapping_data = json.load(f)
        terrain_to_index = mapping_data['terrain_to_index']
        index_to_terrain = {int(k): v for k, v in mapping_data['index_to_terrain'].items()}
    
    # 加载测试数据
    X_test = torch.load(os.path.join(data_dir, 'X_test_independent.pt'))
    y_test = torch.load(os.path.join(data_dir, 'y_test_independent.pt'))
    
    print(f"测试数据形状: {X_test.shape}, 测试标签形状: {y_test.shape}")
    
    # 修复: 根据测试数据中实际存在的类别构建TERRAINS列表
    unique_labels = sorted(np.unique(y_test.numpy()).tolist())
    TERRAINS = [index_to_terrain[int(label)] for label in unique_labels]
    NUM_CLASSES = len(TERRAINS)
    
    print(f"测试数据中实际存在的地形类别: {TERRAINS}")
    print(f"对应的标签索引: {unique_labels}")
    print(f"实际类别数量: {NUM_CLASSES}")
    
    # 加载模型
    best_model = joblib.load(params_path)
    model_state_dict = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    best_model.module_.load_state_dict(model_state_dict)
    best_model.module_.eval()
    
    # 评估模型
    test_accuracy = best_model.score(X_test, y_test)
    print(f"测试集准确率: {test_accuracy:.4f}")
    
    # 预测结果
    y_pred = best_model.predict(X_test)
    y_score = best_model.predict_proba(X_test)
    
    # 1. 混淆矩阵 - 只使用实际存在的类别
    cm = confusion_matrix(y_test.numpy(), y_pred, labels=unique_labels)
    
    fig, ax = plt.subplots(figsize=(10, 8))  
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=TERRAINS)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    
    # 增加混淆矩阵中的文字大小
    for texts in disp.text_:
        for text in texts:
            text.set_fontsize(12)
    
    plt.title('混淆矩阵', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'), dpi=300)
    plt.show()
    
    # 2. 分类报告 - 只使用实际存在的类别
    report = classification_report(y_test.numpy(), y_pred, 
                                   labels=unique_labels,
                                   target_names=TERRAINS, 
                                   digits=4)
    print("\n分类报告:")
    print(report)
    
    with open(os.path.join(results_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)
    
    # 3. 每个类别的准确率
    class_accuracies = []
    for label, terrain in zip(unique_labels, TERRAINS):
        mask = y_test.numpy() == label
        if mask.sum() > 0:
            class_acc = accuracy_score(y_test.numpy()[mask], y_pred[mask])
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(TERRAINS)), class_accuracies, color='steelblue', alpha=0.8)
    plt.xlabel('地形类别', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.title('各地形类别分类准确率', fontsize=14)
    plt.xticks(range(len(TERRAINS)), TERRAINS, rotation=45, fontsize=10)
    plt.ylim([0, 1.1])
    plt.grid(axis='y', alpha=0.3)
    
    for i, (bar, acc) in enumerate(zip(bars, class_accuracies)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{acc:.2%}', ha='center', va='bottom', fontsize=11)  # 增加文字大小
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'class_accuracies.png'), dpi=300)
    plt.show()
    
    # 4. ROC曲线 - 重新映射标签到连续索引
    # 创建标签映射: 原始标签 -> 连续索引 [0, 1, 2, ...]
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    y_test_remapped = np.array([label_mapping[label] for label in y_test.numpy()])
    
    # 只保留实际存在类别的预测概率
    y_score_filtered = y_score[:, unique_labels]
    
    y_test_bin = label_binarize(y_test_remapped, classes=range(NUM_CLASSES))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray'])
    
    for i, color in zip(range(NUM_CLASSES), colors):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score_filtered[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{TERRAINS[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='随机猜测')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率', fontsize=12)
    plt.ylabel('真阳性率', fontsize=12)
    plt.title('多分类ROC曲线', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)  # 增加图例字体大小
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'roc_curves.png'), dpi=300)
    plt.show()
    
    # 5. 预测样本可视化
    num_samples_to_show = min(9, len(X_test))
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
        
        # 使用原始索引获取地形名称
        true_terrain = index_to_terrain[true_label]
        pred_terrain = index_to_terrain[pred_label]
        
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f'真实: {true_terrain}\n预测: {pred_terrain}', 
                     color=color, fontweight='bold', fontsize=15)  # 增加标题字体大小
        ax.legend(fontsize=10)  # 增加图例字体大小
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'prediction_samples.png'), dpi=300)
    plt.show()
    
    # 保存评估结果
    results = {
        'test_accuracy': float(test_accuracy),
        'actual_classes': TERRAINS,
        'actual_class_labels': unique_labels,
        'confusion_matrix': cm.tolist(),
        'class_accuracies': {terrain: float(acc) for terrain, acc in zip(TERRAINS, class_accuracies)},
        'class_report': report
    }
    
    with open(os.path.join(results_dir, 'evaluation_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n评估结果已保存到: {results_dir}")

if __name__ == "__main__":
    # 路径设置
    model_path = 'd:/CodeProject/haptic_ResNet/models/weights/best_model_weights-100epochs.pth'
    params_path = 'd:/CodeProject/haptic_ResNet/models/hyperparams/best_model_params-100epochs.pkl'
    data_dir = 'd:/CodeProject/haptic_ResNet/data'
    results_dir = 'd:/CodeProject/haptic_ResNet/results'
    
    # 评估模型
    evaluate_model(model_path, params_path, data_dir, results_dir)