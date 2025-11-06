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
    
    # 修复: 正确构建TERRAINS列表,使用地形名称而不是索引
    TERRAINS = [index_to_terrain[i] for i in range(len(index_to_terrain))]
    NUM_CLASSES = len(TERRAINS)
    
    print(f"地形类别: {TERRAINS}")
    
    # 加载测试数据
    X_test = torch.load(os.path.join(data_dir, 'X_test.pt'))
    y_test = torch.load(os.path.join(data_dir, 'y_test.pt'))
    
    print(f"测试数据形状: {X_test.shape}, 测试标签形状: {y_test.shape}")
    
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
    
    # 1. 混淆矩阵
    cm = confusion_matrix(y_test.numpy(), y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))  
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=TERRAINS)
    disp.plot(ax=ax, cmap='Blues', values_format='d') 
    plt.title('混淆矩阵')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'), dpi=300)
    plt.show()
    
    # 2. 分类报告
    report = classification_report(y_test.numpy(), y_pred, target_names=TERRAINS, digits=4)
    print("\n分类报告:")
    print(report)
    
    with open(os.path.join(results_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)
    
    # 3. 每个类别的准确率
    class_accuracies = []
    for i, terrain in enumerate(TERRAINS):
        mask = y_test.numpy() == i
        if mask.sum() > 0:
            class_acc = accuracy_score(y_test.numpy()[mask], y_pred[mask])
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(TERRAINS)), class_accuracies, color='steelblue', alpha=0.8)
    plt.xlabel('地形类别')
    plt.ylabel('准确率')
    plt.title('各地形类别分类准确率')
    plt.xticks(range(len(TERRAINS)), TERRAINS, rotation=45)
    plt.ylim([0, 1.1])
    plt.grid(axis='y', alpha=0.3)
    
    for i, (bar, acc) in enumerate(zip(bars, class_accuracies)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{acc:.2%}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'class_accuracies.png'), dpi=300)
    plt.show()
    
    # 4. ROC曲线
    y_test_bin = label_binarize(y_test.numpy(), classes=range(NUM_CLASSES))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray'])
    
    for i, color in zip(range(NUM_CLASSES), colors):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{TERRAINS[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='随机猜测')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('多分类ROC曲线')
    plt.legend(loc="lower right", fontsize=8)
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
        
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f'真实: {TERRAINS[true_label]}\n预测: {TERRAINS[pred_label]}', 
                     color=color, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'prediction_samples.png'), dpi=300)
    plt.show()
    
    # 保存评估结果
    results = {
        'test_accuracy': float(test_accuracy),
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