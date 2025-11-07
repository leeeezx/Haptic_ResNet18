'''
单独处理独立测试集的数据评估任务
此处的独立测试集：非同一批实验条件下采集的数据
'''
import os
import json
import numpy as np
import torch
import joblib

from CustomMultiChannelResNet18 import CustomMultiChannelResNet18
import evaluation_visualizer as ev


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
    y_test_numpy = y_test.numpy()
    
    print(f"测试数据形状: {X_test.shape}, 测试标签形状: {y_test.shape}")
    
    # 根据测试数据中实际存在的类别构建TERRAINS列表
    unique_labels = sorted(np.unique(y_test_numpy).tolist())
    TERRAINS = [index_to_terrain[int(label)] for label in unique_labels]
    
    print(f"测试数据中实际存在的地形类别: {TERRAINS}")
    print(f"对应的标签索引: {unique_labels}")
    
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
    print("\n正在生成混淆矩阵...")
    cm = ev.plot_confusion_matrix(
        y_true=y_test_numpy,
        y_pred=y_pred,
        labels=unique_labels,
        display_labels=TERRAINS,
        save_path=os.path.join(results_dir, 'confusion_matrix.png')
    )
    
    # 2. 分类报告
    print("\n正在生成分类报告...")
    report = ev.generate_and_save_classification_report(
        y_true=y_test_numpy,
        y_pred=y_pred,
        labels=unique_labels,
        target_names=TERRAINS,
        save_path=os.path.join(results_dir, 'classification_report.txt')
    )
    
    # 3. 每个类别的准确率
    print("\n正在生成各类别准确率图...")
    class_accuracies_dict = ev.plot_class_accuracies(
        y_true=y_test_numpy,
        y_pred=y_pred,
        labels=unique_labels,
        target_names=TERRAINS,
        save_path=os.path.join(results_dir, 'class_accuracies.png')
    )
    
    # 4. ROC曲线
    print("\n正在生成ROC曲线图...")
    ev.plot_roc_curves(
        y_true=y_test_numpy,
        y_score=y_score,
        unique_labels=unique_labels,
        class_names=TERRAINS,
        save_path=os.path.join(results_dir, 'roc_curves.png')
    )
    
    # 5. 预测样本可视化
    print("\n正在生成预测样本可视化图...")
    ev.plot_prediction_samples(
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        index_to_terrain=index_to_terrain,
        save_path=os.path.join(results_dir, 'prediction_samples.png')
    )
    
    # 6. 保存评估结果
    results = {
        'test_accuracy': float(test_accuracy),
        'actual_classes': TERRAINS,
        'actual_class_labels': unique_labels,
        'confusion_matrix': cm.tolist(),
        'class_accuracies': class_accuracies_dict,
        'class_report': report
    }
    
    print("\n正在保存评估结果...")
    ev.save_evaluation_results(
        results=results,
        save_path=os.path.join(results_dir, 'evaluation_results.json')
    )

if __name__ == "__main__":
    # 路径设置
    model_path = 'd:/CodeProject/haptic_ResNet/models/weights/best_model_weights-100epochs.pth'
    params_path = 'd:/CodeProject/haptic_ResNet/models/hyperparams/best_model_params-100epochs.pkl'
    data_dir = 'd:/CodeProject/haptic_ResNet/data'
    results_dir = 'd:/CodeProject/haptic_ResNet/results/independent_test_evaluate'
    
    # 评估模型
    evaluate_model(model_path, params_path, data_dir, results_dir)