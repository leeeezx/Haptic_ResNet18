from sklearn.model_selection import GridSearchCV
import torch
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import pandas as pd
import numpy as np
from CustomMultiChannelResNet18 import CustomMultiChannelResNet18
from skorch import NeuralNetClassifier
import joblib
import matplotlib.pyplot as plt





# ================================================= 原程序（by乔州） ==================================================
# # 导入3类切割后的样本数据
# Rawdata = pd.read_excel("F:\\cnn手臂信号\\2025.9.28\\处理后数据\\切割后数据\\滤波后数据\\石头.xlsx")
# data_0 = Rawdata.values
# Rawdata = pd.read_excel("F:\\cnn手臂信号\\2025.9.28\\处理后数据\\切割后数据\\滤波后数据\\剪刀.xlsx")
# data_1 = Rawdata.values
# Rawdata = pd.read_excel("F:\\cnn手臂信号\\2025.9.28\\处理后数据\\切割后数据\\滤波后数据\\布.xlsx")
# data_2 = Rawdata.values
# Data = [data_0, data_1, data_2]

# # 随机取55列作为训练集合，15列作为测试集
# random_columns = np.random.choice(Data[0].shape[1], 70, replace=False)
# shuffled_array = Data[0][:, random_columns]
# train_data_0 = shuffled_array[:, :55]
# test_data_0 = shuffled_array[:, 55:]

# random_columns = np.random.choice(Data[1].shape[1], 70, replace=False)
# shuffled_array = Data[1][:, random_columns]
# train_data_1 = shuffled_array[:, :55]
# test_data_1 = shuffled_array[:, 55:]

# random_columns = np.random.choice(Data[2].shape[1], 70, replace=False)
# shuffled_array = Data[2][:, random_columns]
# train_data_2 = shuffled_array[:, :55]
# test_data_2 = shuffled_array[:, 55:]

# Train_data = [train_data_0, train_data_1, train_data_2]
# Test_data = [test_data_0, test_data_1, test_data_2]

# # 将Test_data存放到excel文件中方便Test文件调用
# num_rows = 5400
# num_columns = 15
# df = pd.DataFrame(np.zeros((num_rows, num_columns)))
# for i in range(3):
#     df.iloc[i * 1800: (i + 1) * 1800, :] = Test_data[i]
# df.to_excel("Test_data.xlsx", index=False)

# # 修改为整体归一化方式（与第二个代码相同）
# print("开始归一化处理...")
# scaler = MinMaxScaler()
# for i in range(len(Train_data)):
#     Train_data[i] = scaler.fit_transform(Train_data[i])
# print("归一化完成")

# # 调整训练数据的数组形状，生成(165, 3, 600)形状的数组，其中165=3*55
# for i in range(len(Train_data)):
#     Train_data[i] = Train_data[i].T  # 转置为(55, 1800)
# for i in range(len(Train_data)):
#     Train_data[i] = Train_data[i].reshape(55, 3, 600)  # 重塑为(55, 3, 600)
# Train_data_final = np.concatenate(Train_data, axis=0)  # 合并为(165, 3, 600)

# # 生成与训练数据相对应的训练标签，形状为(165,)
# Train_data_final_label = [i for i in range(3) for _ in range(55)]

# # 将训练数据与标签数据转换为张量形式
# Train_data_final_tensor = torch.tensor(Train_data_final, dtype=torch.float32)
# Train_data_final_label_tensor = torch.tensor(Train_data_final_label, dtype=torch.long)

# # 打印数据统计信息
# print("训练数据统计:")
# print(f"数据形状: {Train_data_final_tensor.shape}")
# print(f"数据范围: [{Train_data_final_tensor.min():.4f}, {Train_data_final_tensor.max():.4f}]")
# print(f"数据均值: {Train_data_final_tensor.mean():.4f}, 标准差: {Train_data_final_tensor.std():.4f}")

# # 检查CUDA是否可用
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print("使用GPU进行训练")
# else:
#     device = torch.device("cpu")
#     print("使用CPU进行训练")

# # 定义模型
# num_channels = 3
# num_classes = 3
# model = CustomMultiChannelResNet18(num_channels, num_classes).to(device)

# # 定义超参数网格
# param_grid = {
#     'lr': [0.1, 0.01, 0.001],
#     'batch_size': [32, 64, 128],
#     'max_epochs': [100],
# }

# # 创建Skorch神经网络分类器
# net = NeuralNetClassifier(
#     model,
#     max_epochs=100,
#     criterion=nn.CrossEntropyLoss,
#     optimizer=torch.optim.SGD,
#     callbacks=[],
#     device='cuda' if torch.cuda.is_available() else 'cpu',
#     iterator_train__shuffle=True,
#     iterator_valid__shuffle=False,
# )

# # 使用Skorch的GridSearchCV执行超参数搜索
# print("开始超参数搜索...")
# grid_search = GridSearchCV(net, param_grid, cv=3, scoring='accuracy', n_jobs=1, verbose=1)
# grid_search.fit(Train_data_final_tensor, y=Train_data_final_label_tensor)

# # 输出最佳超参数组合和验证集上的性能
# print("Best params:", grid_search.best_params_)
# print("Best validation accuracy:", grid_search.best_score_)

# # 用测试集评估最佳模型
# best_model = grid_search.best_estimator_

# # 保存PyTorch模型的权重
# torch.save(best_model.module_.state_dict(), '33best_model_weights-100epochs.pth')

# # 保存Skorch模型的超参数
# joblib.dump(best_model, '33best_model_params-100epochs.pkl')

# print("模型训练和保存完成")