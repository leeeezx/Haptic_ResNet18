from sklearn.model_selection import GridSearchCV, train_test_split
import torch
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import pandas as pd
import numpy as np
from CustomMultiChannelResNet18 import CustomMultiChannelResNet18
from skorch import NeuralNetClassifier
import joblib
import matplotlib.pyplot as plt
import os

# 1. 定义数据路径和类别
# ==========================================
DATA_ROOT = "D:/Dataset/"  # 存放所有地形文件夹的根目录
TERRAINS = sorted(os.listdir(DATA_ROOT)) # 自动获取所有地形文件夹名称，如 ['terrain_1', 'terrain_2']
NUM_CLASSES = len(TERRAINS) # 地形类别的数量

print(f"发现 {NUM_CLASSES} 种地形: {TERRAINS}")

# 2. 从CSV文件加载数据和标签
# ==========================================
all_data = []  # 用来存放所有样本的数据
all_labels = [] # 用来存放所有样本的标签

# 遍历每一种地形
for label_index, terrain_name in enumerate(TERRAINS):
    terrain_path = os.path.join(DATA_ROOT, terrain_name)
    
    # 获取该地形文件夹下所有的csv文件
    csv_files = [f for f in os.listdir(terrain_path) if f.endswith('.csv')]
    
    # 遍历并读取每一个csv文件
    for csv_file in csv_files:
        file_path = os.path.join(terrain_path, csv_file)
        
        # 使用pandas读取csv
        df = pd.read_csv(file_path)
        
        # 假设您的列名是 'displacement' 和 'force'，请根据实际情况修改
        # .values 会将数据框转换为Numpy数组
        # 我们希望得到的数组形状是 (序列长度, 2)
        sample_data = df[['displacement', 'force']].values 
        
        all_data.append(sample_data)
        all_labels.append(label_index) # 标签是文件夹的索引，例如 terrain_1 -> 0, terrain_2 -> 1

print(f"数据加载完成！共加载了 {len(all_data)} 个样本。")


# 3. 将数据列表转换为一个大的Numpy数组
# ==========================================
# np.array(all_data) 会创建一个形状为 (样本数, 序列长度, 2) 的数组
# 例如，如果您有165个文件，每个文件600行，形状就是 (165, 600, 2)
X = np.array(all_data)
y = np.array(all_labels)

# 我们的模型需要输入的形状是 (样本数, 通道数, 序列长度)
# 所以需要交换最后两个维度 (600, 2) -> (2, 600)
# transpose(0, 2, 1) 的意思是：保持第0维（样本数）不变，将第2维（通道数）和第1维（序列长度）交换
X = X.transpose(0, 2, 1)

print(f"原始数据形状 (X): {X.shape}") # 应该打印 (样本数, 2, 序列长度)
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
# 我们需要对每个通道分别进行归一化
for i in range(X_train.shape[1]):  # X_train.shape[1] 就是通道数，这里是 2
    scaler = MinMaxScaler()
    # scaler学习训练集第i个通道的缩放规则
    X_train[:, i, :] = scaler.fit_transform(X_train[:, i, :])
    # scaler应用在测试集第i个通道上
    X_test[:, i, :] = scaler.transform(X_test[:, i, :])
    
print("归一化完成")


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