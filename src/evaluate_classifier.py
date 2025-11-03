import torch
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import joblib
import numpy as np
from CustomMultiChannelResNet18 import CustomMultiChannelResNet18
import matplotlib.patches as patches
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# 导入Train中生成的Test_data
Raw_test_data = pd.read_excel("Test_data.xlsx")
Test_data = Raw_test_data.values

# 将数据分割成3个子数组
split_data = np.split(Test_data, 3)  # 改为3

# 归一化处理
scaler = MinMaxScaler()
for i in range(len(split_data)):
    split_data[i] = scaler.fit_transform(split_data[i])

# 调整数据形状
for i in range(len(split_data)):
    split_data[i] = split_data[i].T
for i in range(len(split_data)):
    split_data[i] = split_data[i].reshape(15, 3, 600)
Test_data_final = np.concatenate(split_data, axis=0)

# 生成标签 - 改为3类
Test_data_final_label = [i for i in range(3) for _ in range(15)]  # 改为3

# 转换为张量
Test_data_final_tensor = torch.tensor(Test_data_final, dtype=torch.float32)
Test_data_final_label_tensor = torch.tensor(Test_data_final_label, dtype=torch.long)

# 加载模型 - 改为3类
model = CustomMultiChannelResNet18(num_channels=3, num_classes=3)  # num_classes改为3
model.load_state_dict(torch.load('33best_model_weights-100epochs.pth'))  # 加载3类模型权重
model.eval()
best_model = joblib.load('33best_model_params-100epochs.pkl')  # 加载3类模型参数

# 预测
with torch.no_grad():
    outputs = model(Test_data_final_tensor)
    _, predicted = torch.max(outputs, 1)
probabilities = torch.softmax(outputs, dim=1)
print(probabilities)

# 计算混淆矩阵
conf_matrix = confusion_matrix(Test_data_final_label_tensor, predicted)
print(conf_matrix)

class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
overall_accuracy = accuracy_score(Test_data_final_label_tensor, predicted)

# 改为3个类别名称
class_names = ["0", "1", "2"]
class_names_r = ["2", "1", "0"]  # 改为3个类别

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 自定义颜色过渡
from matplotlib.colors import LinearSegmentedColormap
colors = [(250/255, 252/255, 251/255),
          (234/250, 190/250, 197/250),
          (215/255, 88/255, 109/255)]
cmap_name = 'custom_cmap'
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

# 绘制混淆矩阵 - 修复部分
plt.figure(figsize=(10, 7))  # 调整图形大小以适应3类
img = plt.imshow(conf_matrix*10, interpolation='nearest', cmap=custom_cmap, vmin=0, vmax=100)

# 设置正确的坐标轴标签 - 改为3类
plt.xticks(np.arange(3), class_names)  # 改为3
plt.yticks(np.arange(3), class_names_r)  # 改为3

# 添加每个单元格的准确率 - 修复索引问题
for i in range(3):  # 改为3
    for j in range(3):  # 改为3
        total = conf_matrix[i].sum()
        percentage = conf_matrix[i, j] / total * 100 if total != 0 else 0
        accuracy_text = f'{percentage:.1f}%\n{conf_matrix[i, j]}'
        plt.text(j, i, accuracy_text, ha='center', va='center', fontsize=12, fontweight='bold', color='black')

plt.title(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")
cbar = plt.colorbar(img, shrink=0.65, pad=0.01)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")

plt.tight_layout()
# 修复保存路径 - 添加文件名和扩展名
save_path = 'D:\\python_study\\机器学习部分\\Pytorch\\three\\threeinput\\threeInput_IdentifyCharacteristic_\\new\\confusion_matrix_3class.png'
plt.savefig(save_path, dpi=300)
plt.show()