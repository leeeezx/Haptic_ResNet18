import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# 数据读取（三列）
rawdata = pd.read_excel(r"F:\cnn手臂信号\2025.10.19\处理后数据\1.xlsx")
array1 = rawdata.values
x = array1[:, 2]  # 修改这里：使用第三列进行峰值检测（索引2表示第三列）

# 峰谷检测
peaks_max, _ = find_peaks(x, height=0.2, distance=500)
peaks_min, _ = find_peaks(x, height=0.005, distance=500)

# 峰谷配对
min_count = min(len(peaks_max), len(peaks_min))
peaks_max = peaks_max[:min_count]
peaks_min = peaks_min[:min_count]

# 计算中心点
Xave = ((peaks_min + peaks_max) / 2).astype(int)
Xleft = np.clip(Xave - 300, 0, len(array1))
Xright = np.clip(Xave + 300, 0, len(array1))

# 创建空DataFrame用于存储结果
output_columns = []
segment_lengths = []

# 收集所有分段数据
for i, (left, right) in enumerate(zip(Xleft, Xright)):
    seg = array1[left:right]
    seg_length = len(seg)

    # 创建合并列：通道1 + 通道2 + 通道3
    combined = np.zeros(seg_length * 3)
    combined[0:seg_length] = seg[:, 0]  # 通道1数据
    combined[seg_length:2 * seg_length] = seg[:, 1]  # 通道2数据
    combined[2 * seg_length:3 * seg_length] = seg[:, 2]  # 通道3数据

    output_columns.append(combined)
    segment_lengths.append(len(combined))

# 确保所有分段长度一致
max_length = max(segment_lengths) if segment_lengths else 0
all_data = []

for col in output_columns:
    if len(col) < max_length:
        # 填充NaN使长度一致
        padded = np.full(max_length, np.nan)
        padded[:len(col)] = col
        all_data.append(padded)
    else:
        all_data.append(col)

# 创建DataFrame并保存
df_out = pd.DataFrame(all_data).T
df_out.to_excel(r"F:\cnn手臂信号\2025.10.19\处理后数据\切割后数据\1.xlsx", index=False)

# 可视化（更新以反映使用第三列进行峰值检测）
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(array1[:, 0], label='Column 1')
plt.vlines(Xave, ymin=np.nanmin(array1[:, 0]), ymax=np.nanmax(array1[:, 0]), colors='r', linestyles='dashed', alpha=0.5)

plt.subplot(3, 1, 2)
plt.plot(x, label='Column 3 (Peak Detection)')  # 更新标签
plt.plot(peaks_max, x[peaks_max], "x", markersize=10, label='Peaks')
plt.plot(peaks_min, x[peaks_min], "o", markersize=8, label='Valleys')

plt.subplot(3, 1, 3)
plt.plot(array1[:, 1], label='Column 2')  # 更新为显示第二列
plt.vlines(Xave, ymin=np.nanmin(array1[:, 1]), ymax=np.nanmax(array1[:, 1]), colors='r', linestyles='dashed', alpha=0.5)

plt.tight_layout()
plt.suptitle(f"Detected {len(Xave)} periods - 3 Columns Processing (Using Column 3 for Peak Detection)")
plt.show()

print(f"成功处理 {len(Xave)} 个周期，每个周期包含 {max_length // 3} 个数据点")
print(f"输出数据形状: {df_out.shape} (行数: {max_length}, 列数: {len(Xave)})")
print(f"每列数据包含: 前{max_length // 3}行为通道1, 中间{max_length // 3}行为通道2, 后{max_length // 3}行为通道3")