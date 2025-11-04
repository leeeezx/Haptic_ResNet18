import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import sys
sys.path.append(r"d:\CodeProject\haptic_ResNet")
import utils.dataset
import os

def part_signal(file_path: str, left_offset: float, right_offset: float):
    '''
    检测8信号的峰值，即机械臂的z力。
    由于波形是固定的，即平稳->上升->下降，不存在波谷，理论上也只有一个波峰。
    所以只检测8信号峰值，然后通过我记录的前后__time列的距离，来确定一个周期的范围。
    分割完后，依据__time列前后索引范围，来分割其他信号列的数据。
    最终保存为一个新csv文件。加上后缀名p

    Args:
        file_path (str): 输入csv文件路径。
        left_offset (float): 左边界偏移量，用于确定分割起点。
        right_offset (float): 右边界偏移量，用于确定分割终点。
    '''
    # 1. 读取csv文件
    df = pd.read_csv(file_path)
    
    # 检查必要的列是否存在
    if '__time' not in df.columns:
        raise ValueError("CSV文件中未找到 '__time' 列")
    if '/realtime_robot_poseAndextTau/data.8' not in df.columns:
        raise ValueError("CSV文件中未找到 '/realtime_robot_poseAndextTau/data.8' 列")
    
    # 2. 对"/realtime_robot_poseAndextTau/data.8"检测峰值
    signal_8 = df['/realtime_robot_poseAndextTau/data.8'].values
    time_values = df['__time'].values
    
    # 检测峰值,使用适当的参数
    peaks, properties = find_peaks(signal_8, height=10, distance=100)
    
    if len(peaks) == 0:
        raise ValueError(f"错误: 在文件 {file_path} 中未检测到峰值")
    elif len(peaks) > 1:
        raise ValueError(f"错误: 在文件 {file_path} 中检测到 {len(peaks)} 个峰值,理论上应该只有一个峰值")
    
    # 3. 以峰值为中心,确定峰值对应的__time值,然后依据left_offset和right_offset,计算出分割的时间范围
    segments = []
    for peak_idx in peaks:
        peak_time = time_values[peak_idx]
        
        # 计算分割的时间范围
        start_time = peak_time - left_offset
        end_time = peak_time + right_offset
        
        # 找到对应的索引范围
        start_idx = np.searchsorted(time_values, start_time)
        end_idx = np.searchsorted(time_values, end_time)
        
        # 确保索引在有效范围内
        start_idx = max(0, start_idx)
        end_idx = min(len(df), end_idx)
        
        segments.append((start_idx, end_idx))
    
    # 4. 依据这个时间范围,分割整个文件的数据
    segmented_data = []
    for i, (start_idx, end_idx) in enumerate(segments):
        segment = df.iloc[start_idx:end_idx].copy()
        segment['segment_id'] = i  # 添加分段标识
        segmented_data.append(segment)
    
    # 合并所有分段
    result_df = pd.concat(segmented_data, ignore_index=True)
    
    # 5. 将分割后的数据保存为一个新的csv文件,文件名加上后缀名"_p"
    output_dir = os.path.dirname(file_path)
    output_file = utils.dataset.save_data(result_df, file_path, output_dir, suffix="_p")
    
    print(f"成功分割 {len(segments)} 个周期")
    print(f"输出数据形状: {result_df.shape}")
    
    return output_file

# ==================================================================================
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