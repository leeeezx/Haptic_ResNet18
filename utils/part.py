import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import dataset
import os

def find_offset_index(input_dir: str, left_offset: float, right_offset: float) -> tuple:
    '''
    依次处理目标目录的每个文件,计算左右偏移对应的索引,返回95%分位数作为目标偏移量。

    **原理**:
    1. 遍历目录中的所有CSV文件
    2. 对每个文件按照part_signal_time的逻辑检测峰值
    3. 根据左右偏移量计算每个文件的实际索引偏移
    4. 收集所有文件的左右偏移索引
    5. 分别计算左右偏移的95%分位数

    Args:
        input_dir (str): 输入目录路径
        left_offset (float): 左边界偏移量(时间)
        right_offset (float): 右边界偏移量(时间)
    
    Returns:
        tuple: (左偏移95%分位数, 右偏移95%分位数)
    '''
    try:
        # 获取目录中所有CSV文件
        file_list = dataset.get_file_list(input_dir)
        if not file_list:
            raise ValueError(f"在目录 {input_dir} 中未找到CSV文件")
        
        all_left_offsets = []   # 存储所有左偏移索引
        all_right_offsets = []  # 存储所有右偏移索引
        file_info = []  # 存储每个文件的详细信息
        
        print(f"\n开始处理 {len(file_list)} 个文件...")
        print("=" * 80)
        
        for i, file_path in enumerate(file_list):
            try:
                # 读取CSV文件
                df = pd.read_csv(file_path)
                
                # 检查必要的列是否存在
                if '__time' not in df.columns:
                    print(f"警告: 文件 {os.path.basename(file_path)} 缺少 '__time' 列,跳过")
                    continue
                if '/realtime_robot_poseAndextTau/data.8' not in df.columns:
                    print(f"警告: 文件 {os.path.basename(file_path)} 缺少 '/realtime_robot_poseAndextTau/data.8' 列,跳过")
                    continue
                
                # 检测峰值
                signal_8 = df['/realtime_robot_poseAndextTau/data.8'].values
                time_values = df['__time'].values
                
                peaks, _ = find_peaks(signal_8, height=10, distance=100)
                
                if len(peaks) == 0:
                    print(f"警告: 文件 {os.path.basename(file_path)} 未检测到峰值,跳过")
                    continue
                
                # 选择最大峰值
                max_peak_idx = peaks[np.argmax(signal_8[peaks])]
                peak_time = time_values[max_peak_idx]
                
                # 计算时间范围
                start_time = peak_time - left_offset
                end_time = peak_time + right_offset
                
                # 找到对应的索引范围
                start_idx = np.searchsorted(time_values, start_time)
                end_idx = np.searchsorted(time_values, end_time)
                
                # 确保索引在有效范围内
                start_idx = max(0, start_idx)
                end_idx = min(len(df), end_idx)
                
                # 计算实际的左右偏移索引
                left_offset_idx = max_peak_idx - start_idx
                right_offset_idx = end_idx - max_peak_idx
                
                all_left_offsets.append(left_offset_idx)
                all_right_offsets.append(right_offset_idx)
                
                # 记录文件信息
                file_info.append({
                    'file': os.path.basename(file_path),
                    'peak_idx': max_peak_idx,
                    'peak_time': peak_time,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'left_offset_idx': left_offset_idx,
                    'right_offset_idx': right_offset_idx,
                    'total_length': left_offset_idx + right_offset_idx
                })
                
                print(f"[{i+1}/{len(file_list)}] {os.path.basename(file_path)}: "
                      f"峰值索引={max_peak_idx}, 左偏移={left_offset_idx}, 右偏移={right_offset_idx}")
                
            except Exception as e:
                print(f"处理文件 {os.path.basename(file_path)} 时出错: {e}")
                continue
        
        if not all_left_offsets or not all_right_offsets:
            raise ValueError("没有成功处理任何文件")
        
        # 计算95%分位数
        all_left_offsets.sort()
        all_right_offsets.sort()
        
        percentile_95_idx = int(len(all_left_offsets) * 0.95)
        left_offset_95 = all_left_offsets[percentile_95_idx]
        right_offset_95 = all_right_offsets[percentile_95_idx]
        
        # 输出统计信息
        print("\n" + "=" * 80)
        print(f"处理完成! 成功处理 {len(all_left_offsets)} 个文件")
        print(f"\n左偏移索引统计:")
        print(f"  最小值: {min(all_left_offsets)}")
        print(f"  最大值: {max(all_left_offsets)}")
        print(f"  平均值: {np.mean(all_left_offsets):.2f}")
        print(f"  中位数: {np.median(all_left_offsets):.0f}")
        print(f"  95%分位数: {left_offset_95}")
        
        print(f"\n右偏移索引统计:")
        print(f"  最小值: {min(all_right_offsets)}")
        print(f"  最大值: {max(all_right_offsets)}")
        print(f"  平均值: {np.mean(all_right_offsets):.2f}")
        print(f"  中位数: {np.median(all_right_offsets):.0f}")
        print(f"  95%分位数: {right_offset_95}")
        
        print(f"\n建议使用的偏移量:")
        print(f"  左偏移 (left_offset): {left_offset_95}")
        print(f"  右偏移 (right_offset): {right_offset_95}")
        print(f"  总长度: {left_offset_95 + right_offset_95}")
        print("=" * 80)
        
        return left_offset_95, right_offset_95
        
    except Exception as e:
        print(f"\033[91m find_offset_index函数执行出错: {e} \033[0m")
        return None, None

def part_signal_time(file_path: str, left_offset: float, right_offset: float):
    '''
    依据机械臂z力信号，在time列分割周期信号。

    **原理**：
    检测8信号的峰值，即机械臂的z力。
    由于波形是固定的，即平稳->上升->下降，不存在波谷，理论上也只有一个波峰。
    所以只检测8信号峰值，然后通过我记录的前后__time列的距离，来确定一个周期的范围。
    分割完后，依据__time列前后索引范围，来分割其他信号列的数据。
    最终保存为一个新csv文件。加上后缀名p

    Args:
        file_path (str): 输入csv文件路径。
        left_offset (float): 左边界偏移量，用于确定分割起点。
        right_offset (float): 右边界偏移量，用于确定分割终点。
    Returns:
        pd.DataFrame: 分割后的数据DataFrame,如果处理失败则返回None
    '''
    try:
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
        elif len(peaks) > 0:
            max_peak_idx = peaks[np.argmax(signal_8[peaks])]
            peaks = np.array([max_peak_idx])
            print('最大峰值为:', signal_8[max_peak_idx], '，对应时间:', time_values[max_peak_idx])
        
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
        
        print(f"成功分割 {len(segments)} 个周期")
        print(f"输出数据形状: {result_df.shape}")
        
        return result_df
    except Exception as e:
        print(f"\033[91m 使用part_signal函数处理文件 {file_path} 时出错: {e} \033[0m")
        return None
    
def part_signal_num(file_path: str, left_offset: int, right_offset: int):
    '''
    依据机械臂z力信号,按数据点个数分割周期信号。

    **原理**:
    检测8信号的峰值,即机械臂的z力。
    由于波形是固定的,即平稳->上升->下降,不存在波谷,理论上也只有一个波峰。
    所以只检测8信号峰值,然后通过数据点个数(索引),来确定一个周期的范围。
    分割完后,依据索引范围,来分割其他信号列的数据。
    最终保存为一个新csv文件。

    Args:
        file_path (str): 输入csv文件路径。
        left_offset (int): 左边界偏移量(数据点个数),用于确定分割起点。
        right_offset (int): 右边界偏移量(数据点个数),用于确定分割终点。
    Returns:
        pd.DataFrame: 分割后的数据DataFrame,如果处理失败则返回None
    '''
    try:
        # 1. 读取csv文件
        df = pd.read_csv(file_path)
        
        # 检查必要的列是否存在
        if '/realtime_robot_poseAndextTau/data.8' not in df.columns:
            raise ValueError("CSV文件中未找到 '/realtime_robot_poseAndextTau/data.8' 列")
        
        # 2. 对"/realtime_robot_poseAndextTau/data.8"检测峰值
        signal_8 = df['/realtime_robot_poseAndextTau/data.8'].values
        
        # 检测峰值,使用适当的参数
        peaks, properties = find_peaks(signal_8, height=10, distance=100)
        
        if len(peaks) == 0:
            raise ValueError(f"错误: 在文件 {file_path} 中未检测到峰值")
        elif len(peaks) > 0:
            max_peak_idx = peaks[np.argmax(signal_8[peaks])]
            peaks = np.array([max_peak_idx])
            print('最大峰值为:', signal_8[max_peak_idx], '，对应索引:', max_peak_idx)
        
        # 3. 以峰值为中心,确定峰值对应的索引,然后依据left_offset和right_offset,计算出分割的索引范围
        segments = []
        for peak_idx in peaks:
            # 计算分割的索引范围
            start_idx = peak_idx - left_offset
            end_idx = peak_idx + right_offset
            
            # 确保索引在有效范围内
            start_idx = max(0, start_idx)
            end_idx = min(len(df), end_idx)
            
            segments.append((start_idx, end_idx))
        
        # 4. 依据这个索引范围,分割整个文件的数据
        segmented_data = []
        for i, (start_idx, end_idx) in enumerate(segments):
            segment = df.iloc[start_idx:end_idx].copy()
            segment['segment_id'] = i  # 添加分段标识
            segmented_data.append(segment)
        
        # 合并所有分段
        result_df = pd.concat(segmented_data, ignore_index=True)
        
        print(f"成功分割 {len(segments)} 个周期")
        print(f"输出数据形状: {result_df.shape}")
        
        return result_df
    except Exception as e:
        print(f"\033[91m 使用part_signal_num函数处理文件 {file_path} 时出错: {e} \033[0m")
        return None

def batch_part_time(file_list: list[str], left_offset: float, right_offset: float, output_dir: str = None, suffix: str = "_f"):
    '''
    批量处理多个文件，进行周期分割

    Args:
        file_list (list[str]): 待处理的文件路径列表
        left_offset (float): 左边界偏移量
        right_offset (float): 右边界偏移量
        output_dir (str): 输出目录,如果为None则需要指定
        suffix (str): 文件名后缀,默认为"_p"
    
    Returns:
        list[str]: 成功处理并保存的文件路径列表
    '''
    successful_outputs = []
    for i, file_path in enumerate(file_list):
        print(f"\n处理第 {i+1}/{len(file_list)} 个文件...")
        df = part_signal_time(file_path, left_offset, right_offset)
        
        if df is not None:
            output_file = dataset.save_data(df, file_path, output_dir=output_dir, suffix=suffix)
            if output_file:
                successful_outputs.append(output_file)
        else:
            print(f"\033[91m 文件 {file_path} 处理失败,跳过 \033[0m")
    
    print(f"\n批量处理完成! 成功处理 {len(successful_outputs)}/{len(file_list)} 个文件")
    return successful_outputs

def batch_part_num(file_list: list[str], left_offset: int, right_offset: int, output_dir: str = None, suffix: str = "_f"):
    '''
    批量处理多个文件，进行周期分割,并统一长度

    Args:
        file_list (list[str]): 待处理的文件路径列表
        left_offset (int): 左边界偏移量
        right_offset (int): 右边界偏移量
        output_dir (str): 输出目录,如果为None则需要指定
        suffix (str): 文件名后缀,默认为"_f"
    
    Returns:
        list[str]: 成功处理并保存的文件路径列表
    '''
    successful_outputs = []
    target_length = left_offset + right_offset
    
    for i, file_path in enumerate(file_list):
        print(f"\n处理第 {i+1}/{len(file_list)} 个文件...")
        df = part_signal_num(file_path, left_offset, right_offset)
        
        if df is not None:
            current_length = len(df)
            
            # 检查长度是否符合目标
            if current_length < target_length:
                # 重新读取原文件以获取峰值信息
                original_df = pd.read_csv(file_path)
                signal_8 = original_df['/realtime_robot_poseAndextTau/data.8'].values
                peaks, _ = find_peaks(signal_8, height=10, distance=100)
                max_peak_idx = peaks[np.argmax(signal_8[peaks])]
                
                # 计算实际的左右偏移
                actual_start_idx = max(0, max_peak_idx - left_offset)
                actual_end_idx = min(len(original_df), max_peak_idx + right_offset)
                actual_left = max_peak_idx - actual_start_idx
                actual_right = actual_end_idx - max_peak_idx
                
                left_shortage = left_offset - actual_left
                right_shortage = right_offset - actual_right
                
                print(f"  文件长度不足: 当前={current_length}, 目标={target_length}")
                print(f"  左侧缺失={left_shortage}, 右侧缺失={right_shortage}")
                
                # 左侧填充
                if left_shortage > 0:
                    # 使用第一行数据填充
                    first_row = df.iloc[0:1]
                    padding_left = pd.concat([first_row] * left_shortage, ignore_index=True)
                    df = pd.concat([padding_left, df], ignore_index=True)
                    print(f"  左侧填充 {left_shortage} 行")
                
                # 右侧填充
                if right_shortage > 0:
                    # 使用最后一行数据填充
                    last_row = df.iloc[-1:]
                    padding_right = pd.concat([last_row] * right_shortage, ignore_index=True)
                    df = pd.concat([df, padding_right], ignore_index=True)
                    print(f"  右侧填充 {right_shortage} 行")
                
                print(f"  填充后长度: {len(df)}")
            
            elif current_length > target_length:
                # 如果超出目标长度,进行截断
                df = df.iloc[:target_length]
                print(f"  截断到目标长度: {target_length}")
            
            # 保存处理后的文件
            output_file = dataset.save_data(df, file_path, output_dir=output_dir, suffix=suffix)
            if output_file:
                successful_outputs.append(output_file)
        else:
            print(f"\033[91m 文件 {file_path} 处理失败,跳过 \033[0m")
    
    print(f"\n批量处理完成! 成功处理 {len(successful_outputs)}/{len(file_list)} 个文件")
    print(f"所有文件已统一为长度: {target_length}")
    return successful_outputs


if __name__ == "__main__":
# ================================== 批量分割周期信号 ========================================
    # file_list = dataset.get_file_list(r"D:\Dataset\14\14-1-1-1-f")
    # batch_part(file_list, 
    #            left_offset=1.5, right_offset=6.8, 
    #            output_dir=r"D:\Dataset\14\14-1-1-1-p", 
    #            suffix="_p")

# ================================== 计算偏移量索引 ========================================
    # find_offset_index(r"D:\Dataset\14\14-1-1-1-p", left_offset=1.5, right_offset=6.8)

# ================================== 批量分割周期信号 ========================================
    file_list = dataset.get_file_list(r"D:\Dataset\14\14-1-1-1-p")
    batch_part_num(file_list, 
               left_offset=1531, right_offset=6911, 
               output_dir=r"D:\Dataset\14\14-1-1-1-p2", 
               suffix="_p2")


# ==================================== 原程序（by乔州） ==============================================
# # 数据读取（三列）
# rawdata = pd.read_excel(r"F:\cnn手臂信号\2025.10.19\处理后数据\1.xlsx")
# array1 = rawdata.values
# x = array1[:, 2]  # 修改这里：使用第三列进行峰值检测（索引2表示第三列）

# # 峰谷检测
# peaks_max, _ = find_peaks(x, height=0.2, distance=500)
# peaks_min, _ = find_peaks(x, height=0.005, distance=500)

# # 峰谷配对
# min_count = min(len(peaks_max), len(peaks_min))
# peaks_max = peaks_max[:min_count]
# peaks_min = peaks_min[:min_count]

# # 计算中心点
# Xave = ((peaks_min + peaks_max) / 2).astype(int)
# Xleft = np.clip(Xave - 300, 0, len(array1))
# Xright = np.clip(Xave + 300, 0, len(array1))

# # 创建空DataFrame用于存储结果
# output_columns = []
# segment_lengths = []

# # 收集所有分段数据
# for i, (left, right) in enumerate(zip(Xleft, Xright)):
#     seg = array1[left:right]
#     seg_length = len(seg)

#     # 创建合并列：通道1 + 通道2 + 通道3
#     combined = np.zeros(seg_length * 3)
#     combined[0:seg_length] = seg[:, 0]  # 通道1数据
#     combined[seg_length:2 * seg_length] = seg[:, 1]  # 通道2数据
#     combined[2 * seg_length:3 * seg_length] = seg[:, 2]  # 通道3数据

#     output_columns.append(combined)
#     segment_lengths.append(len(combined))

# # 确保所有分段长度一致
# max_length = max(segment_lengths) if segment_lengths else 0
# all_data = []

# for col in output_columns:
#     if len(col) < max_length:
#         # 填充NaN使长度一致
#         padded = np.full(max_length, np.nan)
#         padded[:len(col)] = col
#         all_data.append(padded)
#     else:
#         all_data.append(col)

# # 创建DataFrame并保存
# df_out = pd.DataFrame(all_data).T
# df_out.to_excel(r"F:\cnn手臂信号\2025.10.19\处理后数据\切割后数据\1.xlsx", index=False)

# # 可视化（更新以反映使用第三列进行峰值检测）
# plt.figure(figsize=(12, 8))
# plt.subplot(3, 1, 1)
# plt.plot(array1[:, 0], label='Column 1')
# plt.vlines(Xave, ymin=np.nanmin(array1[:, 0]), ymax=np.nanmax(array1[:, 0]), colors='r', linestyles='dashed', alpha=0.5)

# plt.subplot(3, 1, 2)
# plt.plot(x, label='Column 3 (Peak Detection)')  # 更新标签
# plt.plot(peaks_max, x[peaks_max], "x", markersize=10, label='Peaks')
# plt.plot(peaks_min, x[peaks_min], "o", markersize=8, label='Valleys')

# plt.subplot(3, 1, 3)
# plt.plot(array1[:, 1], label='Column 2')  # 更新为显示第二列
# plt.vlines(Xave, ymin=np.nanmin(array1[:, 1]), ymax=np.nanmax(array1[:, 1]), colors='r', linestyles='dashed', alpha=0.5)

# plt.tight_layout()
# plt.suptitle(f"Detected {len(Xave)} periods - 3 Columns Processing (Using Column 3 for Peak Detection)")
# plt.show()

# print(f"成功处理 {len(Xave)} 个周期，每个周期包含 {max_length // 3} 个数据点")
# print(f"输出数据形状: {df_out.shape} (行数: {max_length}, 列数: {len(Xave)})")
# print(f"每列数据包含: 前{max_length // 3}行为通道1, 中间{max_length // 3}行为通道2, 后{max_length // 3}行为通道3")

