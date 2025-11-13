'''
挨个计算指定文件夹下所有csv文件的数据的频率,并print出来最大、最小、平均频率

输入一个文件夹路径,使用get_file_list获取该文件夹下所有csv文件,
文件格式都是一样的。
读取文件的__time列,首尾相减计算出总时间差,除以行数得到频率。
然后将这个文件夹下所有csv文件的频率计算出来,print最大、最小、平均频率。

'''
import pandas as pd
from dataset import get_file_list


def calculate_frequency(csv_file: str) -> tuple:
    """
    计算单个CSV文件的频率和序列长度
    
    Args:
        csv_file (str): CSV文件路径
        
    Returns:
        tuple: (频率值 (Hz), 序列长度)
    """
    try:
        df = pd.read_csv(csv_file)
        
        # 检查是否存在__time列
        if '__time' not in df.columns:
            print(f"警告: 文件 {csv_file} 中没有找到 __time 列")
            return None, None
        
        # 获取序列长度
        sequence_length = len(df)
        
        # 计算总时间差
        time_diff = df['__time'].iloc[-1] - df['__time'].iloc[0]
        
        # 计算频率: 数据行数 / 总时间
        if time_diff > 0:
            frequency = (len(df) - 1) / time_diff
            return frequency, sequence_length
        else:
            print(f"警告: 文件 {csv_file} 的时间差为0")
            return None, sequence_length
            
    except Exception as e:
        print(f"读取文件 {csv_file} 时发生错误: {e}")
        return None, None


def analyze_folder_frequency(folder_path: str):
    """
    分析文件夹下所有CSV文件的频率和序列长度
    
    Args:
        folder_path (str): 文件夹路径
    """
    # 获取所有CSV文件
    csv_files = get_file_list(folder_path)
    
    if not csv_files:
        print("没有找到CSV文件")
        return
    
    frequencies = []
    sequence_lengths = []
    
    # 计算每个文件的频率和序列长度
    for csv_file in csv_files:
        freq, seq_len = calculate_frequency(csv_file)
        if freq is not None:
            frequencies.append(freq)
            print(f"文件: {csv_file}, 频率: {freq:.2f} Hz, 序列长度: {seq_len}")
        if seq_len is not None:
            sequence_lengths.append(seq_len)
    
    # 统计结果
    if frequencies or sequence_lengths:
        print("\n" + "="*50)
        print(f"总共分析了 {len(csv_files)} 个文件")
        
        if frequencies:
            print(f"\n频率统计:")
            print(f"  最大频率: {max(frequencies):.2f} Hz")
            print(f"  最小频率: {min(frequencies):.2f} Hz")
            print(f"  平均频率: {sum(frequencies)/len(frequencies):.2f} Hz")
        
        if sequence_lengths:
            print(f"\n序列长度统计:")
            print(f"  最长序列: {max(sequence_lengths)} 行")
            print(f"  最短序列: {min(sequence_lengths)} 行")
            print(f"  平均长度: {sum(sequence_lengths)/len(sequence_lengths):.2f} 行")
        
        print("="*50)
    else:
        print("没有成功计算出任何文件的频率或序列长度")


if __name__ == "__main__":
    # 示例用法
    folder_path = r"D:\Dataset\09\09-1-1-1-p"
    analyze_folder_frequency(folder_path)