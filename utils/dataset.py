'''
处理数据集的实用程序函数。
1. 保存
2. 获取文件路径
'''
import os
import glob
import pandas as pd

def get_file_list(input_path: str) -> list:
    """
    根据输入路径获取文件列表
    
    输入路径支持：
    1. 单个文件
    2. 文件夹路径（获取该文件夹下所有的CSV文件）
    3. 文件路径列表，由多个文件组成（直接返回列表）
    
    Args:
        input_path (str): 输入路径，可以是文件路径、文件夹路径或文件路径列表
        
    Returns:
        list: 文件路径列表
    """
    # 如果输入是列表，直接返回
    if isinstance(input_path, list):
        return input_path
    
    # 如果输入是文件夹路径
    if os.path.isdir(input_path):
        # 获取文件夹中所有的csv文件
        csv_files = glob.glob(os.path.join(input_path, "*.csv"))
        if not csv_files:
            print(f"警告：在目录 {input_path} 中没有找到CSV文件")
            return []
        print(f"在目录 {input_path} 中找到 {len(csv_files)} 个CSV文件")
        return csv_files
    
    # 如果输入是单个文件路径
    elif os.path.isfile(input_path):
        return [input_path]
    
    else:
        print(f"错误：路径 {input_path} 不存在")
        return []


def save_data(df: pd.DataFrame, input_file: str, output_dir: str = None, suffix: str = "_f") -> str:
    """
    保存处理后的数据
    
    Args:
        df (pd.DataFrame): 处理后的数据
        input_file (str): 输入文件路径
        output_dir (str): 输出目录，如果为None则自动生成
        suffix (str): 文件名后缀
    
    Returns:
        str: 输出文件路径
    """
    try:
        if output_dir is None:
            raise ValueError("未指定输出目录,请提供output_dir参数")
        
        # 创建输出目录（如果不存在）
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成输出文件名
        input_filename = os.path.splitext(os.path.basename(input_file))[0]
        output_filename = f"{input_filename}{suffix}.csv"
        output_file = os.path.join(output_dir, output_filename)
        
        # 保存文件
        df.to_csv(output_file, index=False)
        print(f"处理结果已保存至: {output_file}")
        
        return output_file
        
    except Exception as e:
        print(f"保存文件时发生错误: {e}")
        return None        