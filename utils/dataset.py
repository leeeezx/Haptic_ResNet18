'''
处理数据集的实用程序函数。
1. 保存
2. 获取文件路径
'''
import os
import glob



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
