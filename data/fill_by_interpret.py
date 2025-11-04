'''
针对频率不一致的数据，进行插值填充以达到统一频率
'''

from math import e
import pandas as pd
import os
import sys
sys.path.append(r"d:\CodeProject\haptic_ResNet")
import utils.dataset

def fill_by_interpret(file_path: str,):
    '''
    1. 加载csv文件中的数据
    2. 
    '''
    try:
        print("")
        # 1. 加载指定路径file_path的CSV文件数据
        df = pd.read_csv(file_path)

        # 2. 对Unix时间戳列（单位：秒）进行归零处理。
        if '__time' in df.columns:
            first_time = df['__time'].dropna().iloc[0]
            df['__time'] = df['__time'] - first_time
            print(f"时间已归零，起始时间偏移: {first_time}")
        else:
            raise ValueError("数据中缺少 '__time' 列")

        # 3. 设置时间索引
        df = df.set_index('__time')
        print("时间索引设置完成")

        # 4. 转换数据类型
        numeric_columns = []
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce') # error='coerce' 将无法转换的值设为NaN
                numeric_columns.append(col)
            except (ValueError, TypeError) as e:
                print(f"列 '{col}' 保持原始类型, 由于: {e}")
        
        # 5. 对数值列进行线性插值填充
        df_interpolated = df.copy()
        for col in numeric_columns:
            # 检查该列是否有NaN值
            nan_count_before = df_interpolated[col].isna().sum()
            total_count = len(df_interpolated[col])
            if nan_count_before == 0:
                print(f"列 '{col}': 未经过插值, 无NaN值")
            elif nan_count_before == total_count:
                # 全为NaN的列，无法进行插值
                print(f"警告: 列 '{col}' 全为NaN值，无法进行插值")
            else:
                # 部分为NaN的列，执行线性插值
                df_interpolated[col] = df_interpolated[col].interpolate(method='linear')
                nan_count_after = df_interpolated[col].isna().sum()
                print(f"列 '{col}': 插值前NaN数量={nan_count_before}, 插值后NaN数量={nan_count_after}")
        
        # 6. 处理开头、结尾的边界nan值
        df_filled = df_interpolated.fillna(method='ffill').fillna(method='bfill')

        # 7. 检查是否还有NaN值
        remain_nans = df_filled.isna().sum().sum() # 两个sum得到总的NaN数量
        if remain_nans > 0:
            print(f"警告: 仍有 {remain_nans} 个NaN值未能填充")
            # 显示哪些列还有NaN
            nan_columns = df_filled.columns[df_filled.isna().any()].tolist()
            print(f"包含NaN的列: {nan_columns}")
        else:
            print("所有NaN值已成功填充")
        
        # 8. 重置索引，将时间列恢复
        df_final = df_filled.reset_index()

        print(f"数据处理完成! 最终数据形状: {df_final.shape}")
        return df_final

    except Exception as e:
        print(f"fill_by_interpret函数运行发生错误: {e}")

def batch_fill(file_list: list[str], output_dir: str = None, suffix: str = "_f"):
    '''
    批量处理多个文件，进行插值
    '''
    successful_outputs = []

    for i, file_path in enumerate(file_list):
        print(f"\n处理第 {i+1}/{len(file_list)} 个文件...")
        df = fill_by_interpret(file_path)

        if df is not None:
            output_file = utils.dataset.save_data(df, file_path, output_dir=output_dir, suffix=suffix)
            if output_file:
                successful_outputs.append(output_file)
        else:
            print(f"文件 {file_path} 处理失败，跳过")


# 已将此函数转移至 utils/dataset.py
# def save_data(df: pd.DataFrame, input_file: str, output_dir: str = None, suffix: str = "_f") -> str:
#     """
#     保存处理后的数据
    
#     Args:
#         df (pd.DataFrame): 处理后的数据
#         input_file (str): 输入文件路径
#         output_dir (str): 输出目录，如果为None则自动生成
#         suffix (str): 文件名后缀
    
#     Returns:
#         str: 输出文件路径
#     """
#     try:
#         if output_dir is None:
#             raise ValueError("未指定输出目录,请提供output_dir参数")
        
#         # 创建输出目录（如果不存在）
#         os.makedirs(output_dir, exist_ok=True)
        
#         # 生成输出文件名
#         input_filename = os.path.splitext(os.path.basename(input_file))[0]
#         output_filename = f"{input_filename}{suffix}.csv"
#         output_file = os.path.join(output_dir, output_filename)
        
#         # 保存文件
#         df.to_csv(output_file, index=False)
#         print(f"处理结果已保存至: {output_file}")
        
#         return output_file
        
#     except Exception as e:
#         print(f"保存文件时发生错误: {e}")
#         return None            
    
if __name__ == "__main__":
    # 测试单个文件
    # test_file = r"G:\\WorkFiles\\科研\\rokae_terrain_dataset\\01\\01-1-1-1\\01-1-1-1-05.csv"  # 替换为实际文件路径
    # batch_fill([test_file], output_dir=r"D:\Dataset")
    
    # 测试批量文件处理
    
    file_list = utils.dataset.get_file_list(r"G:\WorkFiles\科研\rokae_terrain_dataset\14\14-1-1-1")
    batch_fill(file_list, output_dir=r"D:\Dataset\14\14-1-1-1-f")