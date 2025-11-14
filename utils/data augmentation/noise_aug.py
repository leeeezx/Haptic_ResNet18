"""
"D:\Dataset\08\08-1-1-1-p"、"D:\Dataset\09\09-1-1-1-p"
使用dataset中的get_file_list获取这两个文件夹下的所有文件，
分别随机选中五个csv文件（每个文件夹各五个），对每个文件的'/realtime_robot_poseAndextTau/data.8'列添加高斯噪声，不保存文件，绘制出添加噪声前后的对比图。
并将两个文件夹的对比图放在同一个图中进行展示。
"""

import os
import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 将项目根目录加入路径，便于导入dataset工具
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = r'D:\CodeProject\haptic_ResNet'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.dataset import get_file_list  # noqa: E402


TARGET_COLUMN = "/realtime_robot_poseAndextTau/data.8"
FOLDER_PATHS = [
    r"D:\Dataset\08\08-1-1-1-p",
    r"D:\Dataset\09\09-1-1-1-p",
]
SAMPLES_PER_FOLDER = 5
NOISE_FACTOR = 0.009  # 噪声强度（占原始标准差的比例）
RANDOM_SEED = 42


def select_random_files(folder_path: str, sample_size: int):
    file_list = get_file_list(folder_path)
    if len(file_list) < sample_size:
        raise ValueError(f"{folder_path} 中的CSV文件不足 {sample_size} 个。")
    return random.sample(file_list, sample_size)


def add_gaussian_noise(series: pd.Series, factor: float):
    std = series.std(ddof=0)
    noise = np.random.normal(loc=0.0, scale=max(std * factor, 1e-8), size=len(series))
    return series + noise


def plot_folder(ax, file_paths, folder_label: str) -> None:
    cmap = plt.get_cmap("tab10")
    for idx, file_path in enumerate(file_paths):
        df = pd.read_csv(file_path)
        file_name = Path(file_path).name
        if TARGET_COLUMN not in df.columns:
            ax.text(
                0.02,
                0.95 - idx * 0.05,
                f"{file_name} 缺少列",
                transform=ax.transAxes,
                fontsize=8,
                color="red",
            )
            continue

        original_series = df[TARGET_COLUMN]
        noisy_series = add_gaussian_noise(original_series, NOISE_FACTOR)
        color = cmap(idx % 10)

        ax.plot(
            original_series.values,
            label=f"{file_name}-原始",
            linewidth=1.0,
            color=color,
        )
        ax.plot(
            noisy_series.values,
            label=f"{file_name}-加噪",
            linewidth=1.0,
            linestyle="--",
            color=color,
            alpha=0.8,
        )

    ax.set_title(f"{folder_label}", fontsize=11)
    ax.tick_params(labelsize=8)
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(fontsize=7, ncol=2)


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    fig, axes = plt.subplots(
        nrows=len(FOLDER_PATHS),
        ncols=1,
        figsize=(6, 3 * len(FOLDER_PATHS)),
        constrained_layout=True,
    )

    if len(FOLDER_PATHS) == 1:
        axes = [axes]

    for row_idx, folder in enumerate(FOLDER_PATHS):
        selected_files = select_random_files(folder, SAMPLES_PER_FOLDER)
        plot_folder(axes[row_idx], selected_files, Path(folder).name)

    fig.suptitle("高斯噪声前后对比（按文件夹）", fontsize=14)
    plt.show()


if __name__ == "__main__":
    main()