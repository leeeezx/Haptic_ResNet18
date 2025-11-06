import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = {
	"__time",
	"/force_sensor_z/wrench/force/z",
	"/realtime_robot_poseAndextTau/data.2",
	"/realtime_robot_poseAndextTau/data.8",
}


def load_signals(csv_path: str) -> pd.DataFrame:
	# if not csv_path.exists():
	# 	raise FileNotFoundError(f"未找到CSV文件: {csv_path}")

	df = pd.read_csv(csv_path)
	missing = REQUIRED_COLUMNS.difference(df.columns)
	if missing:
		missing_cols = ", ".join(sorted(missing))
		raise ValueError(f"缺少必要列: {missing_cols}")

	return df


def plot_signals(df: pd.DataFrame) -> None:
	fig, axes = plt.subplots(3, 1, figsize=(12, 8))

	axes[0].plot(
		df["__time"],
		df["/force_sensor_z/wrench/force/z"],
		label="力传感器Z轴 (/force_sensor_z/wrench/force/z)",
	)
	axes[0].plot(
		df["__time"],
		df["/realtime_robot_poseAndextTau/data.8"],
		label="外力矩 data.8 (/realtime_robot_poseAndextTau/data.8)",
	)
	axes[0].set_xlabel("时间 (__time)")
	axes[0].set_ylabel("力与外力矩")
	axes[0].legend()
	axes[0].grid(True)

	axes[1].plot(
		df["__time"],
		df["/realtime_robot_poseAndextTau/data.2"],
		label="外力矩 data.2 (/realtime_robot_poseAndextTau/data.2)",
	)
	axes[1].set_xlabel("时间 (__time)")
	axes[1].set_ylabel("外力矩 data.2")
	axes[1].grid(True)

	axes[2].plot(
		df["/realtime_robot_poseAndextTau/data.2"],
		df["/realtime_robot_poseAndextTau/data.8"],
		label="外力矩 data.8 与 data.2",
	)
	axes[2].set_xlabel("外力矩 data.2 (/realtime_robot_poseAndextTau/data.2)")
	axes[2].set_ylabel("外力矩 data.8 (/realtime_robot_poseAndextTau/data.8)")
	axes[2].grid(True)

	fig.tight_layout()


def main() -> None:
	csv_path = r"D:\Dataset\01\01-1-1-1-p2\01-1-1-1-03_f_p_p2.csv"
	df = load_signals(csv_path)
	plot_signals(df)
	plt.show()


if __name__ == "__main__":
	main()
