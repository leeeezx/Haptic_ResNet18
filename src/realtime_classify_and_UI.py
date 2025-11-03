import sys
import time
import threading
from queue import Queue
from collections import deque
from functools import partial

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont
import serial
from serial.tools import list_ports
from scipy.signal import butter, filtfilt, iirnotch, find_peaks
import torch
import joblib
from sklearn.preprocessing import MinMaxScaler
import os
import pandas as pd

# 导入UI文件
from Mainwindow_ import Ui_MainWindow
from CustomMultiChannelResNet18 import CustomMultiChannelResNet18

# 全局常量
HISTORY_LENGTH = 1000  # 实时数据长度
MAX_CUMULATIVE_LENGTH = 10000  # 累计数据最大长度
SAMPLE_RATE = 500  # 采样率
PREDICTION_THRESHOLD = 330  # 预测触发阈值（针对ADC值调整）
PREDICTION_WINDOW = 600  # 预测窗口大小
PEAK_DETECTION_WINDOW = 50  # 峰值检测窗口大小


class DataBuffer:
    """数据缓冲区类，用于管理多通道数据"""

    def __init__(self, num_channels=8, realtime_size=1000, cumulative_size=10000):
        self.num_channels = num_channels
        self.realtime_size = realtime_size
        self.cumulative_size = cumulative_size

        # 实时数据缓冲区（固定大小）
        self.realtime_data = [deque(maxlen=realtime_size) for _ in range(num_channels)]
        self.realtime_timestamps = deque(maxlen=realtime_size)

        # 累计数据缓冲区（可增长，但有最大限制）
        self.cumulative_data = [deque(maxlen=cumulative_size) for _ in range(num_channels)]
        self.cumulative_timestamps = deque(maxlen=cumulative_size)

        self.lock = threading.RLock()  # 使用可重入锁，更安全

    def add_data(self, channel_data, timestamp):
        """添加数据到缓冲区"""
        with self.lock:
            # 添加到实时缓冲区
            self.realtime_timestamps.append(timestamp)
            for i, value in enumerate(channel_data):
                if i < self.num_channels:
                    self.realtime_data[i].append(value)

            # 添加到累计缓冲区
            self.cumulative_timestamps.append(timestamp)
            for i, value in enumerate(channel_data):
                if i < self.num_channels:
                    self.cumulative_data[i].append(value)

    def get_realtime_data(self, channel_idx, start=0, end=None):
        """获取指定通道的实时数据"""
        with self.lock:
            data = self.realtime_data[channel_idx]
            if end is None:
                end = len(data)
            # 直接返回deque的切片视图，避免复制整个列表
            return [data[i] for i in range(start, min(end, len(data)))]

    def get_cumulative_data(self, channel_idx, start=0, end=None):
        """获取指定通道的累计数据"""
        with self.lock:
            data = self.cumulative_data[channel_idx]
            if end is None:
                end = len(data)
            return [data[i] for i in range(start, min(end, len(data)))]

    def get_all_realtime_data(self, start=0, end=None):
        """获取所有通道的实时数据"""
        with self.lock:
            if end is None:
                end = len(self.realtime_timestamps)
            timestamps = [self.realtime_timestamps[i] for i in range(start, min(end, len(self.realtime_timestamps)))]
            channel_data = [
                [self.realtime_data[i][j] for j in range(start, min(end, len(self.realtime_data[i])))]
                for i in range(self.num_channels)
            ]
            return timestamps, channel_data

    def get_all_cumulative_data(self, start=0, end=None):
        """获取所有通道的累计数据"""
        with self.lock:
            if end is None:
                end = len(self.cumulative_timestamps)
            timestamps = [self.cumulative_timestamps[i] for i in
                          range(start, min(end, len(self.cumulative_timestamps)))]
            channel_data = [
                [self.cumulative_data[i][j] for j in range(start, min(end, len(self.cumulative_data[i])))]
                for i in range(self.num_channels)
            ]
            return timestamps, channel_data

    def clear(self):
        """清空缓冲区"""
        with self.lock:
            for channel in self.realtime_data:
                channel.clear()
            self.realtime_timestamps.clear()

            for channel in self.cumulative_data:
                channel.clear()
            self.cumulative_timestamps.clear()


class EMGProcessor:
    """EMG信号处理类"""

    def __init__(self, sample_rate=500):
        self.sample_rate = sample_rate
        self.nyq = 0.5 * sample_rate

        # 设计高通滤波器（去除基线漂移）
        fc_high = 15  # 高通截止频率: 15Hz
        order_high = 3  # 滤波器阶数
        high = fc_high / self.nyq
        self.b_high, self.a_high = butter(order_high, high, btype='high')

        # 设计陷波滤波器（去除工频干扰）- 修改后的版本
        f0 = 50  # 工频干扰频率
        bw = 3  # 带宽: 3Hz

        # 计算归一化频率和带宽（与MATLAB一致）
        wo = f0 / (sample_rate / 2)  # 归一化中心频率
        bw_normalized = bw / (sample_rate / 2)  # 归一化带宽

        # 使用自定义函数实现与MATLAB一致的陷波滤波器
        self.b_notch, self.a_notch = self.design_notch_filter(wo, bw_normalized)

        # 设计低通滤波器（提取包络）
        fc_low = 20  # 低通截止频率: 4Hz
        order_low = 3  # 滤波器阶数
        low = fc_low / self.nyq
        self.b_low, self.a_low = butter(order_low, low, btype='low')

        # 不再使用归一化器
        print("注意: 不使用归一化器，将在原始处理后的数据上进行预测")

    # 添加自定义陷波滤波器设计函数
    def design_notch_filter(self, wo, bw):
        """
        设计与MATLAB iirnotch函数一致的陷波滤波器
        wo: 归一化中心频率 (0 < wo < 1)
        bw: 归一化带宽 (0 < bw < 1)
        返回: 滤波器系数 b, a
        """
        # 计算极点位置
        r = 1 - bw / 2
        # 计算零点位置
        z = np.array([np.exp(1j * np.pi * wo), np.exp(-1j * np.pi * wo)])
        # 计算极点位置
        p = r * z

        # 转换为传递函数形式
        b = np.poly(z)
        a = np.poly(p)

        # 归一化使得直流增益为1
        b = b / np.polyval(b, 1) * np.polyval(a, 1)

        return b, a

    def process_signal(self, signal):
        """处理EMG信号"""
        # 1. 去除基线漂移（高通滤波）
        signal_high = filtfilt(self.b_high, self.a_high, signal)

        # 2. 去除工频干扰（50Hz陷波滤波）
        signal_notch = filtfilt(self.b_notch, self.a_notch, signal_high)

        # 3. 绝对值整流
        signal_rectified = np.abs(signal_notch)

        # 确保整流后的信号没有负值（处理数值精度问题）
        signal_rectified = np.maximum(signal_rectified, 0)

        # 4. 巴特沃斯低通滤波（提取包络）
        signal_filtered = filtfilt(self.b_low, self.a_low, signal_rectified)

        # 确保最终信号没有负值
        signal_filtered = np.maximum(signal_filtered, 0)

        # 打印处理后的信号信息（调试用）
        print(
            f"处理后的信号: 长度={len(signal_filtered)}, 范围=[{np.min(signal_filtered):.6f}, {np.max(signal_filtered):.6f}]")

        return signal_filtered

    def normalize_data(self, data):
        """不使用归一化器，直接返回原始处理后的数据"""
        return np.array(data)


class GestureClassifier:
    """手势分类器类"""

    def __init__(self, model_path, params_path, num_channels=3, num_classes=3):
        self.model_path = model_path
        self.params_path = params_path
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.model = None
        self.best_model = None
        self.is_loaded = False
        self.load_model()

    def load_model(self):
        """加载模型"""
        try:
            # 加载PyTorch模型
            self.model = CustomMultiChannelResNet18(
                num_channels=self.num_channels,
                num_classes=self.num_classes
            )
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.eval()

            # 加载Skorch模型的超参数
            self.best_model = joblib.load(self.params_path)

            self.is_loaded = True
            print("模型加载成功")
            return True
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False

    def predict(self, data):
        """预测手势"""
        if not self.is_loaded:
            print("模型未加载，无法进行预测")
            return None

        if data.shape != (1, self.num_channels, PREDICTION_WINDOW):
            print(f"数据形状不正确: {data.shape}，期望: (1, {self.num_channels}, {PREDICTION_WINDOW})")
            return None

        try:
            # 禁用梯度计算（推理阶段）
            with torch.no_grad():
                outputs = self.model(data)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                return predicted, probabilities
        except Exception as e:
            print(f"预测失败: {e}")
            return None


class SerialThread(threading.Thread):
    """串口读取线程"""

    def __init__(self, serial_port, data_buffer, parent=None):
        super().__init__()
        self.serial_port = serial_port
        self.data_buffer = data_buffer
        self.parent = parent
        self._is_running = True
        self.start_time = time.time()
        self.daemon = True

    def run(self):
        """线程主函数"""
        serial_buffer = b''

        while self._is_running:
            if self.serial_port and self.serial_port.is_open:
                try:
                    # 检查串口是否仍然打开
                    if not self.serial_port.is_open:
                        time.sleep(0.1)
                        continue

                    # 读取串口数据
                    bytes_to_read = self.serial_port.inWaiting()
                    if bytes_to_read > 0:
                        serial_buffer += self.serial_port.read(bytes_to_read)

                    # 处理完整的数据行
                    lines = serial_buffer.split(b'\n')
                    # 保留最后一行不完整的数据
                    serial_buffer = lines[-1]
                    messages = lines[:-1]

                    for dat_bytes in messages:
                        if not dat_bytes:
                            continue

                        try:
                            # 解析数据
                            dat_str = dat_bytes.decode('utf-8', errors='ignore').strip()
                            if not dat_str:
                                continue

                            # 移除可能的非数字字符前缀
                            # 查找第一个数字或负号的位置
                            first_digit_pos = -1
                            for i, char in enumerate(dat_str):
                                if char.isdigit() or char == '-':
                                    first_digit_pos = i
                                    break

                            if first_digit_pos > 0:
                                # 移除前缀的非数字字符
                                dat_str = dat_str[first_digit_pos:]

                            # 分割字符串并尝试转换为浮点数
                            str_values = dat_str.split(',')
                            unjson = []

                            for v in str_values:
                                try:
                                    # 尝试转换为浮点数
                                    unjson.append(float(v))
                                except ValueError:
                                    # 如果转换失败，尝试移除可能的非数字字符
                                    # 只保留数字、小数点、负号和科学计数法符号
                                    cleaned_v = ''.join([c for c in v if c in '0123456789.-eE'])
                                    if cleaned_v:  # 确保清理后的字符串不为空
                                        try:
                                            unjson.append(float(cleaned_v))
                                        except ValueError:
                                            # 如果仍然无法转换，使用0作为默认值
                                            unjson.append(0.0)
                                    else:
                                        unjson.append(0.0)

                            # 确保我们有足够的数据点
                            if len(unjson) < 8:
                                # 如果数据点不足，用0填充
                                unjson.extend([0.0] * (8 - len(unjson)))
                            elif len(unjson) > 8:
                                # 如果数据点过多，只取前8个
                                unjson = unjson[:8]

                            # 直接使用原始ADC值，不转换为电压值
                            adc_values = unjson

                            # 添加时间戳和数据到缓冲区
                            timestamp = time.time() - self.start_time
                            self.data_buffer.add_data(adc_values, timestamp)

                        except (ValueError, IndexError) as e:
                            print(f"数据处理错误: {e} - 原始数据: {dat_bytes}")
                            continue
                except (serial.SerialException, OSError) as e:
                    print(f"串口读取错误: {e}")
                    if self.parent:
                        QTimer.singleShot(0, lambda: self.parent.show_error_message(f"串口读取错误: {e}"))
                    # 等待一段时间后重试
                    time.sleep(0.1)
                except Exception as e:
                    print(f"未知错误: {e}")
                    time.sleep(0.1)
            else:
                time.sleep(0.1)  # 串口未打开时等待

    def stop(self):
        """停止线程"""
        self._is_running = False
        self.join(timeout=1.0)  # 等待线程结束，最多1秒


class MYWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.setupUi(self)  # 加载设计好的界面

        # 初始化变量
        self.start_time = time.time()
        self.start_status = 'stop'
        self.layer_status = 'off'
        self.active_channels = 8
        self.curve_refs = [[None] * 8, [None] * 8]  # 实时图和累计图的曲线引用

        # 初始化数据缓冲区
        self.data_buffer = DataBuffer(
            num_channels=8,
            realtime_size=HISTORY_LENGTH,
            cumulative_size=MAX_CUMULATIVE_LENGTH
        )

        # 初始化EMG处理器
        self.emg_processor = EMGProcessor(sample_rate=SAMPLE_RATE)

        # 初始化手势分类器
        self.gesture_classifier = GestureClassifier(
            model_path='33best_model_weights-100epochs.pth',
            params_path='33best_model_params-100epochs.pkl',
            num_channels=3,
            num_classes=3
        )

        # 初始化串口相关变量
        self.mSerial = None
        self.serial_thread = None

        # 添加峰值检测相关变量
        self.last_peak_idx = -1  # 记录上一次检测到的峰值位置
        self.peak_detection_cooldown = 0  # 峰值检测冷却时间

        # 设置界面
        self.setup_ui()

        # 连接信号和槽
        self.connect_signals()

    def setup_ui(self):
        """设置UI界面"""
        # 设置图表
        self.setup_charts()

        # 设置按钮状态
        self.update_button_states()

        # 设置复选框状态
        self.checkBox.setChecked(True)
        self.all()

    def setup_charts(self):
        """设置图表"""
        pg.setConfigOptions(antialias=True, background='w')
        win = pg.GraphicsLayoutWidget()
        self.gridLayout_2.addWidget(win, 2, 0, 1, 1)
        win.setWindowTitle(u'肌电信号波形图')
        win.resize(800, 500)

        # 创建实时图 - 调整Y轴范围以适应ADC值
        self.p = win.addPlot(row=0, col=0)
        self.p.setRange(xRange=[0, HISTORY_LENGTH], yRange=[0, 1000], padding=0)
        font = QFont()
        font.setFamily("Arial")
        font.setWeight(QFont.Bold)
        self.p.setLabel(axis='left', text='ADC Value', font=font)
        self.p.setLabel(axis='bottom', text='x / point', font=font)
        self.p.setTitle('实时信号', font=font)

        # 创建累计图 - 调整Y轴范围以适应ADC值
        self.p1 = win.addPlot(row=1, col=0)
        self.p1.setRange(xRange=[0, MAX_CUMULATIVE_LENGTH], yRange=[0, 1000], padding=0)
        self.p1.setLabel(axis='left', text='ADC Value', font=font)
        self.p1.setLabel(axis='bottom', text='x / point', font=font)
        self.p1.setTitle('累计信号', font=font)

        # 初始化曲线
        colors = ['r', 'g', 'b', 'c', 'm', (145, 25, 52), (104, 103, 137), 'k']
        for i, color in enumerate(colors):
            pen = pg.mkPen(color=color, width=1)
            self.curve_refs[0][i] = self.p.plot(pen=pen)
            self.curve_refs[1][i] = self.p1.plot(pen=pen)

    def connect_signals(self):
        """连接信号和槽"""
        # 连接按钮信号
        self.pushButton.clicked.connect(self.start)
        self.pushButton_2.clicked.connect(self.stop)
        self.pushButton_3.clicked.connect(self.clear)
        self.pushButton_4.clicked.connect(self.preserve)
        self.pushButton_5.clicked.connect(self.toggle_layered_mode)
        self.pushButton_8.clicked.connect(self.serial_find)
        self.pushButton_7.clicked.connect(self.serial_open)
        self.pushButton_6.clicked.connect(self.serial_close)
        self.pushButton_9.clicked.connect(lambda: self.stackedWidget.setCurrentWidget(self.page_5))
        self.pushButton_10.clicked.connect(lambda: self.stackedWidget.setCurrentWidget(self.page_6))
        self.pushButton_11.clicked.connect(lambda: self.stackedWidget.setCurrentWidget(self.page_7))
        self.pushButton_12.clicked.connect(lambda: self.stackedWidget.setCurrentWidget(self.page_8))

        # 连接复选框信号
        self.checkBox.stateChanged.connect(self.all)
        # 使用部分应用简化重复代码
        for i in range(1, 9):
            checkbox = getattr(self, f"checkBox_{i + 1}")
            checkbox.stateChanged.connect(partial(self.toggle_channel_visibility, i - 1))

        # 设置定时器
        self.data_timer = QTimer()
        self.data_timer.timeout.connect(self.update_data_display)
        self.data_timer.start(50)  # 20Hz刷新率

        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self.update_plots)
        self.plot_timer.start(200)  # 5Hz刷新率

        self.prediction_timer = QTimer()
        self.prediction_timer.timeout.connect(self.check_for_prediction)
        self.prediction_timer.start(100)  # 10Hz刷新率

    def update_button_states(self):
        """更新按钮状态"""
        is_serial_open = self.mSerial is not None and self.mSerial.is_open

        self.pushButton.setEnabled(self.start_status == 'stop')
        self.pushButton_2.setEnabled(self.start_status == 'start')
        self.pushButton_3.setEnabled(True)
        self.pushButton_4.setEnabled(True)
        self.pushButton_5.setEnabled(True)
        self.pushButton_7.setEnabled(not is_serial_open)
        self.pushButton_6.setEnabled(is_serial_open)
        self.pushButton_8.setEnabled(True)

        # 设置复选框启用状态
        for i in range(1, 10):
            checkbox = getattr(self, f"checkBox_{i}" if i > 1 else "checkBox")
            checkbox.setEnabled(True)

    def start(self):
        """开始采集数据"""
        self.start_status = 'start'
        self.start_time = time.time()
        self.update_button_states()

    def stop(self):
        """停止采集数据"""
        self.start_status = 'stop'
        self.update_button_states()

    def clear(self):
        """清除数据"""
        self.data_buffer.clear()
        self.start_time = time.time()
        self.label_8.setText("                                  ")
        self.label_8.setStyleSheet("font-size: 50px;")
        font = QFont()
        font.setFamily("Arial")
        font.setPointSize(30)
        self.label_8.setFont(font)

    def preserve(self):
        """保存数据"""
        timestamps, all_data = self.data_buffer.get_all_cumulative_data()

        if not timestamps:
            self.show_warning_message("没有数据可保存")
            return

        save_data = [timestamps] + all_data
        save_data = list(map(list, zip(*save_data)))  # 转置

        column_names = ['time'] + [f'channel_{i + 1}' for i in range(len(all_data))]
        df = pd.DataFrame(save_data, columns=column_names)

        file_path, _ = QFileDialog.getSaveFileName(
            self, '选择保存路径', 'Data_save/', 'CSV文件 (*.csv)'
        )

        if file_path:
            try:
                df.to_csv(file_path, index=False)
                self.show_info_message(f"数据已保存到: {file_path}")
            except Exception as e:
                self.show_error_message(f"保存失败: {e}")

    def toggle_layered_mode(self):
        """切换分层模式"""
        if self.layer_status == 'off':
            self.layer_status = 'on'
            self.pushButton_5.setText('关闭分层模式')
        else:
            self.layer_status = 'off'
            self.pushButton_5.setText('启动分层模式')

    def serial_find(self):
        """查找可用串口"""
        self.comboBox_2.clear()
        try:
            port_list = [port.device for port in list_ports.comports()]

            for i, port in enumerate(port_list):
                self.comboBox_2.addItem(port)

            if not port_list:
                self.show_info_message("未找到可用串口")
        except Exception as e:
            self.show_error_message(f"查找串口失败: {e}")

    def serial_open(self):
        """打开串口"""
        baudrate = self.comboBox.currentText()
        port_name = self.comboBox_2.currentText()

        if not port_name:
            self.show_warning_message("请先选择串口")
            return

        try:
            self.mSerial = serial.Serial(port_name, int(baudrate), timeout=1)

            # 启动串口读取线程
            self.serial_thread = SerialThread(self.mSerial, self.data_buffer, self)
            self.serial_thread.start()

            self.start()
            self.update_button_states()
            self.show_info_message(f"串口 {port_name} 已打开")

        except Exception as e:
            self.show_error_message(f"打开串口失败: {e}")

    def serial_close(self):
        """关闭串口"""
        # 停止串口线程
        if self.serial_thread:
            self.serial_thread.stop()
            self.serial_thread = None

        # 关闭串口
        if self.mSerial:
            try:
                if self.mSerial.is_open:
                    self.mSerial.close()
            except Exception as e:
                print(f"关闭串口时出错: {e}")
            finally:
                self.mSerial = None

        self.stop()
        self.update_button_states()
        self.show_info_message("串口已关闭")

    def update_data_display(self):
        """更新数据显示"""
        if self.start_status != 'start':
            return

        # 这里可以添加更新其他数据显示的逻辑
        pass

    def update_plots(self):
        """更新图表显示"""
        # 更新实时图
        realtime_timestamps, realtime_data = self.data_buffer.get_all_realtime_data()
        if realtime_timestamps:
            for i in range(min(8, len(realtime_data))):
                if self.curve_refs[0][i] is not None:
                    self.curve_refs[0][i].setData(realtime_data[i])

        # 更新累计图
        cumulative_timestamps, cumulative_data = self.data_buffer.get_all_cumulative_data()
        if cumulative_timestamps:
            for i in range(min(8, len(cumulative_data))):
                if self.curve_refs[1][i] is not None:
                    self.curve_refs[1][i].setData(cumulative_data[i])

    def check_for_prediction(self):
        """检查是否需要进行预测"""
        if self.start_status != 'start':
            return

        # 峰值检测冷却
        if self.peak_detection_cooldown > 0:
            self.peak_detection_cooldown -= 1
            return

        # 获取通道数据（原始ADC值）
        channel_data = []
        for i in range(3):  # 只使用前三个通道
            data = self.data_buffer.get_realtime_data(i)
            channel_data.append(data)

        # 检查数据长度是否足够
        if len(channel_data[0]) < 500:  # 确保数据长度至少为500
            return

        # 检查是否满足预测条件 - 使用通道2的数据在索引450到500进行峰值检测
        recent_data = channel_data[2][450:500]  # 固定提取索引450到500的数据

        # 使用scipy的find_peaks函数检测峰值
        peaks, _ = find_peaks(recent_data, height=PREDICTION_THRESHOLD, distance=20)

        if len(peaks) > 0:
            # 找到最高峰值
            peak_values = [recent_data[p] for p in peaks]
            max_peak_idx = peaks[np.argmax(peak_values)]

            # 计算峰值在完整数据中的绝对位置
            absolute_peak_idx = 450 + max_peak_idx

            # 避免重复检测同一个峰值
            if absolute_peak_idx != self.last_peak_idx:
                self.last_peak_idx = absolute_peak_idx

                # 设置峰值检测冷却时间，避免短时间内重复检测
                self.peak_detection_cooldown = 50  # 约0.5秒

                # 执行预测
                self.perform_prediction(channel_data, absolute_peak_idx)

    def perform_prediction(self, channel_data, peak_idx):
        """执行手势预测（修改后的版本）"""
        try:
            # 计算窗口起始和结束位置
            start_idx = peak_idx - 300
            end_idx = peak_idx + 300

            # 提取三个通道的数据
            all_channels_data = []
            for i in range(3):
                data_segment = channel_data[i][start_idx:end_idx]

                # 如果数据长度不足，用零填充
                if len(data_segment) < PREDICTION_WINDOW:
                    padding = np.zeros(PREDICTION_WINDOW - len(data_segment))
                    data_segment = np.concatenate([data_segment, padding])

                all_channels_data.append(data_segment)

            # 新增：数据预处理 - 去除直流偏移
            processed_channels_data = []
            for channel_data_segment in all_channels_data:
                # 计算前50个数据的平均值
                baseline = np.mean(channel_data_segment[:50]) if len(channel_data_segment) >= 50 else 0
                # 减去平均值以去除直流偏移
                processed_channel = channel_data_segment - baseline
                processed_channels_data.append(processed_channel)

            # 使用处理后的数据继续后续流程
            all_channels_data = processed_channels_data

            # 将三个通道的数据按顺序合并为一个长数组 (1800,)
            combined_data = np.concatenate(all_channels_data)

            # 对整个合并后的数据进行EMG处理
            processed_combined = self.emg_processor.process_signal(combined_data)

            # 使用MinMaxScaler对处理后的数据进行归一化
            scaler = MinMaxScaler(feature_range=(0, 1))
            processed_combined_2d = processed_combined.reshape(-1, 1)  # 转换为2D数组
            normalized_combined = scaler.fit_transform(processed_combined_2d).flatten()

            # 将归一化后的数据重新分割为三个通道
            normalized_channels = []
            for i in range(3):
                channel_start = i * PREDICTION_WINDOW
                channel_end = (i + 1) * PREDICTION_WINDOW
                normalized_channel = normalized_combined[channel_start:channel_end]

                # 如果长度不足，用零填充
                if len(normalized_channel) < PREDICTION_WINDOW:
                    padding = np.zeros(PREDICTION_WINDOW - len(normalized_channel))
                    normalized_channel = np.concatenate([normalized_channel, padding])

                normalized_channels.append(normalized_channel)

            # 转换为模型输入格式 (1, 3, 600)
            test_data_array = np.array(normalized_channels)
            model_input = test_data_array.reshape(1, 3, PREDICTION_WINDOW)
            model_input_tensor = torch.tensor(model_input, dtype=torch.float32)

            # 进行预测
            result = self.gesture_classifier.predict(model_input_tensor)

            if result:
                predicted, probabilities = result
                confidence = torch.max(probabilities).item()
                print(f"预测结果: {predicted.item()}, 置信度: {confidence:.3f}")
                print(f"各类别概率: {probabilities}")

                # 保存预测数据到Excel文件
                self.save_prediction_data(
                    all_channels_data,  # 原始数据
                    processed_combined,  # 处理后的数据
                    normalized_combined,  # 归一化后的数据
                    normalized_channels,  # 分割后的归一化数据
                    predicted.item(),  # 预测结果
                    confidence,  # 置信度
                    probabilities  # 各类别概率
                )

                # 只在高置信度时更新显示
                if confidence > 0.6:  # 置信度阈值
                    self.update_prediction_label(predicted, confidence)
                else:
                    print("置信度过低，不更新显示")

        except Exception as e:
            print(f"预测过程中出错: {e}")
            import traceback
            traceback.print_exc()
    def save_prediction_data(self, raw_data, processed_data, normalized_data, normalized_channels,
                             prediction, confidence, probabilities):
        """保存预测数据到Excel文件"""
        try:
            # 创建目录（如果不存在）
            save_dir = "F:\\cnn手臂信号\\预测数据"
            os.makedirs(save_dir, exist_ok=True)

            # 生成文件名（使用时间戳）
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"prediction_{timestamp}.xlsx"
            filepath = os.path.join(save_dir, filename)

            # 创建Excel写入器
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # 保存原始数据
                raw_df = pd.DataFrame({
                    'Channel_0_Raw': raw_data[0],
                    'Channel_1_Raw': raw_data[1],
                    'Channel_2_Raw': raw_data[2]
                })
                raw_df.to_excel(writer, sheet_name='Raw_Data', index=False)

                # 保存处理后的合并数据
                processed_df = pd.DataFrame({
                    'Processed_Combined': processed_data
                })
                processed_df.to_excel(writer, sheet_name='Processed_Data', index=False)

                # 保存归一化后的合并数据
                normalized_df = pd.DataFrame({
                    'Normalized_Combined': normalized_data
                })
                normalized_df.to_excel(writer, sheet_name='Normalized_Data', index=False)

                # 保存分割后的归一化数据
                normalized_channels_df = pd.DataFrame({
                    'Channel_0_Normalized': normalized_channels[0],
                    'Channel_1_Normalized': normalized_channels[1],
                    'Channel_2_Normalized': normalized_channels[2]
                })
                normalized_channels_df.to_excel(writer, sheet_name='Normalized_Channels', index=False)

                # 保存预测结果
                prediction_df = pd.DataFrame({
                    'Prediction': [prediction],
                    'Confidence': [confidence],
                    'Probability_0': [probabilities[0][0].item()],
                    'Probability_1': [probabilities[0][1].item()],
                    'Probability_2': [probabilities[0][2].item()],
                    'Timestamp': [timestamp]
                })
                prediction_df.to_excel(writer, sheet_name='Prediction_Result', index=False)

            print(f"预测数据已保存到: {filepath}")

        except Exception as e:
            print(f"保存预测数据时出错: {e}")
            import traceback
            traceback.print_exc()

    def update_prediction_label(self, predicted_label, confidence):
        """更新预测结果标签"""
        labels = ["石头", "剪刀", "布"]
        if 0 <= predicted_label.item() < len(labels):
            # 根据置信度设置文本颜色
            if confidence > 0.8:
                color = "green"
            elif confidence > 0.6:
                color = "orange"
            else:
                color = "red"

            self.label_8.setText(f"                                  {labels[predicted_label.item()]}")
            self.label_8.setStyleSheet(f"font-size: 50px; color: {color};")
            font = QFont()
            font.setFamily("Arial")
            font.setPointSize(30)
            self.label_8.setFont(font)

    def all(self):
        """全选/全不选所有通道"""
        is_checked = self.checkBox.isChecked()
        for i in range(1, 9):
            checkbox = getattr(self, f"checkBox_{i + 1}")
            checkbox.setChecked(is_checked)
            self.toggle_channel_visibility(i - 1, is_checked)

    def toggle_channel_visibility(self, channel_idx, is_checked=None):
        """切换通道可见性"""
        if is_checked is None:
            # 从复选框获取状态
            checkbox = getattr(self, f"checkBox_{channel_idx + 2}")
            is_checked = checkbox.isChecked()

        if is_checked:
            color = ['r', 'g', 'b', 'c', 'm', (145, 25, 52), (104, 103, 137), 'k'][channel_idx]
            pen = pg.mkPen(color=color, width=1)
            self.curve_refs[0][channel_idx] = self.p.plot(pen=pen)
            self.curve_refs[1][channel_idx] = self.p1.plot(pen=pen)
        else:
            if self.curve_refs[0][channel_idx]:
                self.p.removeItem(self.curve_refs[0][channel_idx])
                self.curve_refs[0][channel_idx] = None
            if self.curve_refs[1][channel_idx]:
                self.p1.removeItem(self.curve_refs[1][channel_idx])
                self.curve_refs[1][channel_idx] = None

    def show_error_message(self, message):
        """显示错误消息"""
        QMessageBox.critical(self, "错误", message)

    def show_warning_message(self, message):
        """显示警告消息"""
        QMessageBox.warning(self, "警告", message)

    def show_info_message(self, message):
        """显示信息消息"""
        QMessageBox.information(self, "信息", message)

    def closeEvent(self, event):
        """窗口关闭事件"""
        self.serial_close()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mywindow = MYWindow(app)
    mywindow.show()
    sys.exit(app.exec_())