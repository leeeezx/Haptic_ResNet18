'''
作为程序的入口。负责创建ROS2节点、订阅传感器数据、实现状态机逻辑，并在识别出接触阶段后，调用TerrainPredictor进行预测。
'''
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray # 假设传感器数据类型，请根据实际情况修改
from enum import Enum, auto
import os

# 从predictor.py中导入地形预测器
from predictor import TerrainPredictor

# 使用枚举类来清晰地定义状态
class State(Enum):
    WAITING_FOR_CONTACT = auto()  # 等待接触
    RECORDING = auto()            # 正在记录
    PROCESSING = auto()           # 处理数据并预测

class RealtimeRecognizer(Node):
    """
    实时地形识别器ROS2节点。

    订阅传感器数据，通过一个简单的状态机来检测和分割每一次与地形的接触过程。
    当一次完整的接触结束后，调用TerrainPredictor对收集到的数据进行分类，并打印结果。
    """
    def __init__(self):
        super().__init__('realtime_terrain_recognizer') # 继承自Node，设置节点名称

        # --- 1. 配置与初始化 ---
        # 定义模型和数据文件的路径
        # 请确保这些路径是正确的
        base_path = 'd:/CodeProject/haptic_ResNet'
        model_path = os.path.join(base_path, 'models/weights/best_model_weights-100epochs.pth')
        scaler_path = os.path.join(base_path, 'models/scalers/scalers-100epochs-train.pkl')
        mapping_path = os.path.join(base_path, 'data/terrain_mapping.json')
        max_length_path = os.path.join(base_path, 'data/max_length.json')

        # 实例化地形预测器
        self.get_logger().info("正在初始化地形预测器...")
        try:
            self.predictor = TerrainPredictor(model_path, scaler_path, mapping_path, max_length_path) # 将上面的文件加载到预测器中
            self.get_logger().info("地形预测器初始化成功。")
        except Exception as e:
            self.get_logger().error(f"地形预测器初始化失败: {e}")
            # 如果初始化失败，则关闭节点
            self.destroy_node()
            rclpy.shutdown()
            return

        # --- 2. 状态机和触发逻辑参数 ---
        self.state = State.WAITING_FOR_CONTACT
        self.contact_data = []  # 用于存储一次接触过程中的(位移, 力)数据

        # 触发参数 (这些值需要根据实验数据进行调整)
        self.force_threshold = -15       # 力阈值，用于判断是否发生接触
        self.contact_debounce_count = 3  # 连续N次力值超过阈值才确认为接触开始
        self.release_debounce_count = 5  # 连续M次力值低于阈值才确认为接触结束
        self._contact_counter = 0        # 内部计数器，用于接触开始的去抖动
        self._release_counter = 0        # 内部计数器，用于接触结束的去抖动

        # --- 3. ROS2订阅者 ---
        # 订阅传感器数据话题
        # !!!!!!! 重要: 请将 '/your_robot/sensor_topic' 替换为实际的话题名称 !!!!!!!
        # !!!!!!! Float64MultiArray也可能需要换成你实际使用的消息类型 !!!!!!!
        self.subscription = self.create_subscription(
            Float64MultiArray, # 使用的消息类型
            '/realtime_robot_poseAndextTau', # topic名称
            self.listener_callback,
            10) # QoS Profile
        self.get_logger().info(f'已订阅话题 "/realtime_robot_poseAndextTau"，状态: {self.state.name}')

    def listener_callback(self, msg):
        """
        每次接收到传感器数据时被调用的回调函数。
        状态机的主要逻辑在这里实现。
        """
        # !!!!!!! 重要: 假设位移是msg.data的第3个元素(索引2)，力是第9个(索引8) !!!!!!!
        # !!!!!!! 请根据你的实际消息格式进行修改 !!!!!!!
        try:
            displacement = msg.data[2]
            force = msg.data[8]
        except IndexError:
            self.get_logger().warn("接收到的消息格式不符合预期，忽略此消息。")
            return

        # --- 状态机逻辑 ---
        # 状态一: 等待接触
        if self.state == State.WAITING_FOR_CONTACT:
            if force > self.force_threshold: # 检查是否需要开始记录
                self._contact_counter += 1
                if self._contact_counter >= self.contact_debounce_count:
                    # 确认接触开始，切换到RECORDING状态
                    self.state = State.RECORDING
                    self.contact_data.clear() # 清空旧数据
                    self.contact_data.append((displacement, force))
                    self._contact_counter = 0 # 重置计数器
                    self.get_logger().info("接触开始，切换到 [RECORDING] 状态...")
            else:
                self._contact_counter = 0 # 如果力值回落，重置计数器

        # 状态二: 正在记录数据
        elif self.state == State.RECORDING:
            self.contact_data.append((displacement, force))

            if force < self.force_threshold: # 检查是否需要结束记录
                self._release_counter += 1
                if self._release_counter >= self.release_debounce_count:
                    # 确认接触结束，切换到PROCESSING状态
                    self.get_logger().info(f"接触结束，共记录 {len(self.contact_data)} 个数据点。切换到 [PROCESSING] 状态...")
                    self.state = State.PROCESSING
                    self._release_counter = 0 # 重置计数器
                    
                    # 进入处理状态后，立即执行预测
                    self.process_and_predict()
            else:
                self._release_counter = 0 # 如果力值回升，重置计数器

    def process_and_predict(self):
        """
        调用预测器进行预测，并处理结果。
        """
        if not self.contact_data:
            self.get_logger().warn("没有采集到数据，无法预测。")
        else:
            try:
                # 调用预测器进行预测
                predicted_terrain = self.predictor.predict(self.contact_data)
                self.get_logger().info(f'********** 预测结果: {predicted_terrain} **********')
            except Exception as e:
                self.get_logger().error(f"预测过程中发生错误: {e}")

        # 预测完成后，无论成功与否，都清空数据并返回WAITING状态
        self.contact_data.clear()
        self.state = State.WAITING_FOR_CONTACT
        self.get_logger().info(f"已重置状态，返回 [WAITING_FOR_CONTACT]...")


def main(args=None):
    rclpy.init(args=args)
    
    realtime_recognizer = RealtimeRecognizer()
    
    if realtime_recognizer.handle: # handle是节点的内部引用，如果节点在初始化时失败，realtime_recognizer.handle将为None，则不进入spin循环
        try:
            rclpy.spin(realtime_recognizer)
        except KeyboardInterrupt: # 捕获Ctrl+C中断信号，pass表示优雅地忽略中断，让程序进入finally块
            pass
        finally:
            # 销毁节点并关闭rclpy
            realtime_recognizer.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()