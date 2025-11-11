# realtime_recongnizer文件中的注意事项

**最大的注意事项：需要将机械臂静态发布者的受力坐标系修改，手动添加负号，注意分辨静态发布者和动态发布时的坐标系区别**
 
1. model_path = os.path.join(base_path, 'models/weights/best_model_weights-100epochs.pth')
2. 触发参数
3. 机械臂发布者使用的名称与消息类型。'/realtime_robot_poseAndextTau'，消息类型？

## 实验时注意
为确定参数，需要进行实验，`参数确定实验`时需要关注的数据：

1. 机械臂移动中，还未接触地面介质时，受力大小  
-> self.force_threshold = -15       # 力阈值，用于判断是否发生接触
2. 


## 其他注意事项
1. 如果订阅者频率强制和发布者一样（1000hz），是否跟得上处理？