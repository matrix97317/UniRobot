# -*- coding: utf-8 -*-
"""Visualize HDF5 file."""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import time
from typing import Dict, Any
import h5py 
ACTION_DIM=6
class DataVisualizer:
    def __init__(self, data: Dict[str, Any], fps: int = 30):
        """
        初始化可视化器
        
        Args:
            data: 包含 top_img, hand_img, action 的字典
            fps: 播放帧率
        """
        self.data = data
        self.fps = fps
        self.current_frame = 0
        self.total_frames = data['top_img'].shape[0]
        self.paused = False
        
        # 验证数据形状
        self._validate_data()
        
        # 创建窗口
        cv2.namedWindow('Data Visualizer', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Data Visualizer', 1600, 900)
        
        # 初始化matplotlib图形
        self._init_plots()
        
        # 设置统一的图像宽度
        self.target_width = 320
        
    def _validate_data(self):
        """验证数据形状一致性"""
        assert self.data['top_img'].shape[0] == self.data['hand_img'].shape[0] == self.data['action'].shape[0], \
            "帧数不匹配"
        assert self.data['action'].shape[1] == ACTION_DIM, "action维度应为14"
        
    def _init_plots(self):
        """初始化matplotlib图形"""
        self.fig, self.axs = plt.subplots(4, 4, figsize=(12, 8))
        self.fig.tight_layout(pad=3.0)
        self.canvas = FigureCanvasAgg(self.fig)
        
        # 初始化14个动作维度的曲线
        self.lines = []
        action_data = self.data['action']
        
        for i in range(ACTION_DIM):
            row, col = i // 4, i % 4
            ax = self.axs[row, col]
            line, = ax.plot(action_data[:1, i], 'b-')  # 只画第一帧
            ax.set_title(f'Action Dim {i}')
            ax.set_ylim(action_data.min() - 0.1, action_data.max() + 0.1)
            ax.grid(True)
            self.lines.append(line)
        
    def _update_plots(self, frame_idx: int):
        """更新动作曲线图"""
        action_data = self.data['action']
        
        for i in range(ACTION_DIM):
            # 更新曲线数据（显示从开始到当前帧）
            self.lines[i].set_data(range(frame_idx + 1), action_data[:frame_idx + 1, i])
            
            # 自动调整x轴范围
            self.axs[i // 4, i % 4].set_xlim(0, max(10, frame_idx + 1))
        
        # 重绘图形
        self.canvas.draw()
        
    def _matplotlib_to_opencv(self):
        """将matplotlib图形转换为OpenCV图像"""
        # 从canvas获取RGB图像
        buf = np.frombuffer(self.canvas.buffer_rgba(), dtype=np.uint8)
        width, height = self.canvas.get_width_height()
        buf = buf.reshape((height, width, 4))
        
        # 转换为BGR（OpenCV格式）并去除alpha通道
        plot_img = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
        return plot_img
    
    def _resize_to_width(self, image, target_width):
        """调整图像到目标宽度，保持宽高比"""
        if len(image.shape) == 3:
            h, w, _ = image.shape
        else:
            h, w = image.shape
            
        scale_factor = target_width / w
        new_height = int(h * scale_factor)
        return cv2.resize(image, (target_width, new_height))
    
    def _create_info_panel(self, frame_idx: int, width: int):
        """创建信息面板"""
        height = 200
        panel = np.ones((height, width, 3), dtype=np.uint8) * 50  # 深灰色背景
        
        # 添加文本信息
        texts = [
            f"Frame: {frame_idx}/{self.total_frames - 1}",
            f"Status: {'PAUSED' if self.paused else 'PLAYING'}",
            f"Top Image: {self.data['top_img'].shape}",
            f"Hand Image: {self.data['hand_img'].shape}",
            f"Action: {self.data['action'].shape}",
            "Controls:",
            "SPACE - Pause/Resume",
            "LEFT/RIGHT - Navigate",
            "UP/DOWN - Change FPS",
            "ESC - Quit"
        ]
        
        for i, text in enumerate(texts):
            y_pos = 30 + i * 20
            cv2.putText(panel, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return panel
    
    def _process_keyboard(self):
        """处理键盘输入"""
        key = cv2.waitKey(1000 // self.fps) & 0xFF
        
        if key == 27:  # ESC
            return False
        
        elif key == ord(' '):  # 空格键
            self.paused = not self.paused
            print(f"{'Paused' if self.paused else 'Resumed'}")
        
        elif key == ord('a') or key == 81:  # 左箭头或A键
            self.current_frame = max(0, self.current_frame - 1)
            self.paused = True
        
        elif key == ord('d') or key == 83:  # 右箭头或D键
            self.current_frame = min(self.total_frames - 1, self.current_frame + 1)
            self.paused = True
        
        elif key == ord('r'):  # R键重置
            self.current_frame = 0
            self.paused = False
        
        elif key == ord('+'):  # 增加FPS
            self.fps = min(60, self.fps + 5)
            print(f"FPS: {self.fps}")
        
        elif key == ord('-'):  # 减少FPS
            self.fps = max(1, self.fps - 5)
            print(f"FPS: {self.fps}")
        
        return True
    
    def visualize(self):
        """主可视化循环"""
        print("开始可视化...")
        print("控制指令:")
        print("SPACE - 暂停/继续")
        print("LEFT/RIGHT/A/D - 前后导航")
        print("+/- - 增加/减少FPS")
        print("R - 重置到开始")
        print("ESC - 退出")
        
        while True:
            # 处理键盘输入
            if not self._process_keyboard():
                break
            
            # 如果不是暂停状态，自动前进
            if not self.paused:
                self.current_frame = (self.current_frame + 1) % self.total_frames
            
            # 获取当前帧数据
            top_img = self.data['top_img'][self.current_frame]
            hand_img = self.data['hand_img'][self.current_frame]
            action_values = self.data['action'][self.current_frame]
            
            # 确保图像是uint8类型（如果是float类型）
            if top_img.dtype != np.uint8:
                top_img = (top_img * 255).astype(np.uint8)
            if hand_img.dtype != np.uint8:
                hand_img = (hand_img * 255).astype(np.uint8)
            
            # 如果图像是灰度图，转换为BGR
            if len(top_img.shape) == 2:
                top_img = cv2.cvtColor(top_img, cv2.COLOR_GRAY2BGR)
            if len(hand_img.shape) == 2:
                hand_img = cv2.cvtColor(hand_img, cv2.COLOR_GRAY2BGR)
            
            # 调整图像大小到统一宽度
            top_img = self._resize_to_width(top_img, self.target_width)
            hand_img = self._resize_to_width(hand_img, self.target_width)
            
            # 确保两个图像高度相同（如果不问，填充到相同高度）
            max_height = max(top_img.shape[0], hand_img.shape[0])
            if top_img.shape[0] < max_height:
                padding = np.zeros((max_height - top_img.shape[0], top_img.shape[1], 3), dtype=np.uint8)
                top_img = np.vstack([top_img, padding])
            if hand_img.shape[0] < max_height:
                padding = np.zeros((max_height - hand_img.shape[0], hand_img.shape[1], 3), dtype=np.uint8)
                hand_img = np.vstack([hand_img, padding])
            
            # 更新曲线图
            self._update_plots(self.current_frame)
            plot_img = self._matplotlib_to_opencv()
            
            # 调整曲线图宽度与图像一致
            plot_target_width = self.target_width * 2  # 两个图像的宽度之和
            plot_img = self._resize_to_width(plot_img, plot_target_width)
            
            # 创建信息面板（宽度与曲线图一致）
            info_panel = self._create_info_panel(self.current_frame, plot_target_width)
            
            # 在图像上显示当前动作值
            top_img_with_text = top_img.copy()
            hand_img_with_text = hand_img.copy()
            
            cv2.putText(top_img_with_text, "Top View", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(hand_img_with_text, "Hand View", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 组合所有图像（确保宽度一致）
            top_row = np.hstack([top_img_with_text, hand_img_with_text])
            
            # 确保所有组件的宽度一致
            if top_row.shape[1] != plot_img.shape[1]:
                plot_img = cv2.resize(plot_img, (top_row.shape[1], plot_img.shape[0]))
            
            if info_panel.shape[1] != top_row.shape[1]:
                info_panel = cv2.resize(info_panel, (top_row.shape[1], info_panel.shape[0]))
            
            # 垂直堆叠
            combined = np.vstack([top_row, plot_img, info_panel])
            
            # 显示
            cv2.imshow('Data Visualizer', combined)
        
        cv2.destroyAllWindows()
        plt.close('all')

# 示例数据生成函数（用于测试）
def create_sample_data(num_frames=100):
    """创建示例数据用于测试"""
    # 生成不同尺寸的图像数据（模拟实际情况）
    # top_img = np.random.rand(num_frames, 240, 320, 3).astype(np.float32)  # 240x320
    # hand_img = np.random.rand(num_frames, 180, 240, 3).astype(np.float32)  # 180x240
    
    # # 生成动作数据（14维，模拟正弦波）
    # action = np.zeros((num_frames, 14))
    # for i in range(14):
    #     frequency = 0.1 * (i + 1)
    #     action[:, i] = np.sin(2 * np.pi * frequency * np.linspace(0, 1, num_frames)) + np.random.normal(0, 0.1, num_frames)
    
    # return {
    #     'top_img': top_img,
    #     'hand_img': hand_img,
    #     'action': action
    # }
    top_img=None
    hand_img=None
    action=None
    with h5py.File("./episode_0001.hdf5","r") as fin:
        top_img = fin["top_imgs"][()]
        hand_img = fin["hand_imgs"][()]
        action = fin["action"][()]
        print(top_img.shape)
        print(hand_img.shape)
        print(action.shape)
    return {
        'top_img': top_img[()],
        'hand_img': hand_img[()],
        'action': action[()]
    }

if __name__ == "__main__":
    sample_data = create_sample_data(200)
    visualizer = DataVisualizer(sample_data, fps=100)
    visualizer.visualize()