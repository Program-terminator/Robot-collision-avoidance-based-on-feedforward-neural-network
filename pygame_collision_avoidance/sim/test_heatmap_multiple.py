import pygame
import math
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

# 添加map.py中的地图创建函数
from map import create_map

pygame.init()
WIDTH, HEIGHT = 800, 600

ROBOT_RADIUS = 15
SENSOR_LENGTH = 100
SENSOR_COUNT = 32
VELOCITY = 3.0

# 神经网络结构（与训练时一致）
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.fc(x)

# 加载增强模型
model_path = os.path.join("..", "model", "model_enhanced.pth")
model = Net()
model.load_state_dict(torch.load(model_path))
model.eval()

# 创建用于存储预测结果的网格
GRID_SIZE = 10
x_points = np.arange(0, WIDTH, GRID_SIZE)
y_points = np.arange(0, HEIGHT, GRID_SIZE)

# 定义机器人类
class Robot:
    def __init__(self, x, y, obstacles):
        self.x = x
        self.y = y
        self.angle = 0
        self.obstacles = obstacles

    def get_sensor_distances(self):
        angles = [self.angle + i * (360/SENSOR_COUNT) for i in range(SENSOR_COUNT)]
        distances = []
        for a in angles:
            a_rad = math.radians(a)
            for dist in range(0, SENSOR_LENGTH, 2):
                sx = self.x + dist * math.cos(a_rad)
                sy = self.y + dist * math.sin(a_rad)
                if sx < 0 or sx >= WIDTH or sy < 0 or sy >= HEIGHT:
                    distances.append(dist)
                    break
                hit = False
                for obs in self.obstacles:
                    if obs.collidepoint(sx, sy):
                        distances.append(dist)
                        hit = True
                        break
                if hit:
                    break
            else:
                distances.append(SENSOR_LENGTH)
        return distances

# 测试地图10-14
test_maps = [f"v{i}" for i in range(10, 15)]

for map_version in test_maps:
    print(f"Generating heatmap for map version: {map_version}")
    
    # 使用当前地图
    obstacles = create_map(version=map_version)
    
    # 初始化结果网格
    prediction_grid = np.zeros((len(y_points), len(x_points)))
    
    # 遍历网格中的每个点
    for i, y in enumerate(y_points):
        for j, x in enumerate(x_points):
            # 检查点是否在障碍物内部
            point_in_obstacle = False
            for obs in obstacles:
                if obs.collidepoint(x, y):
                    point_in_obstacle = True
                    break
            
            # 如果点在障碍物内部，标记为碰撞
            if point_in_obstacle:
                prediction_grid[i, j] = 1  # 用1表示障碍物区域
                continue
            
            # 创建临时机器人以获取传感器数据
            temp_robot = Robot(x, y, obstacles)
            distances = temp_robot.get_sensor_distances()
            
            # 使用神经网络进行预测
            input_tensor = torch.tensor(distances, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                pred = torch.argmax(output).item()
            
            # 存储预测结果
            prediction_grid[i, j] = pred
    
    # 生成热力图
    plt.figure(figsize=(12, 8))
    
    # 绘制障碍物
    for obs in obstacles:
        rect = plt.Rectangle((obs.left, obs.top), obs.width, obs.height, 
                             facecolor='gray', edgecolor='black', alpha=0.7)
        plt.gca().add_patch(rect)
    
    # 绘制预测结果热力图
    heatmap = plt.imshow(prediction_grid, cmap='coolwarm', alpha=0.6, 
                         extent=[0, WIDTH, HEIGHT, 0], origin='upper', 
                         vmin=0, vmax=1)
    
    # 添加颜色条
    cbar = plt.colorbar(heatmap)
    cbar.set_label('Collision Prediction (0: Safe, 1: Collision)', rotation=270, labelpad=20)
    
    # 设置图表标题和标签
    plt.title(f'Robot Collision Prediction Heatmap - Map {map_version}')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    
    # 保存图像
    filename = f'../data/predict/collision_prediction_heatmap_{map_version}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭当前图形以释放内存
    
    print(f"Heatmap saved as {filename}")

print("All heatmaps generated successfully!")