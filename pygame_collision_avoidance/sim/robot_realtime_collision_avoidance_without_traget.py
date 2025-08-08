import pygame
import math
import torch
import os
import numpy as np
import random

# 添加map.py中的地图创建函数
from map import create_map

pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("智能避障机器人模拟")
clock = pygame.time.Clock()

# 常量定义
ROBOT_RADIUS = 10
SENSOR_LENGTH = 100
SENSOR_COUNT = 32
VELOCITY = 3.0
MAP_VERSION = "v12"  # 使用v12地图，可自行修改为其他版本
FPS = 60

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

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
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
model_enhanced_path = os.path.join(parent_dir, "model", "model_enhanced.pth")
model = Net()
model.load_state_dict(torch.load(model_enhanced_path))
model.eval()

# 机器人类定义
class Robot:
    def __init__(self, x, y, obstacles):
        self.x = x
        self.y = y
        self.angle = random.uniform(0, 360)  # 初始随机方向
        self.velocity = VELOCITY
        self.obstacles = obstacles
        self.sensor_data = []
        self.prediction = 0  # 0: 安全, 1: 碰撞
        self.collision_time = 0  # 用于碰撞后短暂显示红色
        self.is_colliding = False
        
    def update(self):
        # 获取传感器数据
        self.sensor_data = self.get_sensor_distances()
        
        # 使用神经网络预测是否会碰撞
        input_tensor = torch.tensor(self.sensor_data, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            self.prediction = torch.argmax(output).item()
        
        # 如果预测会碰撞，则改变方向（反弹）
        if self.prediction == 1:
            self.angle = (self.angle + 180 + random.uniform(-45, 45)) % 360
            self.is_colliding = True
            self.collision_time = 1  # 碰撞显示持续20帧
        
        # 更新位置
        angle_rad = math.radians(self.angle)
        new_x = self.x + self.velocity * math.cos(angle_rad)
        new_y = self.y + self.velocity * math.sin(angle_rad)
        
        # 边界检查
        if new_x < ROBOT_RADIUS or new_x > WIDTH - ROBOT_RADIUS:
            self.angle = (180 - self.angle) % 360
            return
        if new_y < ROBOT_RADIUS or new_y > HEIGHT - ROBOT_RADIUS:
            self.angle = (360 - self.angle) % 360
            return
            
        # 更新位置
        self.x = new_x
        self.y = new_y
        
        # 更新碰撞显示计时器
        if self.collision_time > 0:
            self.collision_time -= 1
        else:
            self.is_colliding = False

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
    
    def draw(self, screen):
        # 绘制机器人主体
        color = RED if self.is_colliding else (BLUE if self.prediction == 0 else YELLOW)
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), ROBOT_RADIUS)
        
        # 绘制方向指示线
        angle_rad = math.radians(self.angle)
        end_x = self.x + ROBOT_RADIUS * 1.5 * math.cos(angle_rad)
        end_y = self.y + ROBOT_RADIUS * 1.5 * math.sin(angle_rad)
        pygame.draw.line(screen, BLACK, (self.x, self.y), (end_x, end_y), 2)
        
        # 绘制传感器线
        if self.sensor_data:
            for i, dist in enumerate(self.sensor_data):
                a = self.angle + i * (360/SENSOR_COUNT)
                a_rad = math.radians(a)
                end_x = self.x + dist * math.cos(a_rad)
                end_y = self.y + dist * math.sin(a_rad)
                color = GREEN
                pygame.draw.line(screen, color, (self.x, self.y), (end_x, end_y), 1)

def main():
    # 创建地图
    obstacles = create_map(version=MAP_VERSION)
    
    # 创建机器人
    # 尝试在没有障碍物的位置初始化机器人
    robot_created = False
    while not robot_created:
        x = random.randint(ROBOT_RADIUS * 2, WIDTH - ROBOT_RADIUS * 2)
        y = random.randint(ROBOT_RADIUS * 2, HEIGHT - ROBOT_RADIUS * 2)
        
        # 检查是否与障碍物重叠
        overlap = False
        for obs in obstacles:
            if obs.collidepoint(x, y) or (
                x + ROBOT_RADIUS > obs.left and 
                x - ROBOT_RADIUS < obs.left + obs.width and
                y + ROBOT_RADIUS > obs.top and 
                y - ROBOT_RADIUS < obs.top + obs.height
            ):
                overlap = True
                break
        
        if not overlap:
            robot = Robot(x, y, obstacles)
            robot_created = True
    
    # 主循环
    running = True
    paused = False
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:  # 重置机器人位置
                    main()
                    return
        
        if not paused:
            # 更新机器人
            robot.update()
        
        # 绘制
        screen.fill(WHITE)
        
        # 绘制障碍物
        for obs in obstacles:
            pygame.draw.rect(screen, BLACK, obs)
        
        # 绘制机器人
        robot.draw(screen)
        
        # 显示操作说明
        font = pygame.font.SysFont(None, 24)
        instructions = [
            f"MAP: {MAP_VERSION}",
                "SPACE: PAUSE/PLAY",
                "R: RESET",
            f"PRED: {'COLL' if robot.prediction == 1 else 'SAFE'}"
        ]
        for i, text in enumerate(instructions):
            img = font.render(text, True, BLACK)
            screen.blit(img, (10, 10 + i*25))
       
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()
    pygame.quit()

