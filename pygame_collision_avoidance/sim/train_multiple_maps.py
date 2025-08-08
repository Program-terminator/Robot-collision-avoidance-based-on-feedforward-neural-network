import pygame
import math
import csv
import os
import random

# 添加map.py中的地图创建函数
from map import create_map, get_all_map_versions, random_map

pygame.init()
WIDTH, HEIGHT = 800, 600

ROBOT_RADIUS = 15
SENSOR_LENGTH = 100
SENSOR_COUNT = 32
# 加快机器人移动速度
VELOCITY = 3.0

class Robot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0

    def move(self):
        self.x += VELOCITY * math.cos(math.radians(self.angle))
        self.y += VELOCITY * math.sin(math.radians(self.angle))

    def rotate(self, delta):
        self.angle = (self.angle + delta) % 360

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
                for obs in obstacles:
                    if obs.collidepoint(sx, sy):
                        distances.append(dist)
                        hit = True
                        break
                if hit:
                    break
            else:
                distances.append(SENSOR_LENGTH)
        return distances

    def draw(self, win):
        pygame.draw.circle(win, (0, 255, 0), (int(self.x), int(self.y)), ROBOT_RADIUS)
        pygame.draw.circle(win, (0, 255, 0), (int(self.x), int(self.y)), ROBOT_RADIUS)

def is_imminent_collision(distances, threshold=20):
    # 对于32个传感器，检查前几个传感器是否有障碍物
    # 考虑机器人半径，提前检测碰撞
    # 训练时设置较小的距离阈值作为撞墙判断标准
    return int(any(d < threshold for d in distances[:3]))

# 选择5个地图版本进行训练
map_versions = ["v1", "v2", "v3", "v4", "v5"]

data_dir = os.path.join("..", "data")
os.makedirs(data_dir, exist_ok=True)

# 为每个地图生成数据
for map_version in map_versions:
    print(f"Generating data for map version: {map_version}")
    obstacles = create_map(map_version)
    
    # 为每个地图创建单独的CSV文件
    csv_path = os.path.join(data_dir, f"sensor_data_{map_version}.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([f"sensor_{i}" for i in range(SENSOR_COUNT)] + ["label"])
    
    # 设置机器人初始位置在左上角
    robot = Robot(ROBOT_RADIUS + 10, ROBOT_RADIUS + 10)
    
    clock = pygame.time.Clock()
    frame_count = 0
    
    # 增加每个地图的训练数据量
    while frame_count < 5000:  # 增加到5000帧以获得更多数据
        clock.tick(60)
        
        # 使用完整的随机路径方法
        if random.random() < 0.3:  # 30%概率改变方向
            robot.rotate(random.choice([-45, -30, -15, 0, 15, 30, 45]))
        robot.move()
        
        # 检查是否撞墙或撞到障碍物
        hit_wall = False
        if robot.x < ROBOT_RADIUS or robot.x > WIDTH - ROBOT_RADIUS or robot.y < ROBOT_RADIUS or robot.y > HEIGHT - ROBOT_RADIUS:
            hit_wall = True
            robot.rotate(90)
        
        # 检查是否撞到障碍物
        hit_obstacle = False
        robot_rect = pygame.Rect(robot.x - ROBOT_RADIUS, robot.y - ROBOT_RADIUS, 
                                ROBOT_RADIUS * 2, ROBOT_RADIUS * 2)
        for obs in obstacles:
            if robot_rect.colliderect(obs):
                hit_obstacle = True
                break
        
        # 将撞墙和撞障碍物同等处理
        collision = hit_wall or hit_obstacle
        
        distances = robot.get_sensor_distances()
        # 标记为碰撞
        label = 1 if collision else is_imminent_collision(distances)
        writer.writerow(distances + [label])
        
        # 如果发生碰撞，不继续执行后续代码
        if collision:
            continue
        frame_count += 1
    
    csv_file.close()
    print(f"Data saved to {csv_path}")

pygame.quit()