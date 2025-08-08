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
MAP_VERSION = "v12"   # 使用v12地图，可自行修改为其他版本
FPS = 60
TARGET_RADIUS = 10  # 目标点半径
ARRIVAL_DISTANCE = 20  # 认为到达目标的距离阈值

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)  # 紧急返回状态颜色
PURPLE = (128, 0, 128)  # 目标点颜色

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
# 使用绝对路径确保能找到模型文件
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
model_enhanced_path = os.path.join(parent_dir, "model", "model_enhanced.pth")

print(f"当前目录: {current_dir}")
print(f"父目录: {parent_dir}")
print(f"尝试加载增强模型: {model_enhanced_path}")

# 加载模型
model = Net()
try:
    model.load_state_dict(torch.load(model_enhanced_path))
    print("成功加载增强模型!")
except Exception as e:
    print(f"加载模型时发生错误: {str(e)}")
    raise  # 重新抛出异常，终止程序
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
        self.target_x = None  # 目标点x坐标
        self.target_y = None  # 目标点y坐标
        
        # 动态路径缓存系统
        self.path_cache = []  # 短期路径缓存，用于紧急回退
        self.max_cache_size = 25  # 最大缓存步数
        self.path_history = []  # 长期路径历史，用于可视化
        self.max_history = 50  # 最大历史路径点数量
        
        # 碰撞鲁棒性增强
        self.collision_counter = 0  # 连续检测到碰撞的次数
        self.collision_threshold = 3  # 连续几次检测到碰撞触发强制反弹
        self.is_emergency_return = False  # 是否处于紧急返回状态
        self.current_retreat_steps = 0  # 当前回退步数
        self.retreat_attempts = 0  # 回退尝试次数
        self.max_retreat_attempts = 5  # 最大回退尝试次数
        
    def set_target(self, x, y):
        # 检查目标点是否在障碍物内
        for obs in self.obstacles:
            if obs.collidepoint(x, y):
                return False  # 如果在障碍物内，返回False
        
        self.target_x = x
        self.target_y = y
        return True  # 设置成功返回True
        
    def update(self):
        # 更新路径缓存和历史
        current_pos = (self.x, self.y)
        
        # 添加当前位置到路径历史（用于可视化）
        self.path_history.append(current_pos)
        if len(self.path_history) > self.max_history:
            self.path_history.pop(0)
            
        # 添加当前位置到路径缓存（用于回退）
        if not self.is_emergency_return:  # 只在非紧急状态下记录缓存
            self.path_cache.append(current_pos)
            if len(self.path_cache) > self.max_cache_size:
                self.path_cache.pop(0)
            
        # 获取传感器数据
        self.sensor_data = self.get_sensor_distances()
        
        # 使用神经网络预测是否会碰撞
        input_tensor = torch.tensor(self.sensor_data, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            self.prediction = torch.argmax(output).item()
        
        # 碰撞计数逻辑
        if self.prediction == 1:
            self.collision_counter += 1
            self.is_colliding = True
            self.collision_time = 10
        else:
            # 只有在非紧急状态下才重置碰撞计数器
            if not self.is_emergency_return:
                self.collision_counter = 0
            
        # 检查是否触发紧急返回（连续三次检测到碰撞）
        if self.collision_counter >= self.collision_threshold and not self.is_emergency_return:
            # 如果触发了紧急返回，设置第一次回退步数
            self.is_emergency_return = True
            self.retreat_attempts = 1
            self.current_retreat_steps = min(4, len(self.path_cache) - 1)  # 第一次回退4步
            print(f"触发紧急返回! 连续检测到碰撞 {self.collision_counter} 次，尝试回退 {self.current_retreat_steps} 步")
            # 立即调整角度进行反弹（反方向）
            self.angle = (self.angle + 180) % 360
            
        # 如果正在进行紧急返回
        if self.is_emergency_return and self.current_retreat_steps > 0:
            # 使用路径缓存中的点作为临时目标
            retreat_index = max(0, len(self.path_cache) - self.current_retreat_steps)
            if retreat_index < len(self.path_cache):
                retreat_point = self.path_cache[retreat_index]
                
                # 朝向返回点移动
                return_angle = math.degrees(math.atan2(retreat_point[1] - self.y, retreat_point[0] - self.x)) % 360
                
                # 计算当前角度和返回角度之间的差异
                angle_diff = (return_angle - self.angle) % 360
                if angle_diff > 180:
                    angle_diff -= 360
                
                # 调整角度，快速转向返回路径
                turn_rate = 10  # 增加转向速度
                if abs(angle_diff) <= turn_rate:
                    self.angle = return_angle
                elif angle_diff > 0:
                    self.angle = (self.angle + turn_rate) % 360
                else:
                    self.angle = (self.angle - turn_rate) % 360
                    
                # 检查是否到达返回点
                distance_to_return = math.sqrt((self.x - retreat_point[0])**2 + (self.y - retreat_point[1])**2)
                if distance_to_return <= ARRIVAL_DISTANCE:
                    # 到达当前返回点
                    self.current_retreat_steps = 0
                    
                    # 检查到达后是否仍然处于危险状态
                    if self.prediction == 1:
                        # 如果仍然危险，并且没有超过最大尝试次数，尝试更深的回退
                        if self.retreat_attempts < self.max_retreat_attempts:
                            self.retreat_attempts += 1
                            # 每次增加回退步数
                            new_steps = min(4 * self.retreat_attempts, len(self.path_cache) - 1)   
                            self.current_retreat_steps = new_steps
                            print(f"仍处于危险，第 {self.retreat_attempts} 次尝试，回退 {self.current_retreat_steps} 步")
                        else:
                            # 达到最大尝试次数，放弃回退策略，强制大角度转向并尝试随机移动
                            print("达到最大回退尝试次数，尝试随机移动")
                            self.is_emergency_return = False
                            self.collision_counter = 0
                            self.angle = (self.angle + 120 + random.uniform(-60, 60)) % 360
                    else:
                        # 安全了，解除紧急状态
                        print("回退成功，已脱离危险")
                        self.is_emergency_return = False
                        self.collision_counter = 0
            else:
                # 如果索引超出范围，结束紧急状态
                self.is_emergency_return = False
                self.collision_counter = 0
                
        # 正常导航逻辑
        elif not self.is_emergency_return:
            # 如果有目标点，朝向目标方向移动
            if self.target_x is not None and self.target_y is not None:
                # 计算到目标的角度
                target_angle = math.degrees(math.atan2(self.target_y - self.y, self.target_x - self.x)) % 360
                
                # 检查是否到达目标
                distance_to_target = math.sqrt((self.x - self.target_x)**2 + (self.y - self.target_y)**2)
                if distance_to_target <= ARRIVAL_DISTANCE:
                    return  # 到达目标点，停止移动
                
                # 如果预测会碰撞，则暂时改变方向避开障碍物
                if self.prediction == 1:
                    # 计算暂时偏离的角度
                    self.angle = (self.angle + 90 + random.uniform(-45, 45)) % 360
                else:
                    # 如果安全，朝向目标方向调整
                    # 计算当前角度和目标角度之间的差异
                    angle_diff = (target_angle - self.angle) % 360
                    if angle_diff > 180:
                        angle_diff -= 360
                    
                    # 逐渐调整角度，模拟平滑转向
                    turn_rate = 5  # 每帧最大转向角度
                    if abs(angle_diff) <= turn_rate:
                        self.angle = target_angle
                    elif angle_diff > 0:
                        self.angle = (self.angle + turn_rate) % 360
                    else:
                        self.angle = (self.angle - turn_rate) % 360
            else:
                # 如果没有目标点但预测会碰撞，则改变方向（反弹）
                if self.prediction == 1:
                    self.angle = (self.angle + 180 + random.uniform(-45, 45)) % 360
        
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
        # 绘制机器人长期路径历史
        if len(self.path_history) > 1:
            pygame.draw.lines(screen, (200, 200, 255), False, self.path_history, 2)
            
        # 绘制缓存路径 - 使用不同颜色区分
        if len(self.path_cache) > 1:
            # 绘制所有缓存点（小点）
            for i, point in enumerate(self.path_cache):
                # 渐变颜色，越新的点越亮
                intensity = int(155 + (100 * i / len(self.path_cache)))
                color = (0, intensity, 0)  # 绿色渐变
                pygame.draw.circle(screen, color, (int(point[0]), int(point[1])), 2)
                
            # 将缓存点连线
            pygame.draw.lines(screen, (100, 255, 100), False, self.path_cache, 1)
            
        # 绘制目标点(如果有)
        if self.target_x is not None and self.target_y is not None:
            pygame.draw.circle(screen, PURPLE, (int(self.target_x), int(self.target_y)), TARGET_RADIUS)
            
            # 绘制从机器人到目标点的线
            pygame.draw.line(screen, (200, 0, 200), (self.x, self.y), 
                            (self.target_x, self.target_y), 1)
                            
        # 如果处于紧急返回状态，绘制返回路径
        if self.is_emergency_return and self.current_retreat_steps > 0 and len(self.path_cache) > 0:
            retreat_index = max(0, len(self.path_cache) - self.current_retreat_steps)
            if retreat_index < len(self.path_cache):
                retreat_point = self.path_cache[retreat_index]
                
                # 绘制返回目标点
                pygame.draw.circle(screen, ORANGE, (int(retreat_point[0]), int(retreat_point[1])), 8)
                
                # 绘制从机器人到返回点的线
                pygame.draw.line(screen, ORANGE, (self.x, self.y), retreat_point, 2)
                
                # 添加回退尝试信息
                font = pygame.font.SysFont(None, 16)
                attempt_text = font.render(f"Attempt: {self.retreat_attempts}/{self.max_retreat_attempts}", True, (255, 165, 0))
                screen.blit(attempt_text, (int(retreat_point[0]) - 30, int(retreat_point[1]) - 20))
        
        # 绘制机器人主体
        if self.is_emergency_return:
            # 紧急返回状态用橙色表示
            color = (255, 165, 0)  # 橙色
        else:
            color = RED if self.is_colliding else (BLUE if self.prediction == 0 else YELLOW)
        
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), ROBOT_RADIUS)
        
        # 显示连续碰撞计数（如果有）
        if self.collision_counter > 0:
            font = pygame.font.SysFont(None, 14)
            count_text = font.render(str(self.collision_counter), True, BLACK)
            screen.blit(count_text, (int(self.x) - 4, int(self.y) - 7))
        
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
                elif event.key == pygame.K_c:  # 清除当前目标点
                    robot.target_x = None
                    robot.target_y = None
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键点击设置目标点
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    # 检查点击位置是否在障碍物内
                    if robot.set_target(mouse_x, mouse_y):
                        print(f"设置新目标点: ({mouse_x}, {mouse_y})")
                    else:
                        print("无法设置目标点: 位置在障碍物内")
        
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
        
        # 显示操作说明 - 使用较小的字体
        try:
            # 尝试使用系统中可能支持中文的字体，但字体大小更小
            font_options = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'sans-serif']
            font = None
            font_size = 18  # 减小字体大小
            
            # 尝试不同的字体，直到找到可用的
            for font_name in font_options:
                try:
                    font = pygame.font.SysFont(font_name, font_size)
                    break
                except:
                    continue
                
            # 如果没有找到支持中文的字体，则使用默认字体
            if font is None:
                font = pygame.font.SysFont(None, font_size)
                
            instructions = [
                f"Map: {MAP_VERSION}",
                "Space: Pause/Continue",
                "R: Reset Robot",
                "C: Clear Target",
                "Click: Set Target",
                f"Status: {'EMERGENCY!' if robot.is_emergency_return else ('Collision!' if robot.prediction == 1 else 'Safe')}"
            ]
            
            # 如果处于紧急状态，显示更多详情
            if robot.is_emergency_return:
                emergency_info = [
                    f"Retreat attempt: {robot.retreat_attempts}/{robot.max_retreat_attempts}",
                    f"Retreat steps: {robot.current_retreat_steps}",
                    f"Cache size: {len(robot.path_cache)}"
                ]
                instructions.extend(emergency_info)
            for i, text in enumerate(instructions):
                # 紧急状态文字用红色显示
                text_color = RED if robot.is_emergency_return and i == 5 else BLACK
                img = font.render(text, True, text_color)
                screen.blit(img, (10, 10 + i*20))  # 减小行间距
        except Exception as e:
            print(f"渲染文字时出错: {str(e)}")
            # 如果出错，尝试使用最简单的设置
            font = pygame.font.SysFont(None, 30)  # 使用更小的字体
            simple_instructions = [
                f"MAP: {MAP_VERSION}",
                "SPACE: PAUSE/PLAY",
                "R: RESET",
                "C: CLEAR TARGET",
                "CLICK: SET TARGET",
                f"STATUS: {'EMERG!' if robot.is_emergency_return else ('COLL!' if robot.prediction == 1 else 'SAFE')}"
            ]
            
            # 如果处于紧急状态，显示更多详情
            if robot.is_emergency_return:
                emergency_info = [
                    f"RETRY: {robot.retreat_attempts}/{robot.max_retreat_attempts}",
                    f"STEPS: {robot.current_retreat_steps}",
                    f"CACHE: {len(robot.path_cache)}"
                ]
                simple_instructions.extend(emergency_info)
            for i, text in enumerate(simple_instructions):
                # 紧急状态文字用红色显示
                text_color = RED if robot.is_emergency_return and i == 5 else BLACK
                img = font.render(text, True, text_color)
                screen.blit(img, (10, 10 + i*18))  # 减小行间距
        
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()
    pygame.quit()

