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
MAP_VERSION = "v20"   # 使用新添加的简单地图
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
        
        # 路径规划相关
        self.use_path_planning = True  # 是否使用路径规划
        self.planned_path = []  # 计划路径点列表
        self.current_path_index = 0  # 当前路径点索引
        self.path_planning_mode = "horizontal_first"  # 默认先水平后垂直
        self.max_retreat_attempts = 5  # 最大回退尝试次数
        
        # 目标到达状态
        self.target_reached = False  # 是否已到达目标
        self.target_approach_timer = 0  # 接近目标但未到达的计时器
        self.max_approach_time = 50  # 最大接近时间，超过则重新规划路径
        self.obstacle_collisions = {}  # 记录每段路径上的碰撞次数 {路径段索引: 碰撞次数}
        self.path_segment_blocked = set()  # 被认为阻塞的路径段索引集合
        
        # 动态路径规划历史
        self.path_planning_history = []  # 历史路径规划模式列表 [(模式, 起点, 终点, 成功/失败)]
        self.current_direction = None  # 当前主要移动方向: "horizontal", "vertical", "diagonal"
        self.consecutive_collision_spots = []  # 连续碰撞的位置记录
        self.last_replan_position = None  # 上一次重新规划路径的位置
        self.direction_change_count = 0  # 在同一区域改变方向的次数
        
        # 动态路径缓存机制增强
        self.last_planning_mode = None  # 记录上一次使用的路径规划模式
        self.consecutive_collision_count = 0  # 记录连续碰撞次数（与collision_counter不同，这个不会被紧急返回重置）
        self.consecutive_collision_threshold = 8  # 连续碰撞阈值，触发方向反转
        self.used_planning_modes = []  # 记录已使用过的规划模式，最多保留最近5次
        
    def set_target(self, x, y):
        # 检查目标点是否在障碍物内
        for obs in self.obstacles:
            if obs.collidepoint(x, y):
                return False  # 如果在障碍物内，返回False
        
        self.target_x = x
        self.target_y = y
        self.target_reached = False  # 重置目标到达状态
        self.target_approach_timer = 0  # 重置接近计时器
        self.obstacle_collisions = {}  # 重置碰撞记录
        self.path_segment_blocked = set()  # 重置阻塞路径段
        
        # 如果启用路径规划，生成路径点
        if self.use_path_planning:
            self.plan_path_to_target()
            
        return True  # 设置成功返回True
    
    def determine_best_direction(self):
        """基于历史记录和当前状态确定最佳路径方向"""
        # 如果连续碰撞次数超过阈值，需要避开上一次使用的规划模式
        if self.consecutive_collision_count >= self.consecutive_collision_threshold:
            print(f"连续碰撞次数({self.consecutive_collision_count})超过阈值({self.consecutive_collision_threshold})，切换规划模式")
            
            # 重置连续碰撞计数
            self.consecutive_collision_count = 0
            
            # 如果存在上一次规划模式，避开它
            if self.last_planning_mode:
                print(f"避开上一次规划模式: {self.last_planning_mode}")
                
                # 选择一个不同的规划模式
                if self.last_planning_mode == "horizontal_first":
                    return "vertical_first"  # 如果上次是水平优先，现在用垂直优先
                elif self.last_planning_mode == "vertical_first":
                    return "zigzag"  # 如果上次是垂直优先，现在用之字形
                elif self.last_planning_mode == "zigzag":
                    return "direct"  # 如果上次是之字形，现在用直接路径
                else:
                    return "horizontal_first"  # 如果上次是直接路径，现在用水平优先
            
        # 如果没有历史记录，使用默认方向
        if not self.path_planning_history:
            return self.path_planning_mode
        
        # 检查是否在同一区域反复规划（可能被大障碍物困住）
        if self.last_replan_position:
            dist_to_last_replan = math.sqrt(
                (self.x - self.last_replan_position[0])**2 + 
                (self.y - self.last_replan_position[1])**2
            )
            if dist_to_last_replan < ROBOT_RADIUS * 5:
                # 在同一区域反复规划，需要更激进的方向变化
                self.direction_change_count += 1
                print(f"在同一区域反复规划 {self.direction_change_count} 次，尝试更激进的变化")
                
                if self.direction_change_count >= 3:
                    # 尝试更复杂的之字形路径
                    return "zigzag"
            else:
                # 不在同一区域，重置计数
                self.direction_change_count = 0
        
        # 获取最近的路径规划历史
        recent_history = self.path_planning_history[-min(5, len(self.path_planning_history)):]
        
        # 检查最近的成功策略
        successful_modes = [h[0] for h in recent_history if h[3] == "success"]
        if successful_modes:
            # 优先使用最近成功的策略
            return successful_modes[-1]
        
        # 如果没有成功策略，避免最近失败的策略
        failed_modes = [h[0] for h in recent_history if h[3] == "failure"]
        if failed_modes:
            last_mode = failed_modes[-1]
            
            # 切换方向
            if last_mode == "horizontal_first":
                return "vertical_first"
            elif last_mode == "vertical_first":
                return "zigzag"
            elif last_mode == "zigzag":
                return "direct"
            else:
                return "horizontal_first"
        
        # 默认返回当前模式
        return self.path_planning_mode
    
    def record_path_outcome(self, outcome):
        """记录当前路径规划的结果"""
        if not self.planned_path:
            return
            
        # 记录路径规划结果
        start_point = self.planned_path[0] if self.planned_path else (self.x, self.y)
        end_point = self.planned_path[-1] if self.planned_path else (self.target_x, self.target_y)
        
        history_entry = (self.path_planning_mode, start_point, end_point, outcome)
        self.path_planning_history.append(history_entry)
        
        # 限制历史记录长度
        if len(self.path_planning_history) > 10:
            self.path_planning_history.pop(0)
        
        print(f"记录路径规划结果: {self.path_planning_mode} - {outcome}")
    
    def update_current_direction(self):
        """更新当前主要移动方向"""
        if not self.planned_path or self.current_path_index >= len(self.planned_path):
            return
            
        # 获取当前路径段
        current_idx = min(self.current_path_index, len(self.planned_path) - 2)
        current_point = self.planned_path[current_idx]
        next_point = self.planned_path[current_idx + 1]
        
        # 计算水平和垂直距离
        dx = abs(next_point[0] - current_point[0])
        dy = abs(next_point[1] - current_point[1])
        
        # 确定主要方向
        if dx > dy * 2:
            self.current_direction = "horizontal"
        elif dy > dx * 2:
            self.current_direction = "vertical"
        else:
            self.current_direction = "diagonal"
    
    def plan_path_to_target(self):
        """规划从当前位置到目标的路径"""
        if self.target_x is None or self.target_y is None:
            return
        
        # 记录规划位置
        self.last_replan_position = (self.x, self.y)
        
        # 确定最佳路径规划模式
        best_mode = self.determine_best_direction()
        if best_mode != self.path_planning_mode:
            print(f"基于历史记录切换路径规划模式: {self.path_planning_mode} -> {best_mode}")
            self.path_planning_mode = best_mode
        
        # 记录本次使用的规划模式
        self.last_planning_mode = self.path_planning_mode
        # 只有当规划模式不同时才添加到已使用模式列表
        if not self.used_planning_modes or self.used_planning_modes[-1] != self.path_planning_mode:
            self.used_planning_modes.append(self.path_planning_mode)
            # 限制已使用模式的历史长度
            if len(self.used_planning_modes) > 5:
                self.used_planning_modes.pop(0)
        print(f"本次规划使用模式: {self.path_planning_mode}, 历史规划模式: {self.used_planning_modes}")
        
        # 检查连续碰撞情况，可能需要调整路径模式
        if self.consecutive_collision_count >= self.consecutive_collision_threshold / 2:
            print(f"检测到连续碰撞({self.consecutive_collision_count})，考虑调整路径规划模式")
            
            # 如果当前使用的是之前失败的模式，尝试切换
            if self.used_planning_modes and len(self.used_planning_modes) >= 2:
                recent_modes = self.used_planning_modes[-2:]
                if self.path_planning_mode in ["horizontal_first", "vertical_first"]:
                    if all(m == "horizontal_first" for m in recent_modes) or all(m == "vertical_first" for m in recent_modes):
                        # 如果最近两次都是相同模式且碰撞，则切换模式
                        if self.path_planning_mode == "horizontal_first":
                            self.path_planning_mode = "vertical_first"
                        else:
                            self.path_planning_mode = "horizontal_first"
                        print(f"连续使用相同模式失败，切换为: {self.path_planning_mode}")
        
        # 清空之前的路径
        self.planned_path = []
        self.current_path_index = 0
        
        # 获取当前位置和目标位置
        start_x, start_y = self.x, self.y
        end_x, end_y = self.target_x, self.target_y
        
        # 创建路径点
        if self.path_planning_mode == "horizontal_first":
            # 检查是否有水平路径段被阻塞
            if 1 in self.path_segment_blocked:  # 水平段被阻塞
                # 改用垂直优先路径
                print("水平路径被阻塞，切换为垂直优先路径")
                mid_point = (start_x, end_y)
                self.planned_path = [(start_x, start_y), mid_point, (end_x, end_y)]
            else:
                # 先水平移动，再垂直移动
                mid_point = (end_x, start_y)
                self.planned_path = [(start_x, start_y), mid_point, (end_x, end_y)]
                
        elif self.path_planning_mode == "vertical_first":
            # 检查是否有垂直路径段被阻塞
            if 1 in self.path_segment_blocked:  # 垂直段被阻塞
                # 改用水平优先路径
                print("垂直路径被阻塞，切换为水平优先路径")
                mid_point = (end_x, start_y)
                self.planned_path = [(start_x, start_y), mid_point, (end_x, end_y)]
            else:
                # 先垂直移动，再水平移动
                mid_point = (start_x, end_y)
                self.planned_path = [(start_x, start_y), mid_point, (end_x, end_y)]
                
        elif self.path_planning_mode == "zigzag":
            # 之字形路径，增加中间点
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2
            
            # 根据阻塞情况选择不同的中间点
            if 1 in self.path_segment_blocked and 2 in self.path_segment_blocked:
                # 如果之前的两种路径都被阻塞，尝试更多点的路径
                quarter_x1 = start_x + (end_x - start_x) / 4
                quarter_x3 = start_x + 3 * (end_x - start_x) / 4
                quarter_y1 = start_y + (end_y - start_y) / 4
                quarter_y3 = start_y + 3 * (end_y - start_y) / 4
                
                # 根据碰撞记录选择偏移方向
                offset_x = ROBOT_RADIUS * 10
                offset_y = ROBOT_RADIUS * 10
                
                # 根据历史碰撞位置调整偏移方向
                if self.consecutive_collision_spots:
                    avg_x = sum(p[0] for p in self.consecutive_collision_spots) / len(self.consecutive_collision_spots)
                    avg_y = sum(p[1] for p in self.consecutive_collision_spots) / len(self.consecutive_collision_spots)
                    
                    # 远离平均碰撞位置
                    if avg_x > mid_x:
                        offset_x = -offset_x
                    if avg_y > mid_y:
                        offset_y = -offset_y
                
                # 创建曲折路径，避开障碍物
                self.planned_path = [
                    (start_x, start_y),
                    (quarter_x1, start_y),
                    (quarter_x1, quarter_y1),
                    (mid_x + offset_x, mid_y + offset_y),  # 偏移中点，避开障碍
                    (quarter_x3, quarter_y3),
                    (quarter_x3, end_y),
                    (end_x, end_y)
                ]
            else:
                # 默认之字形路径
                self.planned_path = [
                    (start_x, start_y),
                    (mid_x, start_y),
                    (mid_x, mid_y),
                    (end_x, mid_y),
                    (end_x, end_y)
                ]
        else:
            # 直接路径（对角线）
            self.planned_path = [(start_x, start_y), (end_x, end_y)]
        
        # 重置路径相关状态
        self.obstacle_collisions = {}
        self.path_segment_blocked = set()
        self.current_path_index = 0
        self.consecutive_collision_spots = []  # 重置连续碰撞位置
        
        # 更新当前移动方向
        self.update_current_direction()
        
        # 检查路径是否与障碍物相交
        path_collides, collision_segment = self.check_path_collision(self.planned_path)
        
        # 如果路径与障碍物相交，尝试使用不同的规划模式
        max_attempts = 3  # 最大尝试次数
        attempt_count = 0
        
        while path_collides and attempt_count < max_attempts:
            attempt_count += 1
            print(f"检测到路径与障碍物相交，第 {attempt_count} 次尝试重新规划")
            
            # 标记相交的路径段为阻塞
            if collision_segment >= 0:
                self.path_segment_blocked.add(collision_segment)
            
            # 尝试不同的规划模式
            if self.path_planning_mode == "horizontal_first":
                self.path_planning_mode = "vertical_first"
            elif self.path_planning_mode == "vertical_first":
                self.path_planning_mode = "zigzag"
            elif self.path_planning_mode == "zigzag":
                self.path_planning_mode = "direct"
            else:
                self.path_planning_mode = "horizontal_first"
                
            print(f"路径相交，切换规划模式为: {self.path_planning_mode}")
            
            # 清空路径，创建新路径
            self.planned_path = []
            self.current_path_index = 0
            
            # 获取当前位置和目标位置
            start_x, start_y = self.x, self.y
            end_x, end_y = self.target_x, self.target_y
            
            # 根据新模式创建路径点
            if self.path_planning_mode == "horizontal_first":
                # 先水平移动，再垂直移动
                mid_point = (end_x, start_y)
                self.planned_path = [(start_x, start_y), mid_point, (end_x, end_y)]
                
            elif self.path_planning_mode == "vertical_first":
                # 先垂直移动，再水平移动
                mid_point = (start_x, end_y)
                self.planned_path = [(start_x, start_y), mid_point, (end_x, end_y)]
                
            elif self.path_planning_mode == "zigzag":
                # 之字形路径，增加中间点
                mid_x = (start_x + end_x) / 2
                mid_y = (start_y + end_y) / 2
                
                # 增加随机偏移，尝试避开障碍物
                offset_x = ROBOT_RADIUS * 5 * random.choice([-1, 1])
                offset_y = ROBOT_RADIUS * 5 * random.choice([-1, 1])
                
                self.planned_path = [
                    (start_x, start_y),
                    (mid_x + offset_x, start_y),
                    (mid_x + offset_x, mid_y + offset_y),
                    (end_x, mid_y + offset_y),
                    (end_x, end_y)
                ]
            else:
                # 直接路径（对角线）
                self.planned_path = [(start_x, start_y), (end_x, end_y)]
            
            # 更新当前移动方向
            self.update_current_direction()
            
            # 重新检查路径
            path_collides, collision_segment = self.check_path_collision(self.planned_path)
        
        # 如果多次尝试后仍然相交，记录失败
        if path_collides:
            print(f"经过 {max_attempts} 次尝试，无法找到无碰撞路径，使用最后一次规划的路径")
            
        # 如果有新规划的路径，立即更新转向角度到第一个路径点
        if len(self.planned_path) > 1 and self.current_path_index < len(self.planned_path):
            next_point = self.planned_path[self.current_path_index + 1]
            target_angle = math.degrees(math.atan2(next_point[1] - self.y, next_point[0] - self.x)) % 360
            
            # 立即设置角度，避免转圈
            angle_diff = (target_angle - self.angle) % 360
            if angle_diff > 180:
                angle_diff -= 360
                
            # 如果角度差较大，直接设置为目标角度
            if abs(angle_diff) > 90:
                print(f"立即转向目标角度: {target_angle:.1f}°，避免转圈")
                self.angle = target_angle
        
        print(f"路径规划完成: {self.planned_path}")
        print(f"当前主要方向: {self.current_direction}")
        print(f"使用的规划模式: {self.path_planning_mode}")
        print(f"规划模式历史: {self.used_planning_modes}")
        
    def is_path_segment_blocked(self, segment_idx):
        """判断路径段是否被阻塞"""
        collision_count = self.obstacle_collisions.get(segment_idx, 0)
        return collision_count >= 5  # 如果碰撞次数超过5次，认为该路径段被阻塞
        
    def check_path_collision(self, path):
        """检查路径是否与障碍物相交
        
        Args:
            path: 路径点列表 [(x1, y1), (x2, y2), ...]
            
        Returns:
            bool: 如果路径与任何障碍物相交则返回True，否则返回False
            int: 相交的路径段索引，如果没有相交则返回-1
        """
        if not path or len(path) < 2:
            return False, -1
            
        # 对每个路径段进行检查
        for i in range(len(path) - 1):
            start_point = path[i]
            end_point = path[i + 1]
            
            # 创建更多的中间采样点，提高检测准确性
            sample_count = max(10, int(math.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2) / 10))
            
            for j in range(sample_count + 1):
                # 计算采样点位置
                t = j / sample_count
                sample_x = start_point[0] + t * (end_point[0] - start_point[0])
                sample_y = start_point[1] + t * (end_point[1] - start_point[1])
                
                # 检查采样点是否在任何障碍物内
                for obs in self.obstacles:
                    if obs.collidepoint(sample_x, sample_y):
                        print(f"路径段 {i} 与障碍物相交，在点 ({sample_x:.1f}, {sample_y:.1f})")
                        return True, i
                        
        # 如果所有段都没有相交，返回False
        return False, -1
        
    def set_direct_heading_to_next_point(self):
        """立即设置机器人朝向下一个路径点的方向"""
        if not self.planned_path or self.current_path_index >= len(self.planned_path) - 1:
            return False
            
        # 获取下一个路径点
        next_point = self.planned_path[self.current_path_index + 1]
        
        # 计算目标角度
        target_angle = math.degrees(math.atan2(next_point[1] - self.y, next_point[0] - self.x)) % 360
        
        # 计算角度差
        angle_diff = (target_angle - self.angle) % 360
        if angle_diff > 180:
            angle_diff -= 360
            
        # 记录当前角度
        old_angle = self.angle
        
        # 直接设置角度
        self.angle = target_angle
        
        print(f"直接调整朝向: {old_angle:.1f}° -> {target_angle:.1f}°，差值: {angle_diff:.1f}°")
        return True
        
    def toggle_path_planning_mode(self):
        """切换路径规划模式"""
        if self.path_planning_mode == "horizontal_first":
            self.path_planning_mode = "vertical_first"
        elif self.path_planning_mode == "vertical_first":
            self.path_planning_mode = "zigzag"
        elif self.path_planning_mode == "zigzag":
            self.path_planning_mode = "direct"
        else:
            self.path_planning_mode = "horizontal_first"
        
        print(f"路径规划模式切换为: {self.path_planning_mode}")
        
        # 重置路径相关状态
        self.obstacle_collisions = {}
        self.path_segment_blocked = set()
        
        # 如果已经设置了目标，重新规划路径
        if self.target_x is not None and self.target_y is not None:
            self.plan_path_to_target()
        
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
            self.consecutive_collision_count += 1  # 增加连续碰撞计数，不受紧急回退状态影响
            self.is_colliding = True
            self.collision_time = 10
            
            # 如果连续碰撞次数达到阈值，在下一次规划路径时会考虑改变方向
            if self.consecutive_collision_count >= self.consecutive_collision_threshold:
                print(f"连续碰撞次数({self.consecutive_collision_count})达到阈值，将在下次规划时改变方向")
        else:
            # 只有在非紧急状态下才重置碰撞计数器
            if not self.is_emergency_return:
                self.collision_counter = 0
                self.consecutive_collision_count = 0  # 重置连续碰撞计数
            
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
            # 如果有目标点且启用路径规划，按照路径点导航
            if self.target_x is not None and self.target_y is not None:
                # 路径规划导航
                if self.use_path_planning and len(self.planned_path) > 0:
                    # 确保当前路径索引有效
                    if self.current_path_index < len(self.planned_path):
                        # 获取当前目标路径点
                        current_target = self.planned_path[self.current_path_index]
                        
                        # 计算到当前路径点的距离
                        distance_to_point = math.sqrt((self.x - current_target[0])**2 + (self.y - current_target[1])**2)
                        
                        # 检查是否持续无法接近路径点（可能被障碍物阻挡）
                        if distance_to_point < ARRIVAL_DISTANCE * 3 and self.prediction == 1:
                            # 记录当前路径段的碰撞
                            segment_idx = self.current_path_index
                            self.obstacle_collisions[segment_idx] = self.obstacle_collisions.get(segment_idx, 0) + 1
                            
                            # 如果碰撞次数过多，标记该路径段为阻塞
                            if self.obstacle_collisions[segment_idx] >= 5:
                                self.path_segment_blocked.add(segment_idx)
                                print(f"路径段 {segment_idx} 被阻塞，尝试重新规划路径")
                                self.plan_path_to_target()
                                return
                        
                        # 如果到达当前路径点，移动到下一个路径点
                        if distance_to_point <= ARRIVAL_DISTANCE:
                            self.current_path_index += 1
                            # 如果所有路径点都已访问，设置为到达目标
                            if self.current_path_index >= len(self.planned_path):
                                print("已到达最终目标点！")
                                self.target_reached = True
                                # 记录当前路径成功
                                self.record_path_outcome("success")
                                return  # 停止移动
                            current_target = self.planned_path[self.current_path_index]
                            print(f"到达路径点 {self.current_path_index-1}，前往下一路径点: {current_target}")
                            # 重置接近计时器
                            self.target_approach_timer = 0
                        
                        # 计算到当前路径点的角度
                        target_angle = math.degrees(math.atan2(current_target[1] - self.y, current_target[0] - self.x)) % 360
                    else:
                        # 已经到达所有路径点，停止移动
                        self.target_reached = True
                        return
                else:
                    # 传统导航（直接前往目标）
                    target_angle = math.degrees(math.atan2(self.target_y - self.y, self.target_x - self.x)) % 360
                    
                    # 检查是否到达目标
                    distance_to_target = math.sqrt((self.x - self.target_x)**2 + (self.y - self.target_y)**2)
                    
                    # 如果很接近但未到达目标，且持续时间较长，可能是被困住了
                    if distance_to_target <= ARRIVAL_DISTANCE * 2:
                        self.target_approach_timer += 1
                        
                        # 如果在目标附近徘徊太久，重新规划路径
                        if self.target_approach_timer > self.max_approach_time:
                            print("在目标附近徘徊太久，尝试新的路径规划模式")
                            # 切换到下一个路径规划模式
                            self.toggle_path_planning_mode()
                            # 使用新模式重新规划路径
                            self.plan_path_to_target()
                            return
                    else:
                        # 不在目标附近，重置计时器
                        self.target_approach_timer = 0
                    
                    if distance_to_target <= ARRIVAL_DISTANCE:
                        print("已到达目标点！")
                        self.target_reached = True
                        # 记录当前路径成功
                        self.record_path_outcome("success")
                        return  # 到达目标点，停止移动
                
        # 如果预测会碰撞，则暂时改变方向避开障碍物
                if self.prediction == 1:
                    # 记录碰撞位置
                    collision_pos = (self.x, self.y)
                    self.consecutive_collision_spots.append(collision_pos)
                    
                    # 保持最近的10个碰撞位置
                    if len(self.consecutive_collision_spots) > 10:
                        self.consecutive_collision_spots.pop(0)
                    
                    # 记录当前路径段的碰撞
                    segment_idx = self.current_path_index
                    self.obstacle_collisions[segment_idx] = self.obstacle_collisions.get(segment_idx, 0) + 1
                    
                    # 如果连续碰撞次数达到动态路径改变的阈值，触发重规划
                    if self.consecutive_collision_count >= self.consecutive_collision_threshold:
                        print(f"连续碰撞次数({self.consecutive_collision_count})达到动态路径阈值，强制改变路径方向")
                        
                        # 记录当前路径失败
                        self.record_path_outcome("failure")
                        
                        # 根据上一次规划模式选择新的规划模式
                        if self.last_planning_mode == "horizontal_first":
                            self.path_planning_mode = "vertical_first"
                            print(f"上次规划模式为水平优先，切换为垂直优先")
                        elif self.last_planning_mode == "vertical_first":
                            self.path_planning_mode = "zigzag"
                            print(f"上次规划模式为垂直优先，切换为之字形")
                        elif self.last_planning_mode == "zigzag":
                            self.path_planning_mode = "direct"
                            print(f"上次规划模式为之字形，切换为直接路径")
                        else:
                            self.path_planning_mode = "horizontal_first"
                            print(f"上次规划模式为直接路径，切换为水平优先")
                        
                        # 重规划路径
                        self.plan_path_to_target()
                        # 重置连续碰撞计数
                        self.consecutive_collision_count = 0
                        # 立即调整朝向
                        self.set_direct_heading_to_next_point()
                        return
                    
                    # 如果碰撞次数达到阈值，标记为阻塞并重规划
                    if self.obstacle_collisions[segment_idx] >= 5:
                        self.path_segment_blocked.add(segment_idx)
                        print(f"路径段 {segment_idx} 被阻塞，重新规划路径")
                        
                        # 记录当前路径失败
                        self.record_path_outcome("failure")
                        
                        # 重新规划路径，使用不同方向
                        if self.current_direction == "horizontal":
                            # 之前是水平移动，现在改为垂直优先
                            old_mode = self.path_planning_mode
                            self.path_planning_mode = "vertical_first"
                            print(f"检测到水平方向碰撞，切换规划模式: {old_mode} -> {self.path_planning_mode}")
                        elif self.current_direction == "vertical":
                            # 之前是垂直移动，现在改为水平优先
                            old_mode = self.path_planning_mode
                            self.path_planning_mode = "horizontal_first"
                            print(f"检测到垂直方向碰撞，切换规划模式: {old_mode} -> {self.path_planning_mode}")
                        else:
                            # 之前是对角线或其他，尝试之字形
                            old_mode = self.path_planning_mode
                            self.path_planning_mode = "zigzag"
                            print(f"检测到对角线方向碰撞，切换规划模式: {old_mode} -> {self.path_planning_mode}")
                        
                        # 重新规划路径
                        self.plan_path_to_target()
                        # 立即调整朝向
                        self.set_direct_heading_to_next_point()
                        return
                    
                    # 计算暂时偏离的角度
                    self.angle = (self.angle + 90 + random.uniform(-45, 45)) % 360
                else:
                    # 如果安全，朝向目标方向调整
                    # 计算当前角度和目标角度之间的差异
                    angle_diff = (target_angle - self.angle) % 360
                    if angle_diff > 180:
                        angle_diff -= 360
                    
                    # 加快转向速度，特别是当角度差大时
                    if abs(angle_diff) > 90:
                        # 大角度差，快速转向
                        turn_rate = 15  # 大角度转向速度
                        print(f"大角度转向: {angle_diff:.1f}°，使用快速转向速率: {turn_rate}")
                    else:
                        # 小角度调整，平滑转向
                        turn_rate = 8  # 增加默认转向速度 (原来是5)
                    
                    # 应用转向
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
            
        # 绘制规划路径（如果有）
        if self.use_path_planning and len(self.planned_path) > 1:
            # 绘制路径线
            for i in range(len(self.planned_path)-1):
                start_point = self.planned_path[i]
                end_point = self.planned_path[i+1]
                
                # 根据路径段状态选择颜色
                if i+1 in self.path_segment_blocked:
                    line_color = (255, 0, 0)  # 红色表示阻塞
                elif i < self.current_path_index:
                    line_color = (0, 255, 0)  # 绿色表示已走过
                else:
                    line_color = (255, 255, 0)  # 黄色表示待走
                
                pygame.draw.line(screen, line_color, start_point, end_point, 2)
            
            # 绘制路径点
            for i, point in enumerate(self.planned_path):
                # 当前目标点使用不同颜色
                if i == self.current_path_index:
                    pygame.draw.circle(screen, (255, 165, 0), (int(point[0]), int(point[1])), 6)
                else:
                    # 已访问的点用绿色，未访问的点用黄色
                    color = (0, 255, 0) if i < self.current_path_index else (255, 255, 0)
                    pygame.draw.circle(screen, color, (int(point[0]), int(point[1])), 4)
            
        # 绘制目标点(如果有)
        if self.target_x is not None and self.target_y is not None:
            # 如果已到达目标，用不同颜色标记
            if self.target_reached:
                pygame.draw.circle(screen, (0, 255, 0), (int(self.target_x), int(self.target_y)), TARGET_RADIUS)
                # 绘制"已到达"标记
                font = pygame.font.SysFont(None, 16)
                text = font.render("已到达", True, (0, 255, 0))
                screen.blit(text, (int(self.target_x) - 20, int(self.target_y) - 25))
            else:
                pygame.draw.circle(screen, PURPLE, (int(self.target_x), int(self.target_y)), TARGET_RADIUS)
            
            # 绘制从机器人到目标点的线
            if not self.target_reached:  # 仅在未到达时绘制
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
    global MAP_VERSION  # 声明全局变量
    
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
    
    # 创建帮助文本
    font = pygame.font.SysFont(None, 24)
    help_text = [
        "鼠标左键: 设置目标点",
        "空格键: 启用/禁用路径规划",
        "M 键: 切换路径规划模式 (水平优先/垂直优先/直接)",
        "R 键: 重置机器人位置",
        "P 键: 暂停/继续",
        "1-5 键: 切换简单地图 simple1-simple5",
        "S 键: 循环切换所有简单地图"
    ]
    
    # 当前简单地图的索引
    current_simple_map_index = 1
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:  # P键暂停/继续
                    paused = not paused
                    print(f"游戏{'暂停' if paused else '继续'}")
                elif event.key == pygame.K_r:  # 重置机器人位置
                    main()
                    return
                elif event.key == pygame.K_c:  # 清除当前目标点
                    robot.target_x = None
                    robot.target_y = None
                    robot.planned_path = []
                    print("目标点已清除")
                elif event.key == pygame.K_SPACE:  # 启用/禁用路径规划
                    robot.use_path_planning = not robot.use_path_planning
                    print(f"路径规划: {'启用' if robot.use_path_planning else '禁用'}")
                    if robot.target_x is not None and robot.target_y is not None and robot.use_path_planning:
                        robot.plan_path_to_target()
                elif event.key == pygame.K_m:  # 切换路径规划模式
                    robot.toggle_path_planning_mode()
                # 添加地图切换快捷键
                elif event.key == pygame.K_1:  # 切换到简单地图1
                    MAP_VERSION = "simple1"
                    main()
                    return
                elif event.key == pygame.K_2:  # 切换到简单地图2
                    MAP_VERSION = "simple2"
                    main()
                    return
                elif event.key == pygame.K_3:  # 切换到简单地图3
                    MAP_VERSION = "simple3"
                    main()
                    return
                elif event.key == pygame.K_4:  # 切换到简单地图4
                    MAP_VERSION = "simple4"
                    main()
                    return
                elif event.key == pygame.K_5:  # 切换到简单地图5
                    MAP_VERSION = "simple5"
                    main()
                    return
                elif event.key == pygame.K_6:  # 切换到简单地图5
                    MAP_VERSION = "simple6"
                    main()
                    return
                elif event.key == pygame.K_s:  # 循环切换简单地图
                    current_simple_map_index = (current_simple_map_index % 5) + 1
                    MAP_VERSION = f"simple{current_simple_map_index}"
                    print(f"切换到地图: {MAP_VERSION}")
                    main()
                    return
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
                f"地图: {MAP_VERSION}",
                f"路径规划: {'开' if robot.use_path_planning else '关'}",
                f"模式: {robot.path_planning_mode}",
                f"状态: {'已到达!' if robot.target_reached else ('紧急!' if robot.is_emergency_return else ('碰撞风险!' if robot.prediction == 1 else '安全'))}",
                f"路径段: {robot.current_path_index}/{len(robot.planned_path) if len(robot.planned_path) > 0 else 0}",
                f"阻塞段: {list(robot.path_segment_blocked) if robot.path_segment_blocked else '无'}",
                "空格: 路径规划开关",
                "M: 切换路径模式",
                "P: 暂停/继续",
                "R: 重置机器人",
                "C: 清除目标",
                "点击: 设置目标",
                "1-5: 切换简单地图",
                "S: 循环简单地图"
            ]            # 如果处于紧急状态，显示更多详情
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

