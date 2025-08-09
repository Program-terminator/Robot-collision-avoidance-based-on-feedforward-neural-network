import pygame
import random

def create_map(version="v1"):
    obstacles = []
    if version == "v1":
        # 用于训练的数据地图
        # 确保包含至少9个不同形状和方位的障碍物
        obstacles = [
            pygame.Rect(100, 100, 50, 300),
            pygame.Rect(300, 200, 200, 50),
            pygame.Rect(600, 100, 50, 300),
            # 添加更多不同形状和方位的障碍物
            pygame.Rect(200, 400, 100, 50),  # 横向长方形
            pygame.Rect(450, 150, 50, 100),  # 纵向长方形
            pygame.Rect(150, 250, 75, 75),   # 正方形
            pygame.Rect(500, 350, 150, 40),  # 横向长方形
            pygame.Rect(350, 100, 40, 120),  # 纵向长方形
            pygame.Rect(250, 150, 60, 60),   # 正方形
            pygame.Rect(650, 300, 100, 50),  # 横向长方形
            pygame.Rect(100, 500, 50, 80),   # 纵向长方形
        ]
    elif version == "v2":
        # 用于预测的不同地图
        # 确保包含至少9个不同形状和方位的障碍物
        obstacles = [
            pygame.Rect(150, 150, 400, 50),
            pygame.Rect(200, 400, 50, 150),
            pygame.Rect(500, 300, 50, 200),
            # 添加更多不同形状和方位的障碍物
            pygame.Rect(300, 100, 100, 50),  # 横向长方形
            pygame.Rect(100, 300, 50, 100),  # 纵向长方形
            pygame.Rect(400, 250, 75, 75),   # 正方形
            pygame.Rect(600, 150, 150, 40),  # 横向长方形
            pygame.Rect(250, 200, 40, 120),  # 纵向长方形
            pygame.Rect(550, 400, 60, 60),   # 正方形
            pygame.Rect(150, 500, 100, 50),  # 横向长方形
            pygame.Rect(650, 350, 50, 80),   # 纵向长方形
        ]
    elif version == "v3":
        # 更复杂的地图1 - 包含不同形状的障碍物
        obstacles = [
            pygame.Rect(50, 50, 100, 200),
            pygame.Rect(200, 100, 50, 400),
            pygame.Rect(350, 50, 200, 50),
            pygame.Rect(350, 200, 50, 200),
            pygame.Rect(500, 300, 200, 50),
            pygame.Rect(650, 100, 50, 150),
            # 添加更多不同形状和方位的障碍物
            pygame.Rect(150, 400, 75, 75),  # 正方形
            pygame.Rect(450, 150, 150, 30),  # 横向长方形
            pygame.Rect(250, 300, 30, 150),  # 纵向长方形
            pygame.Rect(100, 300, 60, 60),  # 正方形
            pygame.Rect(550, 200, 120, 40),  # 横向长方形
            pygame.Rect(300, 400, 40, 120),  # 纵向长方形
            pygame.Rect(700, 300, 50, 50),  # 正方形
        ]
    elif version == "v4":
        # 更复杂的地图2 - 包含不同形状的障碍物
        obstacles = [
            pygame.Rect(100, 100, 200, 50),
            pygame.Rect(400, 100, 50, 200),
            pygame.Rect(500, 100, 150, 50),
            pygame.Rect(200, 300, 300, 50),
            pygame.Rect(100, 400, 50, 150),
            pygame.Rect(600, 350, 50, 200),
            # 添加更多不同形状和方位的障碍物
            pygame.Rect(300, 200, 60, 60),  # 正方形
            pygame.Rect(550, 250, 120, 40),  # 横向长方形
            pygame.Rect(150, 200, 40, 120),  # 纵向长方形
            pygame.Rect(250, 150, 70, 70),  # 正方形
            pygame.Rect(450, 300, 140, 35),  # 横向长方形
            pygame.Rect(350, 150, 35, 140),  # 纵向长方形
            pygame.Rect(700, 200, 60, 60),  # 正方形
        ]
    elif version == "v5":
        # 更复杂的地图3 - 包含不同形状的障碍物
        obstacles = [
            pygame.Rect(150, 50, 50, 200),
            pygame.Rect(300, 100, 200, 50),
            pygame.Rect(550, 50, 50, 200),
            pygame.Rect(150, 350, 200, 50),
            pygame.Rect(450, 300, 50, 200),
            pygame.Rect(600, 350, 100, 50),
            # 添加更多不同形状和方位的障碍物
            pygame.Rect(250, 200, 80, 80),  # 正方形
            pygame.Rect(400, 150, 180, 35),  # 横向长方形
            pygame.Rect(100, 250, 35, 180),  # 纵向长方形
            pygame.Rect(350, 250, 75, 75),  # 正方形
            pygame.Rect(500, 200, 150, 45),  # 横向长方形
            pygame.Rect(200, 300, 45, 150),  # 纵向长方形
            pygame.Rect(650, 250, 70, 70),  # 正方形
        ]
    elif version == "v6":
        # 更复杂的地图4 - 包含不同形状的障碍物
        obstacles = [
            pygame.Rect(75, 75, 150, 50),
            pygame.Rect(75, 125, 50, 150),
            pygame.Rect(250, 200, 300, 50),
            pygame.Rect(300, 300, 50, 200),
            pygame.Rect(500, 350, 150, 50),
            pygame.Rect(600, 150, 50, 150),
            # 添加更多不同形状和方位的障碍物
            pygame.Rect(200, 100, 70, 70),  # 正方形
            pygame.Rect(450, 250, 130, 45),  # 横向长方形
            pygame.Rect(150, 300, 45, 130),  # 纵向长方形
            pygame.Rect(350, 150, 80, 80),  # 正方形
            pygame.Rect(550, 200, 160, 50),  # 横向长方形
            pygame.Rect(250, 350, 50, 160),  # 纵向长方形
            pygame.Rect(650, 300, 75, 75),  # 正方形
        ]
    elif version == "v7":
        # 更复杂的地图5 - 包含不同形状的障碍物
        obstacles = [
            pygame.Rect(100, 50, 50, 150),
            pygame.Rect(200, 150, 150, 50),
            pygame.Rect(400, 100, 50, 200),
            pygame.Rect(500, 250, 150, 50),
            pygame.Rect(200, 400, 50, 150),
            pygame.Rect(600, 400, 100, 50),
            # 添加更多不同形状和方位的障碍物
            pygame.Rect(300, 50, 90, 90),  # 正方形
            pygame.Rect(550, 150, 140, 50),  # 横向长方形
            pygame.Rect(100, 250, 50, 140),  # 纵向长方形
            pygame.Rect(250, 250, 85, 85),  # 正方形
            pygame.Rect(450, 300, 170, 40),  # 横向长方形
            pygame.Rect(150, 350, 40, 170),  # 纵向长方形
            pygame.Rect(650, 300, 80, 80),  # 正方形
        ]
    elif version == "v8":
        # 更复杂的地图6 - 包含不同形状的障碍物
        obstacles = [
            pygame.Rect(50, 100, 100, 50),
            pygame.Rect(200, 50, 50, 150),
            pygame.Rect(300, 200, 200, 50),
            pygame.Rect(550, 150, 50, 200),
            pygame.Rect(400, 400, 200, 50),
            pygame.Rect(150, 350, 50, 150),
            # 添加更多不同形状和方位的障碍物
            pygame.Rect(250, 100, 75, 75),  # 正方形
            pygame.Rect(500, 200, 160, 35),  # 横向长方形
            pygame.Rect(150, 200, 35, 160),  # 纵向长方形
            pygame.Rect(350, 100, 90, 90),  # 正方形
            pygame.Rect(600, 250, 180, 45),  # 横向长方形
            pygame.Rect(200, 300, 45, 180),  # 纵向长方形
            pygame.Rect(650, 100, 85, 85),  # 正方形
        ]
    elif version == "v9":
        # 更复杂的地图7 - 包含不同形状的障碍物
        obstacles = [
            pygame.Rect(100, 100, 150, 50),
            pygame.Rect(300, 50, 50, 150),
            pygame.Rect(400, 150, 150, 50),
            pygame.Rect(600, 100, 50, 200),
            pygame.Rect(200, 300, 50, 200),
            pygame.Rect(500, 350, 150, 50),
            # 添加更多不同形状和方位的障碍物
            pygame.Rect(200, 200, 85, 85),  # 正方形
            pygame.Rect(450, 250, 170, 40),  # 横向长方形
            pygame.Rect(100, 350, 40, 170),  # 纵向长方形
            pygame.Rect(300, 250, 95, 95),  # 正方形
            pygame.Rect(550, 200, 190, 50),  # 横向长方形
            pygame.Rect(250, 350, 50, 190),  # 纵向长方形
            pygame.Rect(650, 350, 90, 90),  # 正方形
        ]
    elif version == "v10":
        # 更复杂的地图8 - 包含不同形状的障碍物
        obstacles = [
            pygame.Rect(75, 75, 75, 150),
            pygame.Rect(200, 100, 50, 200),
            pygame.Rect(300, 250, 200, 50),
            pygame.Rect(550, 200, 50, 150),
            pygame.Rect(400, 400, 150, 50),
            pygame.Rect(600, 350, 100, 50),
            # 添加更多不同形状和方位的障碍物
            pygame.Rect(250, 50, 95, 95),  # 正方形
            pygame.Rect(500, 100, 190, 45),  # 横向长方形
            pygame.Rect(150, 300, 45, 190),  # 纵向长方形
            pygame.Rect(350, 150, 100, 100),  # 正方形
            pygame.Rect(550, 300, 170, 55),  # 横向长方形
            pygame.Rect(200, 350, 55, 170),  # 纵向长方形
            pygame.Rect(650, 250, 95, 95),  # 正方形
        ]
    elif version == "v11":
        # 更复杂的地图9 - 包含不同形状的障碍物
        obstacles = [
            pygame.Rect(50, 50, 100, 50),
            pygame.Rect(200, 50, 50, 150),
            pygame.Rect(300, 150, 150, 50),
            pygame.Rect(500, 100, 50, 150),
            pygame.Rect(150, 300, 200, 50),
            pygame.Rect(450, 350, 50, 150),
            # 添加更多不同形状和方位的障碍物
            pygame.Rect(250, 200, 100, 100),  # 正方形
            pygame.Rect(550, 250, 200, 30),  # 横向长方形
            pygame.Rect(100, 200, 30, 200),  # 纵向长方形
            pygame.Rect(350, 250, 110, 110),  # 正方形
            pygame.Rect(600, 200, 180, 40),  # 横向长方形
            pygame.Rect(200, 350, 40, 180),  # 纵向长方形
            pygame.Rect(650, 350, 100, 100),  # 正方形
        ]
    elif version == "v12":
        # 更复杂的地图10 - 包含不同形状的障碍物
        obstacles = [
            pygame.Rect(100, 75, 50, 150),
            pygame.Rect(250, 100, 150, 50),
            pygame.Rect(450, 50, 50, 200),
            pygame.Rect(550, 200, 100, 50),
            pygame.Rect(200, 350, 50, 150),
            pygame.Rect(600, 350, 50, 150),
            # 添加更多不同形状和方位的障碍物
            pygame.Rect(300, 250, 110, 110),  # 正方形
            pygame.Rect(500, 150, 150, 40),  # 横向长方形
            pygame.Rect(150, 250, 40, 150),  # 纵向长方形
            pygame.Rect(350, 150, 120, 120),  # 正方形
            pygame.Rect(550, 300, 160, 45),  # 横向长方形
            pygame.Rect(250, 300, 45, 160),  # 纵向长方形
            pygame.Rect(650, 250, 110, 110),  # 正方形
        ]
    elif version == "v13":
        # 更复杂的地图11 - 包含不同形状的障碍物
        obstacles = [
            pygame.Rect(75, 100, 100, 50),
            pygame.Rect(225, 50, 50, 150),
            pygame.Rect(325, 150, 100, 50),
            pygame.Rect(475, 100, 50, 200),
            pygame.Rect(175, 300, 150, 50),
            pygame.Rect(425, 350, 50, 150),
            # 添加更多不同形状和方位的障碍物
            pygame.Rect(275, 200, 120, 120),  # 正方形
            pygame.Rect(575, 250, 160, 35),  # 横向长方形
            pygame.Rect(125, 200, 35, 160),  # 纵向长方形
            pygame.Rect(375, 250, 130, 130),  # 正方形
            pygame.Rect(625, 200, 170, 50),  # 横向长方形
            pygame.Rect(225, 350, 50, 170),  # 纵向长方形
            pygame.Rect(675, 350, 120, 120),  # 正方形
        ]
    elif version == "v14":
        # 更复杂的地图12 - 包含不同形状的障碍物
        obstacles = [
            pygame.Rect(50, 75, 75, 150),
            pygame.Rect(175, 100, 50, 200),
            pygame.Rect(275, 250, 150, 50),
            pygame.Rect(475, 200, 50, 150),
            pygame.Rect(350, 400, 150, 50),
            pygame.Rect(550, 350, 100, 50),
            # 添加更多不同形状和方位的障碍物
            pygame.Rect(225, 50, 130, 130),  # 正方形
            pygame.Rect(525, 150, 170, 45),  # 横向长方形
            pygame.Rect(125, 300, 45, 170),  # 纵向长方形
            pygame.Rect(325, 150, 140, 140),  # 正方形
            pygame.Rect(575, 250, 180, 55),  # 横向长方形
            pygame.Rect(225, 350, 55, 180),  # 纵向长方形
            pygame.Rect(675, 300, 130, 130),  # 正方形
        ]
    elif version == "v15":
        # 更复杂的地图13 - 包含不同形状的障碍物
        obstacles = [
            pygame.Rect(100, 50, 50, 150),
            pygame.Rect(200, 150, 100, 50),
            pygame.Rect(350, 100, 50, 150),
            pygame.Rect(450, 200, 100, 50),
            pygame.Rect(250, 300, 50, 150),
            pygame.Rect(550, 350, 100, 50),
            # 添加更多不同形状和方位的障碍物
            pygame.Rect(300, 250, 140, 140),  # 正方形
            pygame.Rect(500, 100, 180, 50),  # 横向长方形
            pygame.Rect(150, 250, 50, 180),  # 纵向长方形
            pygame.Rect(400, 300, 150, 150),  # 正方形
            pygame.Rect(600, 200, 190, 60),  # 横向长方形
            pygame.Rect(250, 350, 60, 190),  # 纵向长方形
            pygame.Rect(650, 300, 140, 140),  # 正方形
        ]
    elif version == "v16":
        # 更复杂的地图14 - 包含不同形状的障碍物
        obstacles = [
            pygame.Rect(75, 75, 100, 50),
            pygame.Rect(225, 50, 50, 150),
            pygame.Rect(325, 150, 150, 50),
            pygame.Rect(525, 100, 50, 200),
            pygame.Rect(175, 300, 150, 50),
            pygame.Rect(475, 350, 50, 150),
            # 添加更多不同形状和方位的障碍物
            pygame.Rect(275, 200, 150, 150),  # 正方形
            pygame.Rect(575, 250, 190, 40),  # 横向长方形
            pygame.Rect(125, 200, 40, 190),  # 纵向长方形
            pygame.Rect(375, 250, 160, 160),  # 正方形
            pygame.Rect(625, 300, 200, 50),  # 横向长方形
            pygame.Rect(225, 350, 50, 200),  # 纵向长方形
            pygame.Rect(675, 250, 150, 150),  # 正方形
        ]
    elif version == "v17":
        # 更复杂的地图15 - 包含不同形状的障碍物
        obstacles = [
            pygame.Rect(50, 100, 75, 150),
            pygame.Rect(175, 100, 50, 200),
            pygame.Rect(275, 250, 100, 50),
            pygame.Rect(425, 200, 50, 150),
            pygame.Rect(350, 400, 150, 50),
            pygame.Rect(550, 350, 100, 50),
            # 添加更多不同形状和方位的障碍物
            pygame.Rect(225, 50, 160, 160),  # 正方形
            pygame.Rect(525, 150, 200, 45),  # 横向长方形
            pygame.Rect(125, 300, 45, 200),  # 纵向长方形
            pygame.Rect(325, 150, 170, 170),  # 正方形
            pygame.Rect(575, 250, 180, 55),  # 横向长方形
            pygame.Rect(225, 350, 55, 180),  # 纵向长方形
            pygame.Rect(675, 350, 160, 160),  # 正方形
        ]
    elif version == "v18":
        # 更复杂的地图16 - 包含不同形状的障碍物
        obstacles = [
            pygame.Rect(100, 50, 50, 150),
            pygame.Rect(200, 150, 150, 50),
            pygame.Rect(400, 100, 50, 150),
            pygame.Rect(500, 200, 100, 50),
            pygame.Rect(250, 300, 50, 150),
            pygame.Rect(550, 350, 100, 50),
            # 添加更多不同形状和方位的障碍物
            pygame.Rect(300, 250, 170, 170),  # 正方形
            pygame.Rect(500, 100, 170, 55),  # 横向长方形
            pygame.Rect(150, 250, 55, 170),  # 纵向长方形
            pygame.Rect(400, 300, 180, 180),  # 正方形
            pygame.Rect(600, 250, 190, 60),  # 横向长方形
            pygame.Rect(250, 350, 60, 190),  # 纵向长方形
            pygame.Rect(650, 350, 170, 170),  # 正方形
        ]
    elif version == "v19":
        # 更复杂的地图17 - 包含不同形状的障碍物
        obstacles = [
            pygame.Rect(75, 75, 100, 50),
            pygame.Rect(225, 50, 50, 150),
            pygame.Rect(325, 150, 100, 50),
            pygame.Rect(475, 100, 50, 200),
            pygame.Rect(175, 300, 150, 50),
            pygame.Rect(475, 350, 50, 150),
            # 添加更多不同形状和方位的障碍物
            pygame.Rect(275, 200, 180, 180),  # 正方形
            pygame.Rect(575, 250, 180, 50),  # 横向长方形
            pygame.Rect(125, 200, 50, 180),  # 纵向长方形
            pygame.Rect(375, 250, 190, 190),  # 正方形
            pygame.Rect(625, 200, 200, 60),  # 横向长方形
            pygame.Rect(225, 350, 60, 200),  # 纵向长方形
            pygame.Rect(675, 350, 180, 180),  # 正方形
        ]
    elif version == "v20":
        # 更复杂的地图18 - 包含不同形状的障碍物
        obstacles = [
            pygame.Rect(50, 100, 75, 150),
            pygame.Rect(175, 100, 50, 200),
            pygame.Rect(275, 250, 150, 50),
            pygame.Rect(475, 200, 50, 150),
            pygame.Rect(350, 400, 100, 50),
            pygame.Rect(550, 350, 100, 50),
            # 添加更多不同形状和方位的障碍物
            pygame.Rect(225, 50, 190, 190),  # 正方形
            pygame.Rect(525, 150, 190, 55),  # 横向长方形
            pygame.Rect(125, 300, 55, 190),  # 纵向长方形
            pygame.Rect(375, 150, 200, 200),  # 正方形
            pygame.Rect(575, 250, 200, 60),  # 横向长方形
            pygame.Rect(225, 350, 60, 200),  # 纵向长方形
            pygame.Rect(675, 300, 190, 190),  # 正方形
        ]
    elif version == "v21":
        # 更复杂的地图19 - 包含不同形状的障碍物
        obstacles = [
            pygame.Rect(100, 50, 50, 150),
            pygame.Rect(200, 150, 100, 50),
            pygame.Rect(350, 100, 50, 150),
            pygame.Rect(450, 200, 150, 50),
            pygame.Rect(250, 300, 50, 150),
            pygame.Rect(550, 350, 100, 50),
            # 添加更多不同形状和方位的障碍物
            pygame.Rect(300, 250, 200, 200),  # 正方形
            pygame.Rect(500, 100, 200, 60),  # 横向长方形
            pygame.Rect(150, 250, 60, 200),  # 纵向长方形
            pygame.Rect(400, 300, 180, 180),  # 正方形
            pygame.Rect(600, 200, 190, 65),  # 横向长方形
            pygame.Rect(250, 350, 65, 190),  # 纵向长方形
            pygame.Rect(650, 350, 200, 200),  # 正方形
        ]
    elif version == "v22":
        # 更复杂的地图20 - 包含不同形状的障碍物
        obstacles = [
            pygame.Rect(75, 75, 100, 50),
            pygame.Rect(225, 50, 50, 150),
            pygame.Rect(325, 150, 150, 50),
            pygame.Rect(525, 100, 50, 200),
            pygame.Rect(175, 300, 100, 50),
            pygame.Rect(425, 350, 50, 150),
            # 添加更多不同形状和方位的障碍物
            pygame.Rect(275, 200, 175, 175),  # 正方形
            pygame.Rect(575, 250, 175, 60),  # 横向长方形
            pygame.Rect(125, 200, 60, 175),  # 纵向长方形
            pygame.Rect(375, 250, 185, 185),  # 正方形
            pygame.Rect(625, 200, 185, 65),  # 横向长方形
            pygame.Rect(225, 350, 65, 185),  # 纵向长方形
            pygame.Rect(675, 350, 175, 175),  # 正方形
        ]
    # 简单地图1 - 几个简单的障碍物，开放空间较多
    elif version == "simple1":
        obstacles = [
            pygame.Rect(200, 100, 50, 200),  # 左侧纵向墙
            pygame.Rect(500, 300, 50, 200),  # 右侧纵向墙
            pygame.Rect(350, 200, 100, 50),  # 中间横向墙
        ]
    # 简单地图2 - L形障碍物
    elif version == "simple2":
        obstacles = [
            pygame.Rect(300, 100, 50, 300),  # 竖直部分
            pygame.Rect(300, 350, 300, 50),  # 水平部分
        ]
    # 简单地图3 - 几个分散的小方块
    elif version == "simple3":
        obstacles = [
            pygame.Rect(150, 150, 70, 70),  # 左上角方块
            pygame.Rect(550, 150, 70, 70),  # 右上角方块
            pygame.Rect(150, 400, 70, 70),  # 左下角方块
            pygame.Rect(550, 400, 70, 70),  # 右下角方块
            pygame.Rect(350, 275, 70, 70),  # 中心方块
        ]
    # 简单地图4 - 单一通道
    elif version == "simple4":
        obstacles = [
            pygame.Rect(100, 100, 600, 50),  # 上方墙
            pygame.Rect(100, 400, 600, 50),  # 下方墙
            pygame.Rect(400, 150, 50, 100),  # 中间障碍物上部分
            pygame.Rect(400, 300, 50, 100),  # 中间障碍物下部分
        ]
    # 简单地图5 - 简单迷宫
    elif version == "simple5":
        obstacles = [
            pygame.Rect(200, 100, 50, 200),  # 第一道墙
            pygame.Rect(400, 250, 50, 200),  # 第二道墙
            pygame.Rect(600, 100, 50, 200),  # 第三道墙
        ]
    # 简单地图6 - 回形
    elif version == "simple6":
        obstacles = [
            pygame.Rect(360, 210, 180, 180),
            # 左上角
            pygame.Rect(250, 100, 150, 50),
            pygame.Rect(250, 100, 50, 150),
            # 右上角
            pygame.Rect(500, 100, 150, 50),
            pygame.Rect(600, 100, 50, 150),
            # 左下角
            pygame.Rect(250, 450, 150, 50),
            pygame.Rect(250, 350, 50, 150),
            # 右下角
            pygame.Rect(500, 450, 150, 50),
            pygame.Rect(600, 350, 50, 150)
        ]
    return obstacles

# 获取所有地图版本
def get_all_map_versions():
    # 数字版本的地图
    numeric_versions = [f"v{i}" for i in range(1, 23)]
    # 简单地图版本
    simple_versions = [f"simple{i}" for i in range(1, 6)]
    # 合并所有版本
    return numeric_versions + simple_versions

# 随机选择地图
def random_map(exclude=None):
    versions = get_all_map_versions()
    if exclude:
        versions = [v for v in versions if v != exclude]
    return random.choice(versions)
