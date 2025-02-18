import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 定义骰子点数和的概率分布
dice_probs = np.zeros(13)
dice_probs[2] = 1/36
dice_probs[3] = 2/36
dice_probs[4] = 3/36
dice_probs[5] = 4/36
dice_probs[6] = 5/36
dice_probs[7] = 6/36
dice_probs[8] = 5/36
dice_probs[9] = 4/36
dice_probs[10] = 3/36
dice_probs[11] = 2/36
dice_probs[12] = 1/36

# 初始化掷骰子转移矩阵 D
D = np.zeros((40, 40))

# 计算掷骰子的转移概率矩阵 D
for i in range(40):
    for dice_sum in range(2, 13):
        D[i, (i + dice_sum) % 40] += dice_probs[dice_sum]

# 初始化抽卡和特殊格子转移矩阵 C
C = np.eye(40)

# 定义辅助函数来找到当前格子之后的第一个目标格子
def find_next(current, targets):
    for i in range(1, 40):
        if (current + i) % 40 in targets:
            return (current + i) % 40
    return current

# 目标格子集合
railroads = [5, 15, 25, 35]
utilities = [12, 28]

# 宝箱卡 (CC)
CC_squares = [2, 17, 33]
CC_transitions = [(0, 1/8), (10, 1/8)]
sum_prob = 0
for square in CC_squares:
    for target, prob in CC_transitions:
        C[square, target] += prob
        sum_prob += prob
    C[square, square] = 1-2/8

# 机会卡 (CH)
sum_prob = 0
CH_squares = [7, 22, 36]
CH_transitions = [(0, 1/16), (10, 1/16), (11, 1/16), (24, 1/16), (39, 1/16), (5, 1/16), (-3, 1/16)]
for square in CH_squares:
    for target, prob in CH_transitions:
        if target == -3:
            C[square, (square - 3) % 40] += prob
            sum_prob += prob
        else:
            C[square, target] += prob
            sum_prob += prob
    
    next_railroad = find_next(square, railroads)
    C[square, next_railroad] += 1/8
    
    next_utility = find_next(square, utilities)
    C[square, next_utility] += 1/16

    C[square, square] = 1-10/16

# 特殊格子 G2J
C[30] = 0
C[30, 10] = 1
C[30, 30] = 0
# 计算最终的转移矩阵 P
P = np.dot(D, C)
labels = [
    "GO", "A1", "CC1", "A2", "T1", "R1", "B1", "CH1", "B2", "B3", "JAIL", "C1", "U1", "C2", 
    "C3", "R2", "D1", "CC2", "D2", "D3", "FP", "E1", "CH2", "E2", "E3", "R3", "F1", "F2", 
    "U2", "F3", "G2J", "G1", "G2", "CC3", "G3", "R4", "CH3", "H1", "T2", "H2"
]
import numpy as np
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Bar3D
from pyecharts.commons.utils import JsCode

# 假设 P 是转移概率矩阵
# labels 是格子名称列表
P = np.dot(D, C)
labels = [
    "GO", "A1", "CC1", "A2", "T1", "R1", "B1", "CH1", "B2", "B3", "JAIL", "C1", "U1", "C2", 
    "C3", "R2", "D1", "CC2", "D2", "D3", "FP", "E1", "CH2", "E2", "E3", "R3", "F1", "F2", 
    "U2", "F3", "G2J", "G1", "G2", "CC3", "G3", "R4", "CH3", "H1", "T2", "H2"
]

# 准备3D柱状图数据
data = []
for i in range(len(labels)):
    for j in range(len(labels)):
        data.append([i, j, P[i, j]])

# 创建3D柱状图
c = (
    Bar3D(init_opts=opts.InitOpts(width="900px", height="600px"))
    .add(
        series_name="转移概率",
        data=data,
        xaxis3d_opts=opts.Axis3DOpts(type_="category", data=labels, name="起点"),
        yaxis3d_opts=opts.Axis3DOpts(type_="category", data=labels, name="终点"),
        zaxis3d_opts=opts.Axis3DOpts(type_="value", name="概率"),
        grid3d_opts=opts.Grid3DOpts(
            width=100,
            height=100,
            depth=100,
            rotate_speed=5
    ),
    )
    .set_global_opts(
        title_opts=opts.TitleOpts("转移概率 3D柱状图"),
        visualmap_opts=opts.VisualMapOpts(
            max_=np.max(P),
            range_color=[
                "#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8",
                "#ffffbf", "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026",
            ],
        ),
        tooltip_opts=opts.TooltipOpts(
            formatter=JsCode(
                "function(params){return params.seriesName + ' 起点: ' + params.value[1] + ' 终点: ' + params.value[0] + ' 概率: ' + params.value[2].toFixed(4);}"
            )
        )
    )
)

# 渲染图表到HTML文件
file_path = r"C:\Users\August\Desktop\学习\随机过程\P_3D柱状图.html"  # 请修改为实际的保存路径
c.render(file_path)
