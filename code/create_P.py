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
P[:,30]


#转移矩阵热力图展示
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap
def plot_heatmap_with_values(matrix, title, filename):
    plt.figure(figsize=(10, 8), dpi=300)  # 增加 dpi 参数
    # 创建自定义的颜色映射，最小值为白色
    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap",
        [(0, "white"), (0.5, "yellow"), (1, "red")]
    )
    ax = sns.heatmap(matrix, cmap=cmap, cbar=True, square=True, linewidths=.5, linecolor='black', annot=True, fmt='.2f', annot_kws={"size": 8})  # 修改 annot_kws 的 size 值以调整文本大小
    
    # 获取每个单元格的文本对象
    for text in ax.texts:
        value = float(text.get_text())
        if value == 0:
            text.set_text('')
        else:
            text.set_text(f"{value}")
            text.set_fontsize(4.5)  # 这里设置你想要的字体大小
    
    plt.title(title, fontsize=16)  # 增加标题字体大小
    plt.xlabel('To Square', fontsize=14)  # 增加 x 轴标签字体大小
    plt.ylabel('From Square', fontsize=14)  # 增加 y 轴标签字体大小
    plt.xticks(fontsize=12)  # 调整 x 轴刻度字体大小
    plt.yticks(fontsize=12)  # 调整 y 轴刻度字体大小
    plt.tight_layout()  # 确保图形不被截断
    
    plt.savefig(filename, format='pdf')  # 保存为 PDF 格式
    plt.close()  # 关闭当前图形，避免重复显示

# 定义保存路径
save_path = r'C:\Users\August\Desktop\学习\随机过程'

# 确保路径存在
os.makedirs(save_path, exist_ok=True)

# 绘制并保存掷骰子转移矩阵 D 的热图
plot_heatmap_with_values(D, 'Dice Roll Transition Probability Matrix (D)', os.path.join(save_path, 'D_heatmap.pdf'))

# 绘制并保存抽卡和特殊格子转移矩阵 C 的热图
plot_heatmap_with_values(C, 'Card and Special Square Transition Probability Matrix (C)', os.path.join(save_path, 'C_heatmap.pdf'))

# 绘制并保存最终转移矩阵 P 的热图
plot_heatmap_with_values(P, 'Final Transition Probability Matrix (P)', os.path.join(save_path, 'P_heatmap.pdf'))

#检查行和概率为1
row_sums = np.sum(P, axis=1)
print(row_sums)

# 计算 P 的三次方
P_cubed = np.linalg.matrix_power(P, 3)
# 展示第一行
first_row = P_cubed[0]
data = first_row
# 状态空间
states = list(range(40))

# 创建柱状图
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
plt.figure(figsize=(12, 6))
plt.bar(states, data, color='skyblue')
# 设置x轴刻度
plt.xticks(states, rotation=90)
# 添加标题和标签

plt.title('P^3 的第一行柱状图')
plt.xlabel('状态')
plt.ylabel('概率')

# 显示网格
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 显示图形
plt.show()






import numpy as np
import matplotlib.pyplot as plt

map_size = 40
simulations = 10000

steps_list = []
total_rolls = 0




for _ in range(simulations):
    # 生成初始网格
    visited = [False] * map_size
    current_pos = [0, 0, 0]
    initial_visitedset = [0,2,7,10,17,20,22,30,33,36]
    for i in initial_visitedset:
        visited[i] = True
    steps = 0
    visited_count = len(initial_visitedset)

    while visited_count < map_size:
        for i in range(len(current_pos)):
            # 根据转移矩阵确定下一个位置
            current_pos[i] = np.random.choice(range(map_size), p=P[current_pos[i]])
            if not visited[current_pos[i]]:
                visited[current_pos[i]] = True
                visited_count += 1
        steps += 1
    
    total_rolls += steps
    steps_list.append(steps)

average_steps = total_rolls / simulations
print(average_steps)

# Create scatter plot
plt.figure(figsize=(12, 6))
plt.scatter(range(simulations), steps_list, alpha=0.6, edgecolors='w', linewidth=0.5)
plt.xlabel('Simulation Index')
plt.ylabel('Number of Steps')
plt.title('Steps to Visit All Grid Cells in 40-Cell Cycle Map (Three Players, Transition Matrix)')
plt.show()




#计算极限分布
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_steady_vector(P, tol=1e-10):
    vec_i = np.array([1] + [0] * (P.shape[0] - 1))
    iterations = [vec_i]
    while True:
        vec_s = vec_i @ P
        if(vec_s[30]>0):
            print(vec_s[30])
        iterations.append(vec_s)
        if np.linalg.norm(vec_s - vec_i) < tol:
            return vec_s, iterations
        vec_i = vec_s

steady_vector, iterations = get_steady_vector(P)

# Convert the steady vector to a DataFrame
labels = [
    "GO", "A1", "CC1", "A2", "T1", "R1", "B1", "CH1", "B2", "B3", "JAIL", "C1", "U1", "C2", 
    "C3", "R2", "D1", "CC2", "D2", "D3", "FP", "E1", "CH2", "E2", "E3", "R3", "F1", "F2", 
    "U2", "F3", "G2J", "G1", "G2", "CC3", "G3", "R4", "CH3", "H1", "T2", "H2"
]

steady_df = pd.DataFrame({
    "编号": range(40),
    "名称": labels,
    "极限分布概率": steady_vector,
    "平均访回时间": range(40)
})
for i in range(40):
    steady_df.loc[i, "平均访回时间"] = 1 / steady_vector[i]
    if steady_vector[i] == 0:
        steady_df.loc[i, "平均访回时间"] = 0

print(steady_df)

sum(steady_vector)
print(steady_vector[30])
# Plotting the line chart with points
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

#画出极限概率分布折线图
plt.figure(figsize=(12, 6))
plt.plot(steady_df["编号"], steady_df["极限分布概率"], marker='o')
plt.xticks(steady_df["编号"], steady_df["名称"], rotation=90)
plt.xlabel('编号')
plt.ylabel('极限分布概率')
plt.title('极限分布概率')
# 关闭网格
plt.grid(False)
# 计算概率均值并添加横线
mean_prob = steady_df["极限分布概率"].mean()
plt.axhline(mean_prob, color='r', linestyle='--', linewidth=1, label=f'均值: {mean_prob:.4f}')
# 添加图例
plt.legend()
plt.tight_layout()
plt.show()

#画出平均访回时间柱状图
plt.figure(figsize=(12, 6))
plt.bar(steady_df["编号"], steady_df["平均访回时间"])
plt.xticks(steady_df["编号"], steady_df["名称"], rotation=90)
plt.xlabel('编号')
plt.ylabel('平均访回时间')
plt.title('平均访回时间')
# 关闭网格
plt.grid(False)
#在柱状图上添加数值（保留1位小数）
for x, y in enumerate(steady_df["平均访回时间"]):
    plt.text(x, y + 0.15, f'{y:.1f}', ha='center', va= 'bottom')

# 计算平均时间均值并添加横线
mean_time = steady_df["平均访回时间"].mean()
plt.axhline(mean_time, color='r', linestyle='--', linewidth=1, label=f'均值:{mean_time:.4f}')
# 添加图例
plt.legend()
plt.tight_layout()
plt.show()

#最终画图，将两个图合并，同时展示概率分布和平均访问时间，分别用折线图和柱状图展示
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

# 创建画布
fig, ax1 = plt.subplots(figsize=(12, 6))

# 画平均访问时间柱状图（使用蓝色）
ax1.bar(steady_df["编号"], steady_df["平均访回时间"], color='#6AA4EB', zorder=1)
ax1.set_xlabel('编号')
ax1.set_ylabel('平均访问时间', color='black')  # 设置y轴标签颜色与柱状图一致
# 关闭网格，添加数值
ax1.grid(False)
for x, y in enumerate(steady_df["平均访回时间"]):
    ax1.response(x, y + 0.15, f'{y:.1f}', ha='center', va='bottom', color='#3C6AC8')  # 设置文本颜色与柱状图一致
# 计算平均时间均值并添加横线
mean_time = steady_df["平均访回时间"].mean()
ax1.axhline(mean_time, color='#C91D32', linestyle='-', linewidth=1, label=f'平均访问时间均值:{mean_time:.4f}')

# 创建第二个坐标轴
ax2 = ax1.twinx()

# 画概率分布折线图（使用黄色）
ax2.plot(steady_df["编号"], steady_df["极限分布概率"], marker='o', color='#FEDB61', zorder=2)
ax2.set_ylabel('极限分布概率', color='black')  # 设置y轴标签颜色与折线图一致
# 关闭网格
ax2.grid(False)
# 计算概率均值并添加横线
mean_prob = steady_df["极限分布概率"].mean()
ax2.axhline(mean_prob, color='#7CDED7', linestyle='-', linewidth=1, label=f'极限分布概率均值: {mean_prob:.4f}')

# 设置 x 轴标签
ax1.set_xticks(steady_df["编号"])
ax1.set_xticklabels(steady_df["名称"], rotation=90)

# 设置标题（统一标题）
fig.suptitle('极限分布概率与平均返回轮数', fontsize=16)

# 添加图例（合并图例，并移动到合适位置）
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

# 调整布局
plt.tight_layout()

# 显示图表
plt.show()



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import time


map_size = 40
simulations = 1  # 为了演示，我们只运行一次模拟

# 初始化绘图
fig, ax = plt.subplots(figsize=(8, 8))
cmap = mcolors.ListedColormap(['white', 'yellow', 'red'])
norm = mcolors.BoundaryNorm([0, 0.5, 1.5, 2], cmap.N)

# 编号对应的名称
labels = [
    "GO", "A1", "CC1", "A2", "T1", "R1", "B1", "CH1", "B2", "B3",
    "JAIL", "C1", "U1", "C2", "C3", "R2", "D1", "CC2", "D2", "D3",
    "FP", "E1", "CH2", "E2", "E3", "R3", "F1", "F2", "U2", "F3",
    "G2J", "G1", "G2", "CC3", "G3", "R4", "CH3", "H1", "T2", "H2"
]

# 生成初始网格
grid = np.zeros((map_size, 1))
visited = [False] * map_size
current_pos = [0, 0, 0]
initial_visitedset = [0,2,7,10,17,20,22,30,33,36]
for i in initial_visitedset:
    grid[i] = 2
    visited[i] = True
visited_count = len(initial_visitedset)
steps = 0

def update_grid(current_positions, visited):
    grid = np.zeros((map_size, 1))
    for pos in current_positions:
        grid[pos] = 2
    for idx, v in enumerate(visited):
        if v:
            grid[idx] = max(grid[idx], 1)
    return grid

def animate(i):
    global steps, visited_count
    ax.clear()
    steps += 1

    if visited_count < map_size:
        for j in range(len(current_pos)):
            # 根据转移矩阵确定下一个位置
            current_pos[j] = np.random.choice(range(map_size), p=P[current_pos[j]])
            if not visited[current_pos[j]]:
                visited[current_pos[j]] = True
                visited_count += 1

    grid = update_grid(current_pos, visited)
    ax.imshow(grid, cmap=cmap, norm=norm)
    
    # 添加网格分隔线
    for x in range(map_size):
        ax.axhline(x - 0.5, color='black', linewidth=1)
    ax.axvline(-0.5, color='black', linewidth=1)
    ax.axvline(0.5, color='black', linewidth=1)

    # 添加编号标签
    for idx, label in enumerate(labels):
        ax.text(0, idx, label, ha='center', va='center', color='black')

    ax.set_xticks([])
    ax.set_yticks([])
    
    # 显示移动轮数
    ax.set_title(f"Steps: {steps}")

    # 暂停0.5秒以便观察
    time.sleep(0.5)

    # 停止动画如果所有格子都被访问了
    if visited_count >= map_size:
        ani.event_source.stop()

# 创建动画
ani = FuncAnimation(fig, animate, frames=200, interval=200, repeat=False)

# 显示动画
plt.show()


#计算离出概率、离出分布
#对P进行修正，去掉第0行和第10行为P0
P0 = np.delete(P, [0, 10], axis=0)
#打印维度
print(P0.shape)
#提取R1为P0的第0列
R1 = P0[:, 0]
#打印维度
print(R1.shape)
#删去P0的第0、10列
P1 = np.delete(P0, [0, 10], axis=1)
#打印维度
print(P1.shape)
#计算离出概率
#生成一个P1维数的单位矩阵
I = np.eye(P1.shape[0])
pro = np.dot(np.linalg.inv(I - P1), R1)
#计算离出时间
#生成一个P1长度的元素均为1的列向量
ones = np.ones(P1.shape[0])
T = np.dot(np.linalg.inv(I - P1), ones)

import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
# 标签
labels = [
     "A1", "CC1", "A2", "T1", "R1", "B1", "CH1", "B2", "B3",
     "C1", "U1", "C2", "C3", "R2", "D1", "CC2", "D2", "D3",
    "FP", "E1", "CH2", "E2", "E3", "R3", "F1", "F2", "U2", "F3",
    "G2J", "G1", "G2", "CC3", "G3", "R4", "CH3", "H1", "T2", "H2"
]

# 创建图像和坐标轴对象
fig, ax2 = plt.subplots()
# 绘制离出时间 T，使用蓝色的柱状图
ax2.bar(labels, T, color='#7898e1', alpha=1, label='Time')
ax2.set_ylabel('Time', color='black')


# 创建第二个坐标轴对象
ax1 = ax2.twinx()
# 绘制离出概率 pro，使用红色的折线图
ax1.plot(labels, pro, color='#eddd86', alpha=1,marker='o', label='Probability')
ax1.set_xlabel('Labels')
ax1.set_ylabel('Probability', color='black')


# 添加图例
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax1.get_legend_handles_labels()
lines = lines1 + lines2
labels = labels1 + labels2
ax2.legend(lines, labels, loc='upper left')

# 设置标题
plt.title('每个格子到达起点的离出概率与离出时间')

# 旋转 x 轴标签，以防止重叠
plt.xticks(rotation=90)

# 显示图像
plt.show()