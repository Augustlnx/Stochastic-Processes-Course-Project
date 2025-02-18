import numpy as np
import itertools
import pandas as pd

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

# 定义转移概率矩阵P
# 假设这是一个40x40的矩阵，您需要根据具体情况进行调整


# 求解平衡流率r(P的极限分布)

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

# 定义服务速率u_i
u = np.ones(40) * 1/8  # 默认服务速率
special_indices = [2, 7, 17, 22, 33, 36]
u[special_indices] = 1/4
u[10] = 1/10
u[30] = 1
u[[0, 20]] = 1/3
print(len(u))
r = steady_vector
# 计算吞吐率R
R = np.sum(r)

# 计算pi
pi = r

# 计算归一化常数C_N
N = 3
K = 40
i=0
def generate_combinations(K, N):
    """生成所有满足和为N的K个元素的组合"""
    
    # 保存最终的结果
    results = []
    
    # 3的整数分拆
    partitions = [
        [3],
        [2, 1],
        [1, 1, 1]
    ]
    
    for partition in partitions:
        # 如果分拆为 [3]
        if partition == [3]:
            for i in range(K):
                combo = [0] * K
                combo[i] = 3
                results.append(tuple(combo))
            print(len(results))
        # 如果分拆为 [2, 1]
        elif partition == [2, 1]:
            for i, j in itertools.permutations(range(K), 2):
                combo = [0] * K
                combo[i] = 2
                combo[j] = 1
                results.append(tuple(combo))
            print(len(results))
        # 如果分拆为 [1, 1, 1]
        elif partition == [1, 1, 1]:
            for indices in itertools.combinations(range(K), 3):
                combo = [0] * K
                for index in indices:
                    combo[index] = 1
                results.append(tuple(combo))
            print(len(results))
    
    return results

combos = generate_combinations(K, N)
print(len(combos))
C_N = 0
for combo in combos:
    product = 1
    for i in range(K):
        n_i = combo[i]
        product *= (pi[i] ** n_i) / (np.math.factorial(n_i) * (u[i] ** n_i))
    C_N += product

# 计算平稳分布
def P_N(combo, C_N, pi, u):
    product = 1
    for i in range(len(combo)):
        n_i = combo[i]
        product *= (pi[i] ** n_i) / (np.math.factorial(n_i) * (u[i] ** n_i))
    return  product/C_N 

# 计算所有组合的概率
probabilities = []
for combo in combos:
    probabilities.append((combo, P_N(combo, C_N, pi, u)))
#计算所有组合中概率最大的前3个组合，并展示该组合
probabilities_sorted = sorted(probabilities, key=lambda x: x[1], reverse=True)
top_3 = probabilities_sorted[:3]
print(top_3)
#找到上面top_3中的组合中，格子为3的位置
def find_3(combo):
    for i in range(len(combo)):
        if combo[i] == 3:
            return i
    return -1
# 找出前3个组合中，格子上数值3的下标
top_3_indices = [find_3(combo[0]) for combo in top_3]
print(top_3_indices)

#计算所有组合中只有1个格子为3的概率最大的前3个组合，并展示该组合
probabilities_sorted = sorted(probabilities, key=lambda x: sum([x[1] for i in range(len(x[0])) if x[0][i] == 3]), reverse=True)
top_3 = probabilities_sorted[:3]
print(top_3)

import matplotlib.pyplot as plt
# 初始化一个长度为40的数组
prob_sums = [0] * 40

# 遍历probabilities
for state, prob in probabilities:
    for idx, count in enumerate(state):
        prob_sums[idx] += count * prob

# 绘制柱状图
plt.figure(figsize=(10, 6))
plt.bar(range(40), prob_sums)
plt.xlabel('位置')
plt.ylabel('概率和')
plt.title('每个位置的概率和')
plt.show()


# 绘制双纵轴图
fig, ax1 = plt.subplots(figsize=(12, 6))

# 绘制第一个折线图：概率和
color = 'tab:red'
ax1.set_xlabel('位置')
ax1.set_ylabel('玩家平均人数', color="black")
ax1.plot(range(40), prob_sums, marker='o', color=color, label='平均玩家数')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(steady_df["编号"])
ax1.set_xticklabels(steady_df["名称"], rotation=90)

# 创建第二个纵轴
ax2 = ax1.twinx()

# 绘制第二个折线图：极限分布概率
color = 'tab:blue'
ax2.set_ylabel('极限分布概率', color="black")
ax2.plot(steady_df["编号"], steady_df["极限分布概率"], marker='o', color=color, label='极限分布概率')
ax2.tick_params(axis='y', labelcolor=color)

# 添加均值线
mean_prob = steady_df["极限分布概率"].mean()
ax2.axhline(mean_prob, color='r', linestyle='-', linewidth=1, label=f'极限分布概率均值: {mean_prob:.4f}')

# 合并图例
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

plt.title('每个位置的平均玩家数和与极限分布概率')
plt.tight_layout()
plt.show()