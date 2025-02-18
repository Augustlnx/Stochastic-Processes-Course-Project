import numpy as np
import matplotlib.pyplot as plt

# 定义骰子点数和的概率分布
dice_prob = np.array([0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0, 0]) / 36

# 初始化初始概率分布
pi = np.zeros(12)
pi[11] = 1

# 构建递推转移矩阵 A
A = np.zeros((12, 12))
for i in range(11):
    A[i, i+1] = 1
A[11, :] = dice_prob[1:13]

# 函数计算下一步概率分布
def next_pi(current_pi, A):
    return A @ current_pi

# 设置最大迭代次数和收敛判定条件
max_iterations = 10000
tolerance = 1e-10

# 存储所有分布
pi_distributions = []

# 迭代计算概率分布
for iteration in range(max_iterations):
    pi_new = next_pi(pi, A)
    pi_distributions.append(pi_new)
    
    # 判断是否收敛
    if np.linalg.norm(pi_new - pi, ord=1) < tolerance:
        convergence_iteration = iteration
        break
    pi = pi_new

# 输出收敛得到的结果
convergence_pi = pi

# 绘制收敛过程
iterations = len(pi_distributions)
pi_distributions = np.array(pi_distributions)
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

plt.figure(figsize=(10, 6))
for i in range(12):
    plt.plot(range(iterations), pi_distributions[:, i], label=f'前{i+1}位')
plt.xlabel('Iterations')
plt.ylabel('Probability')
plt.legend()
plt.title('第n个位置前12位的概率分布')
plt.grid(True)
plt.show()

convergence_iteration, convergence_pi


