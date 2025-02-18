import numpy as np

# 定义模拟次数
num_simulations = 10000

# 定义每个玩家初始金额
initial_money = 5000

# 定义转移概率和金额
probabilities = [8/70, 12/70, 4/70, 16/70, 10/70, 10/70, 10/70]
p0 = [[0,8/70,10/70],
      [12/70,0,4/70],
      [10/70,16/70,0],]
print(p0)
a0 = [[0,300,250],
      [200,0,400],
      [250,100,0]]
print(a0)
transfers = [
    (0, 1, 300),
    (1, 0, 200),
    (1, 2, 400),
    (2, 1, 100),
    (0, 2, 250),
    (2, 0, 250),
    (-1, -1, 0)  # No transfer
]

probabilities = [8/60, 12/60, 4/60, 16/60, 10/60, 10/60, 0]

f0 = 0
for i in range(len(probabilities)):
    f0 += probabilities[i]*transfers[i][2]**2
print(f0)
#不包含轮空的概率

#平均转移状况
probabilities = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6, 0]

transfers = [
    (0, 1, 200),
    (1, 0, 200),
    (1, 2, 200),
    (2, 1, 200),
    (0, 2, 200),
    (2, 0, 200),
    (-1, -1, 0)  # No transfer
]
#probabilities每个值乘以transfers中转移金额的平方累加起来：

#包含轮空的平均转移状况
probabilities = [1/7,1/7,1/7,1/7,1/7,1/7,1/7]
f0 = 6/7*200*200
print(initial_money**2*3/f0)

print(sum(probabilities))
# 定义模拟一轮游戏的函数

def simulate_game():
    money = [initial_money, initial_money, initial_money]
    rounds = 0
    rounds1 = 0
    sump = 0
    while all(m > 0 for m in money):
        
        rounds += 1
        
        # 随机选择一个转移规则
        transfer_idx = np.random.choice(len(probabilities), p=probabilities)
        payer, receiver, amount = transfers[transfer_idx]
        if payer != -1 and money[payer] > 0:
            # 转移金额，确保玩家有足够的钱进行转移
            money[payer] -= amount
            money[receiver] += amount#min(money[payer], amount)
    # 找到剩余的两个玩家的财富
    remaining_money = [m for m in money if m > 0]
    wealth_product = remaining_money[0] * remaining_money[1] if len(remaining_money) == 2 else 0
    #计算三个玩家俩俩财富乘积
    wealth_product = money[0]*money[1]+money[1]*money[2]+money[0]*money[2]
    #根据大于0位置的索引确定剩下的玩家，继续游戏
    remain_player = [i for i, m in enumerate(money) if m > 0]
    leave_player = [i for i, m in enumerate(money) if m <= 0]
    sump = probabilities[6]+p0[remain_player[0]][remain_player[1]]+p0[remain_player[1]][remain_player[0]]
    p2 = [p0[remain_player[0]][remain_player[1]]/sump, p0[remain_player[1]][remain_player[0]]/sump, probabilities[6]/sump]
    a2 = [a0[remain_player[0]][remain_player[1]], a0[remain_player[1]][remain_player[0]], 0]
    f1 = p2[0]*a2[0]**2+p2[1]*a2[1]**2+p2[2]*0
    tra2 = [
    (remain_player[0], remain_player[1], a2[0]),
    (remain_player[1], remain_player[0], a2[1]),
    (-1, -1, 0)  # No transfer
    ]
    remain_player.append(3)
    rounds1 = rounds
    remaining_money = money
    remaining_money[leave_player[0]] = 1
    while all(m>0 for m in remaining_money):
        rounds += 1
        # 随机选择一个转移规则
        transfer_idx = np.random.choice(len(p2), p=p2)
        payer, receiver, amount = tra2[transfer_idx]
        if payer != -1 and money[payer] > 0:
            # 转移金额，确保玩家有足够的钱进行转移
            remaining_money[payer] -= amount
            remaining_money[receiver] += amount#min(money[payer], amount
    
    
    return rounds1,rounds, wealth_product,f1

# 运行模拟
initial_money = 1600
num_simulations = 10000
results = [simulate_game() for _ in range(num_simulations)]

# 提取结果
times1, times2,wealth_products,f1 = zip(*results)

# 计算平均值
mean_time = np.mean(times1)
over_time = np.mean(times2)
mean_wealth_product = np.mean(wealth_products)

print('Mean time until a player exits:', mean_time)
print('Mean time until a game over:', over_time)
print('Mean wealth product of the remaining two players:', mean_wealth_product)
print((initial_money*initial_money*3-mean_wealth_product)/f0)
print((initial_money*initial_money*3-mean_wealth_product)/40000)
#下界估计
f1max = max(f1)
f1min = min(f1)
S0 = initial_money*initial_money*3
K = (3 * initial_money + np.max(a0))
print((S0-initial_money*initial_money*9/4)/f0)
print(S0/f1+((S0-(3/2*initial_money+np.max(a0))**2)/f0))
print((initial_money*initial_money*3-2*50*300)/f0)
print((S0+50*(3*initial_money-50))/f1max + (S0 - K**2/4) / f0*(1-f0/f1min))