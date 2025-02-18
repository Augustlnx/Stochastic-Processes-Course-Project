# The Game of Dice-Determined Fate: Probability Analysis and Strategy Application of Stochastic Processes in Monopoly

# 骰子下的命运游戏——随机过程在大富翁游戏中的概率分析与策略应用

## Introduce

这是一个XMU大二下课程 **应用随机过程** 的课程论文，本文只回顾并详解了经典书籍 ***Essentials of Stochastic Processes (Rick Durrett .2ndEdition)*** 和 ***STOCHASTIC PROCESSES (Ross .2ndEdition)*** 中提到的大富翁游戏的经典例题，并通过马尔科夫链和离出概率分析了一些游戏中的简单情形下的问题，并最后通过鞅论讨论了一种简化的资金流动模型和破产时间估计。作为概率领域的初学者，该课设必然存在很多缺陷和不足请读者多多包涵

This is a course paper for the Applied Stochastic Processes course in the second semester of sophomore year at XMU. This paper reviews and provides detailed explanations of the classic Monopoly game examples mentioned in Essentials of Stochastic Processes (Rick Durrett, 2nd Edition) and Stochastic Processes (Sheldon M. Ross, 2nd Edition). It analyzes some simple in-game scenarios using Markov chains and exit probabilities, and finally discusses a simplified capital flow model and bankruptcy time estimation using martingale theory.

注：本项目在文件夹***code***中附带的py文件为输出论文中结果所用到的所有Python代码，但因为时间过长缺少维护整理可能存在某种潜在的问题，仅提供分析示例

Note: The py file in ***code*** attached to this project is all the Python code used to output the results in the paper, but there may be some potential problems due to the lack of maintenance for too long, and only analysis examples are provided

不同语言版本的项目简介：
Project profiles in different languages:

- [中文版本](#项目简介)
- [English Version](#项目简介)


---

## 项目简介

### 论文预览：

![image](https://github.com/user-attachments/assets/67274c39-456b-4c52-bd2d-a704d2e2b1e1)



### 基本规则与假设

#### 地图设置：

当前流行的版本具有多种复杂的规则，作为一个简单应用这里只应用了一个老版本的简单地图来构建toy model：

<div align=center>
<img src="https://github.com/user-attachments/assets/e9fae56b-96c4-4a33-9439-1c537560cac1" width="460px">
</div>

如图所示，地图为边长11个格子的正方形轮廓，总共40个格子，并从起点GO开始依次映射为 0~39 的数字，便于后续讨论。

四个角落之外的、没有颜色（即非CC或CH）的格子为普通地产，我们事先将它们视作一般性的相同作用的格子。接下来着重介绍四个角落的特殊格子和带颜色的CC（获得宝箱卡）、CH（获得机会卡）：

- **GO**:玩家出发的起点，左上角。

- **G2J**：监狱传送门，当玩家该回合落在此格子，将会被传送至 JAIL。

- **JAIL**：监狱，当玩家该回合结束落在此格子（包括从G2J传送来的情况），将会被关进监狱，接下来两轮将无法行动。

- **FP**：FreePark，免费停泊地，当玩家该回合结束落在此格子，不会发生任何事情。

对于到达CC格子和CH格子将分别获得的宝箱卡和机会卡，我们暂时只关注会带来位移作用的卡片效果（宝箱卡2/16张，机会卡10/16张），例如“移动到监狱JAIL”，并忽略其他类型卡片的效果。

#### 游戏流程：
游戏开始，三位玩家每人有 1600元 作为初始资产。

- **掷骰**：每位玩家每轮开始时，同时掷出两颗骰子，并向前行走骰子点数之和。

- **购买地产**：玩家到达无人拥有的地产（格子），默认将在资金允许的情况下购买地产。

- **过路费**：当某一玩家回合结束的落点在其他玩家的地产上，需要向对方缴纳过路费 200元，相应也会在每一回合收到由其他玩家触发并缴纳的过路费。

- **破产**：当某轮中，玩家需要缴纳过路费但资金不足时，将缴纳剩余所有资金并宣告破产退出游戏。


#### 游戏模型假设

每名玩家行动相互独立，即每轮前进步数仅由掷骰子的点数决定。
正常游戏中获得机会卡和宝箱卡需要从牌堆中抽卡，为了实现马尔科夫性的构造，我们视作每次抽卡为有放回和重新洗牌的抽取。


### 关注的问题

这一部分具有比较浓郁的建模国赛风格……

在大富翁游戏中，我们关心并提出以下问题：

- **问题 1**：
  对于玩家来说，他们希望估计自己在若干轮后所处位置的概率，从而对自己的资产进行合理的规划，有如下情况：
  
  - 考虑一个简化的规则，对于所处某位置 i 的玩家 A，如果只通过掷骰子移动，在相当多轮后他在超过上一轮位置 i 时，超过步数的数值分布是多少？
  - 如果玩家位于格子 i，在到达格子 10（监狱） 前到达格子 0（起点） 的概率是多少？大概要过几轮？
  - 假设游戏刚刚开始，到所有格子都被三位玩家访问并购买时，大概过去了多久？

- **问题 2**：
长远来说，当游戏进行到相当多轮的时候玩家会出现在哪个位置？我们可以从不同的视角建立模型来考虑这个问题：

  - 从离散的轮数角度来看，在某轮中玩家停留在每个格子的概率分布如何？如果当前位于格子 i，大概需要多少轮才能重新回到这里？
  - 从连续的时间角度来看，在相当长时间后，三个玩家停留在地图上的位置有怎样的概率分布？根据这一结果，玩家应该制定怎样的购买地产策略来使得自己尽可能获利？

- **问题 3**：
我们开始考虑玩家资金转移的过程，我们希望知道对于初始资金相同的三位玩家，给定地图上某一种地产的分布情况，当有一位玩家因为破产退出游戏需要多少时间？当游戏结束，即出现两位玩家破产时，又需要多少时间？

注：规则和问题1的前置分析参考了[project euler 84](https://projecteuler.net/problem%3D84)，这一部分的核心做法同知乎文章[使用马尔科夫链分析大富翁游戏](https://zhuanlan.zhihu.com/p/355680952)基本一致（事实上这个简单的模型似乎也只能这么处理），问题1.1来自于***Essentials of Stochastic Processes (Rick Durrett .2ndEdition 中译版)*** P95 例3.10，这一部分本文采用了两种方法从头详细推导并验证了结果的一致性；问题2.1来自于 ***Essentials of Stochastic Processes (Rick Durrett .2ndEdition)*** P28 例1.28，原书采用了蒙特卡洛的方法，本文采用了通过马尔科夫链直接进行的理论推导并验证了结果的一致性，问题2.2的分析方法应用了 ***STOCHASTIC PROCESSES (Ross .2ndEdition)*** 的封闭排队网络理论；问题3的建模参考了知乎文章[三人赌博交换硬币](https://zhuanlan.zhihu.com/p/461404452)的模型，在这里添加了资产交换的金额不固定等扩展条件，并推广了利用该鞅论分析的一般性情境。

以上具体解决方法请参照原文。

---

## Project Introduction

### Paper preview:

![image](https://github.com/user-attachments/assets/67274c39-456b-4c52-bd2d-a704d2e2b1e1)



### Basic Rules and Assumptions

#### Map Layout:

The current popular versions have many complex rules. For simplicity, this analysis uses an old version of the Monopoly map to construct a toy model:

<div align=center>
<img src="https://github.com/user-attachments/assets/e9fae56b-96c4-4a33-9439-1c537560cac1" width="460px">
</div>

As shown in the figure, the map is a square contour with 11 squares on each side, totaling 40 squares, numbered from 0 to 39 starting from GO for ease of discussion.

The squares outside the four corners that are not colored (i.e., not CC or CH) are ordinary properties, which we treat as having the same general function. The following focuses on the special corner squares and the colored CC (Community Chest) and CH (Chance) squares:

- **GO**: The starting point of the players, located at the top-left corner.
- **G2J**: The Go To Jail square. When a player lands on this square, they are immediately sent to JAIL.
- **JAIL**: The jail square. When a player lands on this square (including being sent here from G2J), they are put in jail and cannot move for the next two rounds.
- **FP**: Free Parking. When a player lands on this square, nothing happens.

For the Community Chest (CC) and Chance (CH) squares, we only consider the cards that cause movement (2 out of 16 Community Chest cards and 10 out of 16 Chance cards), such as "Go to Jail," and ignore the effects of other types of cards.

#### Game Process:
The game begins with three players, each starting with 1600 units of currency as their initial capital.

- **Rolling Dice**: At the start of each round, players roll two dice and move forward by the sum of the dice.
- **Buying Properties**: When a player lands on an unowned property square, they will purchase it if they have sufficient funds.
- **Rent Payment**: If a player lands on a property owned by another player, they must pay a rent of 200 units of currency to the owner. Players also receive rent payments from other players landing on their properties.
- **Bankruptcy**: If a player cannot afford the rent payment during a round, they will pay all their remaining funds and declare bankruptcy, exiting the game.

#### Game Model Assumptions

- Each player's actions are independent, meaning the number of steps taken in each round is solely determined by the dice roll.
- In normal gameplay, drawing Community Chest or Chance cards involves drawing from a deck. To construct a Markov process, we assume that each card draw is with replacement and reshuffling.

---

### Research Questions

This section has a strong flavor of modeling for national competitions.

In the game of Monopoly, we are interested in the following questions:

- **Question 1**:
  Players wish to estimate the probability of their position after several rounds to make reasonable financial plans. The specific scenarios are:
  
  - Under a simplified rule, for a player A at position i, if they only move by rolling dice, what is the distribution of the number of steps they exceed position i after many rounds?
  - If a player is at square i, what is the probability of reaching square 0 (GO) before reaching square 10 (Jail)? How many rounds does it take on average?
  - Assuming the game just started, how long does it take for all squares to be visited and purchased by the three players?

- **Question 2**:
  In the long run, after many rounds, where will the players be located? We can model this problem from different perspectives:

  - From a discrete round perspective, what is the probability distribution of a player staying on each square in a given round? If currently at square i, how many rounds does it take to return to this square on average?
  - From a continuous time perspective, what is the probability distribution of the three players' positions on the map after a long time? Based on this result, what purchasing strategy should players adopt to maximize their profits?

- **Question 3**:
  We begin to consider the process of players' capital transfer. We want to know, for three players with the same initial capital, given a certain distribution of properties on the map, how long does it take for one player to go bankrupt and exit the game? When the game ends, i.e., when two players go bankrupt, how long does it take?

**Note**: The rules and preliminary analysis of Question 1 refer to [Project Euler 84](https://projecteuler.net/problem=84). The core approach is consistent with the article [Using Markov Chains to Analyze Monopoly](https://zhuanlan.zhihu.com/p/355680952) on Zhihu. Question 1.1 is from ***Essentials of Stochastic Processes (Rick Durrett, 2nd Edition, Chinese Translation)***, page 95, Example 3.10. This paper provides detailed derivations using two methods and verifies the consistency of the results. Question 2.1 is from ***Essentials of Stochastic Processes (Rick Durrett, 2nd Edition)***, page 28, Example 1.28. The original book used a Monte Carlo method, while this paper provides a theoretical derivation using Markov chains and verifies the consistency of the results. The analysis of Question 2.2 applies the closed queuing network theory from ***Stochastic Processes (Ross, 2nd Edition)***. The modeling of Question 3 refers to the article [Three People Gambling and Exchanging Coins](https://zhuanlan.zhihu.com/p/461404452) on Zhihu, with added extensions such as variable amounts of capital exchange and generalizations of the application of martingale theory.

For specific solutions, please refer to the original text.
