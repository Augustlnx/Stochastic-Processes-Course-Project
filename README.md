# The Game of Dice-Determined Fate: Probability Analysis and Strategy Application of Stochastic Processes in Monopoly

# 骰子下的命运游戏——随机过程在大富翁游戏中的概率分析与策略应用

## Introduce

这是一个XMU大二下课程 **应用随机过程** 的课程论文，本文只回顾并详解了经典书籍 ***Essentials of Stochastic Processes (Rick Durrett .2ndEdition)*** 和 ***STOCHASTIC PROCESSES (Ross .2ndEdition)*** 中提到的大富翁游戏的经典例题，并通过马尔科夫链和离出概率分析了一些游戏中的简单情形下的问题，并最后通过鞅论讨论了一种简化的资金流动模型和破产时间估计。作为概率领域的初学者，该课设必然存在很多缺陷和不足请读者多多包涵

This is a course paper for the Applied Stochastic Processes course in the second semester of sophomore year at XMU. This paper reviews and provides detailed explanations of the classic Monopoly game examples mentioned in Essentials of Stochastic Processes (Rick Durrett, 2nd Edition) and Stochastic Processes (Sheldon M. Ross, 2nd Edition). It analyzes some simple in-game scenarios using Markov chains and exit probabilities, and finally discusses a simplified capital flow model and bankruptcy time estimation using martingale theory.

注：本项目在文件夹***code***中附带的py文件为输出论文中结果所用到的所有Python代码，但因为时间过长缺少维护整理可能存在某种潜在的问题，仅提供分析示例

Note: The py file in ***code*** attached to this project is all the Python code used to output the results in the paper, but there may be some potential problems due to the lack of maintenance for too long, and only analysis examples are provided

不同语言版本的项目简介：
Project profiles in different languages:

- [English Version](#项目简介)
- [中文版本](#中文)

---

## 项目简介

### 论文预览：

![image](https://github.com/user-attachments/assets/67274c39-456b-4c52-bd2d-a704d2e2b1e1)

### 基本规则与假设

#### 地图设置：

当前流行的版本具有多种复杂的规则，作为一个简单应用这里只应用了一个老版本的简单地图来构建toy model：

![map](https://github.com/user-attachments/assets/e9fae56b-96c4-4a33-9439-1c537560cac1)

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
我们开始考虑玩家资金转移的过程，我们希望知道对于初始资金均为 K₀ 的三位玩家，给定地图上某一种地产的分布情况，当有一位玩家因为破产退出游戏需要多少时间？当游戏结束，即出现两位玩家破产时，又需要多少时间？



---

## 中文

// Your Chinese content here
