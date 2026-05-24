# Tracking 模块总结：从贝叶斯网络到粒子滤波

> 对应 PPCA-AIPacMan tracking 模块 Q1~Q11。
> 核心目标：通过噪声距离观测追踪不可见幽灵的位置。

---

## 一、知识体系总览

tracking 模块是一条完整的技术线索：**贝叶斯网络 → 精确推断 → 隐马尔可夫模型 → 粒子滤波**。

```
贝叶斯网络（结构建模）
    ↓
变量消除（精确推断，小问题可行）
    ↓
HMM 前向算法（时序精确推断，每步 O(N²)）
    ↓
粒子滤波（近似推断，O(M·N)，M=粒子数，N=状态数）
```

### 1.1 贝叶斯网络 (Bayes Net)

**问题**：Pacman 只能通过噪声距离传感器感知幽灵位置。如何用概率图模型描述这个场景？

**贝叶斯网络**是一个有向无环图 (DAG)，节点是随机变量，边表示条件依赖。对于追踪问题，结构如下：

```
       Pacman
      / |  \ \
     /  |   \ \
  G0   G1    O0 O1
   \         /
    +---O0--+
```

- **Pacman**：根节点，Pacman 的已知位置（网格中任意格）
- **Ghost0 / Ghost1**：两个幽灵的未知位置
- **Observation0 / Observation1**：到每个幽灵的噪声曼哈顿距离读数

关键设计决策：两个幽灵之间没有边——即**条件独立假设**。这简化了推断，但丢失了幽灵之间可能的位置相关性。

### 1.2 因子 (Factor) 与条件概率表 (CPT)

**因子**存储了一个概率表，形式为 `f(X₁,...,Xₘ, y₁,...,yₙ | Z₁,...,Zₚ)`。竖线左边是非条件变量，右边是条件变量。表中每一行是一个条件赋值下的概率值。

**条件概率表 (CPT)** 是一种特殊的因子：只有一个非条件变量，且在每组条件赋值下，所有行的概率之和为 1。例如 `P(Traffic | Raining, Ballgame)`。

因子的两个核心操作：

**连接 (Join)**：`joinFactors(f₁, f₂)` = 将两个因子相乘。新的无条件变量 = 并集；新的条件变量 = 并集 - 出现在无条件集合中的变量。表中的每一行 = 两个输入因子对应行的概率乘积。

**消除 (Eliminate)**：`eliminate(f, X)` = 对变量 X 求和（边缘化）。新因子中 X 被移出无条件变量集合，表中每一行是对 X 所有可能值的概率求和。

### 1.3 变量消除 (Variable Elimination)

**问题**：给定贝叶斯网络和证据，如何计算 `P(查询变量 | 证据变量)`？

**枚举法**的问题：先把所有 CPT 合并成一个巨大的联合概率表（指数级大小），再对隐藏变量逐个求和。表的大小随变量数指数增长——不可行。

**变量消除**的改进：交替进行"合并"和"消除"——每轮只处理一个隐藏变量：
1. 合并包含该变量的所有因子
2. 立即消除该变量（求和）
3. 用缩小的因子替换原来的因子
4. 处理下一个隐藏变量

**为什么更快**：中间因子的大小取决于当前涉及的变量数，而不是全部变量。消除了一个变量后，表立即缩小，后续操作更快。

### 1.4 隐马尔可夫模型 (HMM) 与前向算法

当追踪随时间推进时，贝叶斯网络在每一个时间步展开一个副本：

```
G_t → G_{t+1} → G_{t+2} → ...
 ↓      ↓          ↓
O_t    O_{t+1}    O_{t+2}
```

- **转移模型** `P(G_{t+1} | G_t)`：幽灵移动规则（不能穿墙、可被 Pacman 捕获进入监狱）
- **观测模型** `P(O_t | G_t, Pacman_t)`：噪声距离读数的概率分布

**精确推断（前向算法）**的每一步包含：
1. **预测步 (elapseTime)**：`P(G_{t+1}) = Σ_{g_t} P(G_{t+1} | g_t) · P(g_t)` — 按幽灵移动规则扩散信念
2. **更新步 (observeUpdate)**：`P(G_{t+1} | o_{t+1}) ∝ P(o_{t+1} | G_{t+1}) · P(G_{t+1})` — 用新观测修正信念

精确推断的代价：每个时间步要遍历所有可能的幽灵位置（~400 格），对每个位置计算转移概率（遍历邻居）。O(N²) 其中 N=格子数。

### 1.5 粒子滤波 (Particle Filter)

**核心思想**：用 M 个带权重的样本（粒子）近似表示概率分布，而不是跟踪完整的概率表。

**粒子滤波的一个完整周期**：

```
1. 预测步 (elapseTime)：每个粒子按转移模型随机移动
   对于 t 时刻的每个粒子位置 oldPos：
      newPosDist = P(G_{t+1} | G_t = oldPos)     ← 转移分布
      从 newPosDist 中随机采样 → 新粒子位置

2. 更新步 (observeUpdate)：根据观测重新加权并重采样
   对于每个粒子位置 pos：
      weight(pos) = P(obs | ghost = pos)           ← 观测似然
   如果所有权重之和为 0 → 重新初始化（粒子全部"死亡"）
   否则从加权分布中重采样 M 个粒子（有放回）
```

**粒子滤波 vs 精确推断的权衡**：

| | 精确推断 | 粒子滤波 |
|------|---------|--------|
| 表示 | 完整概率表（每格一个值） | M 个样本 |
| 精度 | 精确 | 近似（M 越大越准） |
| 每步复杂度 | O(N²) | O(M·N) |
| N=400, M=300 | 160,000 次操作 | 120,000 次操作 |
| 粒子退化 | 不存在 | 可能发生（需要重采样策略应对） |

**为什么粒子滤波在 Pacman 中更实用**：300 个粒子 ≈ 精确策略的 75% 计算量，但可以进一步减少到 100 个粒子以加速推理（仅 40,000 次操作 = 精确策略的 1/4）。且粒子数不随地图增大而增加——地图越大，优势越明显。

---

## 二、代码详解

### 2.1 Q1: constructBayesNet — 构建贝叶斯网络结构

**文件**：`inference.py`，第 31~84 行

```python
def constructBayesNet(gameState: hunters.GameState):
    # 定义变量名常量
    PAC = "Pacman"
    GHOST0 = "Ghost0"
    GHOST1 = "Ghost1"
    OBS0 = "Observation0"
    OBS1 = "Observation1"
    X_RANGE = gameState.getWalls().width      # 迷宫列数
    Y_RANGE = gameState.getWalls().height     # 迷宫行数
    MAX_NOISE = 7                              # 噪声幅值

    # --- 变量列表 ---
    variables = [PAC, GHOST0, GHOST1, OBS0, OBS1]

    # --- 边列表 ---
    edges = [
        (PAC, GHOST0),    # Pacman 位置影响幽灵 0 位置
        (PAC, GHOST1),    # Pacman 位置影响幽灵 1 位置
        (PAC, OBS0),      # Pacman 位置影响距离观测 0
        (PAC, OBS1),      # Pacman 位置影响距离观测 1
        (GHOST0, OBS0),   # 幽灵 0 位置影响距离观测 0
        (GHOST1, OBS1),   # 幽灵 1 位置影响距离观测 1
    ]

    # --- 变量值域 ---
    # Pacman 和幽灵可以在网格中任意位置
    all_positions = [(x, y) for x in range(X_RANGE) for y in range(Y_RANGE)]

    # 观测值: 噪声曼哈顿距离 ∈ [0, max_距离 + 噪声]
    max_dist = (X_RANGE - 1) + (Y_RANGE - 1)
    obs_values = list(range(max_dist + MAX_NOISE + 1))

    variableDomainsDict[PAC]    = all_positions
    variableDomainsDict[GHOST0] = all_positions
    variableDomainsDict[GHOST1] = all_positions
    variableDomainsDict[OBS0]   = obs_values
    variableDomainsDict[OBS1]   = obs_values

    return bn.constructEmptyBayesNet(variables, edges, variableDomainsDict)
```

**注释**：
- 幽灵之间无边 → 独立假设（实际并非独立，但这是简化）
- 观测变量的域要把噪声考虑进去：真实最大曼哈顿距离 + MAX_NOISE
- constructEmptyBayesNet 只建结构不填数字——CPT 由工作人员代码后续填入

---

### 2.2 Q2: joinFactors — 连接因子

**文件**：`factorOperations.py`，第 62~137 行

```python
def joinFactors(factors: List[Factor]):
    factors = list(factors)  # 可能以 dict_values 传入，先转为列表

    # 计算新因子的变量集合
    new_unconditioned = set()
    new_conditioned = set()
    domains = factors[0].variableDomainsDict()

    for f in factors:
        new_unconditioned |= f.unconditionedVariables()   # 无条件变量取并集
        new_conditioned |= f.conditionedVariables()       # 条件变量取并集

    # 关键规则：如果某变量在任一因子中是无条件的，
    # 它就不能在结果因子中是条件的
    new_conditioned -= new_unconditioned

    # 创建空白结果因子
    joined = Factor(list(new_unconditioned), list(new_conditioned), domains)

    # 遍历所有可能的变量赋值组合
    for assignment in joined.getAllPossibleAssignmentDicts():
        prob = 1.0
        for f in factors:
            prob *= f.getProbability(assignment)
            # getProbability 会自动忽略 assignment 中
            # 该因子不包含的变量
        joined.setProbability(assignment, prob)

    return joined
```

**注释**：
- 核心规则 `new_conditioned -= new_unconditioned`：消除了冗余的条件依赖。例如 `P(X|Y) ⋈ P(Y)` → 无条件 = {X, Y}，条件 = {Y} - {X, Y} = ∅ → `P(X, Y)`
- `getProbability` 可以接收超集赋值（assign 了多余变量），自动忽略不相关的变量
- 所有输入因子的 `variableDomainsDict` 相同（来自同一个贝叶斯网络）

---

### 2.3 Q3: eliminate — 消除变量

**文件**：`factorOperations.py`，第 135~195 行

```python
def eliminate(factor: Factor, eliminationVariable: str):
    # 新因子: 无条件变量去掉被消除的那个，条件变量不变
    new_unconditioned = list(factor.unconditionedVariables() - {eliminationVariable})
    new_conditioned = list(factor.conditionedVariables())
    domains = factor.variableDomainsDict()

    eliminated = Factor(new_unconditioned, new_conditioned, domains)

    # 遍历原始因子的所有赋值
    for assignment in factor.getAllPossibleAssignmentDicts():
        # 去掉被消除的变量 → 得到新因子的键
        reduced = {k: v for k, v in assignment.items() if k != eliminationVariable}
        # 累加概率（对消除变量的所有取值求和）
        eliminated.setProbability(
            reduced,
            eliminated.getProbability(reduced) + factor.getProbability(assignment)
        )

    return eliminated
```

**注释**：
- `getProbability(reduced)` 初始返回 0（Factor 表初始化全 0），后续累加即可
- 求和操作实现了 `Σ_{eliminationVariable} f(..., eliminationVariable, ...)`
- 例如 `eliminate(P(X, Y | Z), Y)` → 对 Y 的所有值求和 → `P(X | Z)`
- 必须保证消除的是无条件变量（调用前已 typecheck）

---

### 2.4 Q4: inferenceByVariableElimination — 变量消除推断

**文件**：`inference.py`，第 155~216 行

```python
def inferenceByVariableElimination(bayesNet, queryVariables, evidenceDict, eliminationOrder):
    # 1. 获取所有 CPT，带入证据（缩小表）
    factors = bayesNet.getAllCPTsWithEvidence(evidenceDict)

    # 2. 按指定顺序逐变量消除
    for var in eliminationOrder:
        # 合并包含此变量的所有因子
        not_joined, joined = joinFactorsByVariable(factors, var)

        # 合并后若还有多个无条件变量 → 消除并保留
        if len(joined.unconditionedVariables()) > 1:
            factors = not_joined
            factors.append(eliminate(joined, var))
        else:
            # 只剩一个无条件变量 → 求和为 1，丢弃
            factors = not_joined

    # 3. 合并剩余因子
    result = joinFactors(factors)

    # 4. 归一化为条件概率 P(query | evidence)
    return normalize(result)
```

**注释**：
- `joinFactorsByVariable` 是增强版：自动找出包含指定变量的因子，join 它们，返回 (未参与的因子, join 结果)
- "只有一个无条件变量"的特殊情况：消除后该变量消失，表变成"无条件变量为空，条件变量原有的"，其和为 1（全概率公理），所以丢弃
- `normalize` 将因子缩放使概率和为 1，变为合法的条件概率表

---

### 2.5 Q5a: DiscreteDistribution.normalize 和 sample

**文件**：`inference.py`，第 347~396 行

```python
def normalize(self):
    """将分布归一化，使所有键的值之和为 1。保持比例不变。"""
    total = self.total()
    if total == 0:
        return      # 全零分布不做操作
    for key in self:
        self[key] /= total

def sample(self):
    """从分布中加权随机采样一个键。
    概率与各键的值成比例。分布不需要预先归一化。"""
    r = random.random() * self.total()      # [0, total) 随机点
    cumulative = 0.0
    for key, weight in self.items():
        cumulative += weight
        if cumulative >= r:                 # 累加值超过随机点 → 命中
            return key
    return None  # 空分布的兜底
```

**注释**：
- `normalize` 是原地操作（修改 self，不返回新对象）
- `sample` 的加权采样：值越大的键，覆盖的区间越长，被选中的概率越高
- 不需要预先归一化——`self.total()` 包含了所有值的和
- 复杂度 O(K)，K=键的数量

---

### 2.6 Q5b: getObservationProb — 观测似然

**文件**：`inference.py`，第 465~477 行

```python
def getObservationProb(self, noisyDistance, pacmanPosition, ghostPosition, jailPosition):
    # 特殊情况 1: 幽灵在监狱 → 传感器必然返回 None
    if ghostPosition == jailPosition:
        return 1.0 if noisyDistance is None else 0.0

    # 特殊情况 2: 传感器读数为 None 但幽灵不在监狱 → 不可能
    if noisyDistance is None:
        return 0.0

    # 正常情况: 计算真实距离，查概率分布表
    trueDistance = manhattanDistance(pacmanPosition, ghostPosition)
    return busters.getObservationProbability(noisyDistance, trueDistance)
```

**注释**：
- 监狱位置 `(2*ghostIndex - 1, 1)` 是特殊状态——幽灵被捕获后在此
- 传感器模型：`P(noisyDist | trueDist)` 由 `busters.getObservationProbability` 提供
- 三个分支覆盖了所有 (ghostPos, observation) 的组合

---

### 2.7 Q6: ExactInference.observeUpdate — 精确推断更新步

**文件**：`inference.py`，第 580~602 行

```python
def observeUpdate(self, observation, gameState):
    pacPos = gameState.getPacmanPosition()
    jailPos = self.getJailPosition()

    # 贝叶斯更新: P(pos|obs) ∝ P(obs|pos) × P(pos)
    for pos in self.allPositions:
        likelihood = self.getObservationProb(observation, pacPos, pos, jailPos)
        self.beliefs[pos] *= likelihood

    # 归一化使总信念为 1
    self.beliefs.normalize()
```

**注释**：
- 这是贝叶斯公式的直接应用：后验 ∝ 似然 × 先验
- `self.beliefs` 是一个 `DiscreteDistribution`，键=位置，值=概率
- 遍历 `self.allPositions`（包含监狱位置）而不仅是 `self.legalPositions`
- `getObservationProb` 已处理了监狱和无观测的特殊情况

---

### 2.8 Q7: ExactInference.elapseTime — 精确推断预测步

**文件**：`inference.py`，第 604~629 行

```python
def elapseTime(self, gameState):
    new_beliefs = DiscreteDistribution()

    for oldPos, oldProb in self.beliefs.items():
        if oldProb == 0:
            continue    # 跳过零信念位置，节省计算

        # 获取从 oldPos 出发的转移分布 P(new | old)
        newPosDist = self.getPositionDistribution(gameState, oldPos)

        # 对每个可能的新位置，累加 oldProb × transProb
        for newPos, transProb in newPosDist.items():
            new_beliefs[newPos] += oldProb * transProb

    self.beliefs = new_beliefs
```

**注释**：
- 这是 HMM 前向算法的预测步：P(X_{t+1}) = Σ P(X_{t+1}|X_t)·P(X_t)
- `getPositionDistribution` 包含幽灵的移动规则：不能穿墙、可能被 Pacman 捕获
- 跳过零概率位置是重要优化——精确模型有 ~400 个位置，但多数信念集中在小范围

---

### 2.9 Q8: GreedyBustersAgent.chooseAction — 贪心追鬼策略

**文件**：`bustersAgents.py`，第 139~158 行

```python
def chooseAction(self, gameState):
    pacmanPosition = gameState.getPacmanPosition()
    legal = [a for a in gameState.getLegalPacmanActions()]

    # 获取每个存活幽灵的最可能位置
    livingGhostPositionDistributions = [
        beliefs for i, beliefs in enumerate(self.ghostBeliefs)
        if livingGhosts[i+1]
    ]
    ghostPositions = [dist.argMax() for dist in livingGhostPositionDistributions]

    # 选择使 Pacman 最接近最近幽灵的动作
    bestAction = None
    bestDist = float('inf')
    for action in legal:
        successorPos = Actions.getSuccessor(pacmanPosition, action)
        closestGhostDist = min(
            self.distancer.getDistance(successorPos, gp) for gp in ghostPositions
        )
        if closestGhostDist < bestDist:
            bestDist = closestGhostDist
            bestAction = action

    return bestAction
```

**注释**：
- `argMax()` 取信念分布中概率最大的位置——"最可能是鬼在哪？"
- `distancer.getDistance` 返回真实的迷宫距离（考虑墙壁），不是曼哈顿距离
- 贪心策略：假设每个鬼就在其最可能的位置，冲向最近的那个

---

### 2.10 Q9: ParticleFilter 初始化和信念分布

**文件**：`inference.py`，第 641~664 行

```python
def initializeUniformly(self, gameState):
    """均匀地（不是随机地）分布粒子到合法位置"""
    self.particles = []
    n_legal = len(self.legalPositions)
    for i in range(self.numParticles):
        self.particles.append(self.legalPositions[i % n_legal])

def getBeliefDistribution(self):
    """将粒子列表转换为归一化的分布"""
    dist = DiscreteDistribution()
    for p in self.particles:
        dist[p] += 1.0      # 统计每个位置的粒子数
    dist.normalize()         # 归一化
    return dist
```

**注释**：
- `i % n_legal` 实现了均匀分配：300 粒子 ÷ 200 合法位置 = 每个位置 1 或 2 个粒子
- "均匀"而非"随机"是刻意要求——随机初始化可能产生不均衡的先验
- `getBeliefDistribution` 将采样表示转回概率分布，用于可视化和决策

---

### 2.11 Q10: ParticleFilter.observeUpdate — 粒子滤波更新步

**文件**：`inference.py`，第 672~696 行

```python
def observeUpdate(self, observation, gameState):
    pacPos = gameState.getPacmanPosition()
    jailPos = self.getJailPosition()

    # 为每个粒子计算重要性权重
    # 关键: 用 (位置, 索引) 作为键，确保相同位置的粒子有独立的权重条目
    weights = DiscreteDistribution()
    for i, p in enumerate(self.particles):
        weights[(p, i)] = self.getObservationProb(observation, pacPos, p, jailPos)

    # 所有权重为 0 → 粒子全部"死亡"，重新初始化
    if weights.total() == 0:
        self.initializeUniformly(gameState)
        return

    # 从加权分布中有放回地抽样 numParticles 次
    # sample() 返回 (pos, index)，取 [0] 只保留位置
    self.particles = [weights.sample()[0] for _ in range(self.numParticles)]
```

**注释**：
- **最关键的设计**：`weights[(p, i)]` 而非 `weights[p]`。用 `(位置, 索引)` 做键保证了即使多个粒子在同一位置，每个粒子仍有独立的权重条目。这样做重采样时，粒子多的位置在权重分布中自然有更多条目，被抽中的概率更高
- 权重全零检查：当观测与所有粒子的预测都不兼容时（极端噪声），粒子滤波会"死亡"。重新初始化是标准应对策略
- 有放回抽样：一个位置可以被抽中多次，高权重位置获得更多粒子

---

### 2.12 Q11: ParticleFilter.elapseTime — 粒子滤波预测步

**文件**：`inference.py`，第 702~709 行

```python
def elapseTime(self, gameState):
    new_particles = []
    for oldPos in self.particles:
        # 获取从 oldPos 出发的转移分布
        newPosDist = self.getPositionDistribution(gameState, oldPos)
        # 从转移分布中随机采样一个后继位置
        new_particles.append(newPosDist.sample())
    self.particles = new_particles
```

**注释**：
- 精确推断的 `elapseTime` 是概率扩散（每个位置→所有可能后继的加权求和）
- 粒子滤波的 `elapseTime` 是采样（每个粒子→一个随机后继）
- `getPositionDistribution` 在两种情况下是同一个函数——模型相同，表示方法不同
- 采样引入随机性：粒子可能"走错了"（选了低概率的方向），但有足够多粒子时统计上是对的
- 复杂度 O(M·N) vs 精确推断的 O(N²)

---

## 三、关键设计决策

### 3.1 幽灵独立假设

两个幽灵的推理是独立的——每个幽灵有自己的 ExactInference 或 ParticleFilter 实例，有自己的信念分布。这在贝叶斯网络中也体现在 Ghost0 和 Ghost1 之间没有边。

**后果**：丢失了"两个幽灵都在左上角"这种联合概率信息。但大大简化了推断——否则联合状态空间是 O(N²) 而不是 O(2N)。

### 3.2 粒子滤波的粒子数选择

骨架默认 300 个粒子。实际：
- 精确推断有 ~400 个状态（网格大小），每步 O(400²)
- 300 个粒子：每步 O(300×400) ≈ 120,000 操作 → 比精确慢？不一定——精确推断要遍历每个信念位置的每个可能转移，常数更大
- 实际部署到 Pacman 推理管线时，可以降到 100~150 个粒子以加速

### 3.3 从 tracking 到项目感知模块

在你后续的 decision-report 实现中，tracking 模块的成果可以直接用到感知层：
- 粒子滤波输出的 `getBeliefDistribution()` 返回的概率热力图，天然是 ViT/GAT 感知模块的一个输入通道
- 或者直接用粒子滤波独立追踪幽灵位置，输出给 DT，不需要感知模块处理幽灵位置
