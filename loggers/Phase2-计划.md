# Phase 2 计划：PPO 微调管线

> 原则：延用骨架的游戏引擎、交互方式和特征提取，只替换决策模型为 DT。

---

## 已完成（Phase 1）

```
✅ transformer_block.py    GPT-2 风格的 Pre-LN block
✅ decision_transformer.py DT 主模型（BC 预训练可用）
✅ train_bc.py             BC 训练入口
✅ dataset.py              轨迹数据集加载
```

---

## Phase 2A：RL 基础组件（1 天）

| 序号 | 文件 | 内容 | 依赖 |
|------|------|------|------|
| 1 | `rl/gae.py` | `compute_gae(rewards, values, dones, gamma, lam)` → `(advantages, returns)` | 无（纯函数） |
| 2 | `rl/ppo_adapter.py` | 包装 DT：`act()` 采样动作+log_prob+value；`evaluate()` 算 PPO clip loss + value loss + entropy | GAE |

验证方式：`compute_gae` 用已知数组手算对照；`PPOAdapter` 用随机 batch 跑通 forward+backward。

---

## Phase 2B：训练循环 + 环境对接（1 天）

| 序号 | 文件 | 内容 | 依赖 |
|------|------|------|------|
| 3 | `rl/trainer.py` | `PPOTrainer`：`collect_rollout()` 与环境交互收集 2048 步；`train_epoch()` 多轮 PPO 更新 | PPOAdapter |
| 4 | `scripts/train_ppo.py` | 加载 BC checkpoint → 调用骨架的 `GameState` + `extract_features` 跑环境 → PPO 在线微调 → 保存 | trainer |

环境对接方式：

```python
# 直接用骨架，不写新 wrapper
from PPCA_AIPacMan_2024_main.reinforcement.pacman import GameState
# 特征提取沿用 collection 脚本的 extract_features
from scripts.collect_expert_data import extract_features
# DT 加载 BC 权重，PPOTrainer 接管训练循环
```

---

## 验证里程碑

| 节点 | 验证方式 |
|------|---------|
| GAE 写完 | 手工数组对照：`A_t = δ_t + γλ·A_{t+1}` 递推结果正确 |
| PPOAdapter 写完 | 随机 batch 前向+反向不报错 |
| Trainer 写完 | `smallGrid` 上跑 100 局，score 有上升趋势 |
| PPO 脚本跑通 | 加载 BC checkpoint，微调后在 `smallGrid` 上胜率 > BC |

---

## 暂存 / 删除

| 文件 | 处置 |
|------|------|
| `perception/vit_encoder.py` | 删除 — Pacman 用结构化状态，不需要 ViT |
| `perception/gat_encoder.py` | 删除 — 同上 |
| `perception/fusion.py` | 删除 — 同上 |
| `cuda/flash_attention.py` | 删除 — 性能优化，功能跑通后再看 |
| `cuda/fused_mlp.py` | 删除 — 同上 |
| `perception/particle_filter.py` | 保留 — Phase 3 可选增强 |
| `env/gym_wrapper.py` | 保留 — 如果后续想标准化接口可用 |
| `model/world_model.py` | 保留 — Phase 3 MCTS 需要 |
| `planning/mcts.py` | 保留 — Phase 3 |
| `agent/pacman_agent.py` | 保留 — Phase 3 组装 |

---

## 从哪开始

`rl/gae.py` — 零依赖，写完就能验证。
