# World Models & Test-Time Adaptation

这一类关注 learned world model / latent dynamics model 在部署阶段如何从「训练后冻结」走向「闭环内持续自适应」。重点不是单个感知或生成模型，而是把 test-time adaptation、self-supervised online update、planning 闭环、延迟预算和稳定性护栏组织成可维护的系统边界——这与 LLM serving 侧的 online 自校准、局部参数更新、部署后自改进有直接类比。

## 当前论文

| 优先级 | 论文 | 阶段 | 核心问题 | 系统启发 | 笔记 |
| --- | --- | --- | --- | --- | --- |
| P1 | AdaJEPA: An Adaptive Latent World Model | test-time adaptation | latent world model 部署后冻结，distribution shift 下预测失准导致 MPC planning 失败 | 用 MPC 闭环里观测到的 next-state transition 作 self-supervised 信号，单步、局部（encoder 末段 + predictor 最后一层）更新，几乎零延迟地持续校准 world model | [notes](../papers/2026-adajepa-adaptive-latent-world-model.md) |

## 阅读重点

- Test-time adaptation 信号从哪里来：AdaJEPA 复用 pretraining 的 prediction loss，用真实 transition 作监督，而非 entropy/reconstruction 代理目标。
- 「更新哪些参数、更新几步」如何决定延迟与稳定性：单步 + 局部参数 + 小 replay buffer 的护栏设计。
- Planning 闭环（MPC / CEM / GD）与在线权重更新如何耦合，以及闭环延迟预算的硬约束。
- 与 text-space 部署后自改进（SkillOpt、Self-Harness）在更新对象、频率、可回滚性上的取向差异。

## 后续可补充方向

- DINO-WM、TD-MPC、Dreamer 等 latent world model / model-based control 的系统化对比。
- Test-time training（TTT/TTT-MAE）、Tent/EATA/CoTTA 等 test-time adaptation 方法在闭环控制中的迁移。
- Online / continual adaptation 在 LLM serving 中的对应形态：adapter-only 在线微调、feedback-driven 自校准与回滚机制。
