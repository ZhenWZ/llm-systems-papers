# Agent Runtime & Skill Optimization

这一类关注 LLM agent runtime 中的 skill / procedural memory、tool-use policy、validation-gated optimization、execution harness integration 和跨模型/跨 harness 迁移。重点不是改 model weights，而是如何把 agent 的可执行流程、验证纪律和领域经验沉淀成可审计、可部署的系统状态。

## 当前论文

| 优先级 | 论文 | 阶段 | 核心问题 | 系统启发 | 笔记 |
| --- | --- | --- | --- | --- | --- |
| P1 | SkillOpt: Executive Strategy for Self-Evolving Agent Skills | agent adaptation | 人工或一次性生成的 agent skill 缺少稳定的训练式优化机制，容易漂移、过拟合或难以复用 | 把 `best_skill.md` 当作 external trainable state，用 rollout evidence、bounded edits 和 held-out validation gate 训练 compact procedural skill | [notes](../papers/2026-skillopt-agent-skills.md) |

## 阅读重点

- Skill 文档在 agent stack 中扮演的是 prompt、procedural memory、tool policy 还是 configuration。
- Validation gate、rejected-edit buffer 和 slow/meta update 如何把 self-reflection 变成可控优化。
- Optimized skill 是否能跨 target model、Codex/Claude Code harness 和相邻 benchmark 迁移。
- Skill artifact 的部署边界：训练期 optimizer model 与推理期 target agent 应严格分离。

## 后续可补充方向

- Trace2Skill、AutoSkill、SkillForge、SkillClaw、SkillX 等 trajectory-to-skill / skill library 工作。
- GEPA、TextGrad、DSPy 等 prompt/program optimization 系统。
- Codex、Claude Code、SWE-agent 等工具型 agent harness 的 skill / memory 机制。
