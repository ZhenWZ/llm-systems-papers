# Agent Skill Optimization & Procedural Memory

这一类关注 LLM agent 的 skill artifact、procedural memory、trajectory feedback、validation-gated updates 和 execution harness integration。重点是如何在不更新 model weights 的情况下，把 agent 的领域操作策略训练成可审计、可复用、可部署的文本状态。

## 当前论文

| 优先级 | 论文 | 阶段 | 核心问题 | 系统启发 | 笔记 |
| --- | --- | --- | --- | --- | --- |
| P0 | SkillOpt: Executive Strategy for Self-Evolving Agent Skills | agent adaptation | 手写或 one-shot skill 难以稳定自我改进，unbounded self-revision 又容易回退 | 把 skill document 当作 trainable external state，用 rollout evidence、bounded edits、held-out gate 和 rejected buffer 训练 `best_skill.md` | [notes](../papers/2026-skillopt-agent-skills.md) |
| P1 | Self-Harness: Harnesses That Improve Themselves | agent adaptation | 人工 harness engineering 难以跟上不同模型的 tool-use habit、错误模式和 prompt sensitivity | 把 harness edit 变成 evidence-driven、regression-gated state transition，用 failure clustering、minimal proposal 和 held-out gate 改进 Codex/Claude Code 类 execution harness | [notes](../papers/2026-self-harness-agent-harnesses.md) |

## 阅读重点

- Skill document 和 model weights、prompt template、agent memory 的职责边界。
- Rollout evidence、tool traces、verifier feedback 如何变成可验证的 skill edits。
- Textual learning rate、validation gate 和 rejected-edit buffer 如何防止 skill drift。
- Codex / Claude Code 等 execution harness 中，skill artifact 如何跨环境迁移。
- Harness-level editable surfaces、failure clustering 和 regression gate 如何把 agent 自我改进变成可审计的系统变更。

## 后续可补充方向

- Trace2Skill、EvoSkill、SkillForge、SkillFoundry、ProcMEM、AutoRefine。
- Agent skill benchmark 与 procedural memory benchmark。
- Skill library、skill routing、multi-skill conflict resolution。
- Self-Harness、Meta-Harness、Agentic Harness Engineering、HarnessFix 等 harness-level 自动优化方向。
