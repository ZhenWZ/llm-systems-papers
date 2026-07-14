# Agent Skill Optimization & Procedural Memory

这一类关注 LLM agent 的 skill artifact、procedural memory、agent memory、trajectory feedback、validation-gated updates 和 execution harness integration。重点是如何把 agent 的领域操作策略、memory scaffold、execution harness 和必要的 memory-specialist 参数状态优化成可审计、可复用、可部署的系统能力。

## 当前论文

| 优先级 | 论文 | 阶段 | 核心问题 | 系统启发 | 笔记 |
| --- | --- | --- | --- | --- | --- |
| P0 | SkillOpt: Executive Strategy for Self-Evolving Agent Skills | agent adaptation | 手写或 one-shot skill 难以稳定自我改进，unbounded self-revision 又容易回退 | 把 skill document 当作 trainable external state，用 rollout evidence、bounded edits、held-out gate 和 rejected buffer 训练 `best_skill.md` | [notes](../papers/2026-skillopt-agent-skills.md) |
| P0 | SkillComposer: Learning to Evolve Agent Skills for Specification and Generalization | agent skill evolution | 一次性 skill extraction 在 specification 和 generalization 之间失衡：太具体无法迁移，太抽象缺少操作指导 | 把 skill lifecycle 拆成 create、merge、improve，并用 delta pass rate rejection sampling 训练 composer，支持 offline/online/hybrid 三种推理时演化模式 | [notes](../papers/2026-skillcomposer-skill-evolution.md) |
| P0 | AutoMem: Automated Learning of Memory as a Cognitive Skill | agent memory | 长程 agent 任务中，固定 RAG / context management 难以决定 what to encode、when to retrieve、how to organize，且 memory mistake 的影响常延迟数百到数千步才暴露 | 把 file-system memory operations 放进 action space，用 meta-LLM review 完整 trajectory 来优化 scaffold，再从 agent 自身轨迹中训练 memory specialist | [notes](../papers/2026-automem-memory-cognitive-skill.md) |
| P1 | Self-Harness: Harnesses That Improve Themselves | agent adaptation | 人工 harness engineering 难以跟上不同模型的 tool-use habit、错误模式和 prompt sensitivity | 把 harness edit 变成 evidence-driven、regression-gated state transition，用 failure clustering、minimal proposal 和 held-out gate 改进 Codex/Claude Code 类 execution harness | [notes](../papers/2026-self-harness-agent-harnesses.md) |

## 阅读重点

- Skill document 和 model weights、prompt template、agent memory 的职责边界。
- Rollout evidence、tool traces、verifier feedback 如何变成可验证的 skill edits。
- Skill create / merge / improve 如何分别负责知识捕获、generalization 和 specification。
- Memory files、LOG/PLAN routines 和 memory operations 如何变成可观察、可训练、可审计的 agent action。
- Textual learning rate、validation gate 和 rejected-edit buffer 如何防止 skill drift。
- Codex / Claude Code 等 execution harness 中，skill artifact 如何跨环境迁移。
- Harness-level editable surfaces、failure clustering 和 regression gate 如何把 agent 自我改进变成可审计的系统变更。
- Memory scaffold optimization 与 memory-specialist finetuning 的边界：哪些能力应留在 prompt/code/file schema，哪些值得进入 LoRA / model weights。

## 后续可补充方向

- Trace2Skill、EvoSkill、SkillForge、SkillFoundry、ProcMEM、AutoRefine。
- Agent skill benchmark 与 procedural memory benchmark。
- Skill library、skill routing、multi-skill conflict resolution。
- Skill lifecycle operations：create、merge、improve、retire、reject、rollback、versioning。
- Self-Harness、Meta-Harness、Agentic Harness Engineering、HarnessFix 等 harness-level 自动优化方向。
- Agent memory benchmarks：BALROG、Crafter、MiniHack、NetHack，以及真实软件工程长程任务中的 memory regression tests。
