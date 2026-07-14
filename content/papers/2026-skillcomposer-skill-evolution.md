# SkillComposer: Learning to Evolve Agent Skills for Specification and Generalization

## Metadata

- Year: 2026
- Authors: Qi Zhang, Zhaopeng Feng, Xiaonan Shi, Xiaomeng Hu, Chu Liu, Pengjun Xie, Xiaobin Wang, Jieping Ye, Bryan Hooi, Haobo Wang, Junbo Zhao
- Category: Agent Skill Optimization & Procedural Memory
- Priority: P0
- Links: [arXiv](https://arxiv.org/abs/2606.06079), [HF Papers](https://huggingface.co/papers/2606.06079)
- Keywords: agent skills, skill evolution, procedural memory, specification, generalization, rejection sampling, delta pass rate, LiveCodeBench, AppWorld, tau2-Bench

## 一句话结论

SkillComposer 把 agent skill construction 拆成三个可学习操作：`create` 从 trajectory 中抽取技能，`merge` 把相似技能合并成更泛化的 skill library，`improve` 基于新执行经验做任务特化；再用 delta pass rate 过滤的 rejection sampling 训练 4B composer，使 agent 能在 inference time 以 offline / online / hybrid 三种模式自我演化 skill。

## Related Works

- Agent skills / skill libraries: Anthropic Skills、agentskills、SkillX、SkillRL、SkillNet、SkillClaw 等把可复用 procedural knowledge 写成可检索 skill。SkillComposer 的区别是显式建模 skill lifecycle，而不是一次性从成功轨迹中抽取。
- Procedural memory: MemP 等方法从 valid trajectories 中提取 procedural memory。论文结果显示 uncontrolled skill extraction 可能在 code tasks 上显著降级，说明 memory/skill 写入需要质量门控和可演化机制。
- Skill optimization: SkillOpt 把 skill document 当成 trainable external state；SkillComposer 更偏训练一个 composer model，让模型学会生成、合并、细化技能，并在 inference time 执行 skill evolution。
- Harness / scaffold optimization: Self-Harness、AFlow、ADAS、PromptBreeder 等把 agent harness 或 prompt 作为可搜索对象。SkillComposer 的优化对象更窄：可复用自然语言 skill 的 specification/generalization trade-off。
- Agent and code benchmarks: tau2-Bench、LiveCodeBench v6、AppWorld 分别覆盖 multi-turn tool-use agent、code generation、interactive app usage，用来验证 skill evolution 是否跨 domain / scale / task type 迁移。

## 问题与背景

Agent skills 通常是一个 markdown-style 文档，包含 name、description/trigger condition 和 procedural body。把相关 skill 加入 context 可以在不改 model weights 的情况下复用经验，因此适合 Codex / Claude Code 这类 execution agent。

现有方法的问题是 skill construction 常被当成 one-shot extraction：从一次成功 trajectory 中抽一段说明就加入 skill library。这带来两个相反风险：

- 过度特化：skill 只适合原任务，不能迁移到相似任务。
- 过度抽象：skill 太泛，无法给具体任务足够操作指导。

论文把这个矛盾命名为 specification 与 generalization 的张力。真正可用的 skill system 需要在 library 构建、检索、任务执行和在线修订之间持续调节，而不是只保存一次经验总结。

## 方法与系统设计

SkillComposer 的系统边界包括 skill executor、skill library、composer model 和 rejection-sampling training loop。

### Skill 表示

论文把每个 skill 表示为三元组：`name`、`description`、`body`。`description` 是触发条件，决定什么时候应加载这个 skill；`body` 是 procedural strategy，指导 executor 的 reasoning/action。

### 三个 Skill Operations

SkillComposer 定义三个可组合操作：

- `Skill Create`: 当 agent 在没有 skill 的情况下完成任务后，从 raw trajectory 中抽取 reusable procedural knowledge，形成新 skill。
- `Skill Merge`: 当 skill library 变大、存在重叠或过细 skill 时，计算 skill pair 的 multi-view similarity，把相似 skill 合并成更一般、更可迁移的 skill。
- `Skill Improve`: 当 agent 在某个 skill 指导下执行新任务后，把新 trajectory 中暴露的 pattern 写回 skill，使其更贴近目标任务。

这三个操作对应 skill 生命周期：create 负责知识捕获，merge 负责 generalization，improve 负责 specification。

### 三种 Inference-Time 模式

SkillComposer 支持三种部署方式：

- Offline: 先在任务集上生成 raw trajectories，通过 create 建库，再迭代 merge，得到 generalized skill library；推理时检索 top-k skills，并让 executor self-select 是否使用。
- Online: 对每个新任务从空 skill 开始，先 create，再多轮执行和 improve，适合没有预建任务集或 streaming/open-ended 任务。
- Hybrid: 先用 offline library 提供一般先验，再对目标任务做 online improve，兼顾覆盖面和任务特化。

### Rejection Sampling 训练

论文指出任意 LLM 都能 zero-shot 生成 skill，但质量不稳定。因此作者用 delta pass rate 作为统一质量信号做 rejection sampling。候选 skill 只有在相对 baseline 提升 executor pass@ 超过阈值时才进入 supervised fine-tuning 数据。

训练覆盖 create、merge、improve 三种操作。Create 比较有/无候选 skill 的 pass@；Merge 比较合并后 skill 与两个原 skill 在各自 source tasks 上的表现；Improve 同时看 source task 和 target task，使训练信号覆盖 specification 与 generalization。论文合成约 7,000 条 SFT 数据，用 LlamaFactory 微调 Qwen3.5-4B，并用 vLLM 运行 composer/executor。

## 创新点

- 把 skill construction 从 one-shot extraction 拆成 create / merge / improve 三个可学习操作。
- 明确区分 skill quality 的两个维度：merge 驱动 generalization，improve 驱动 specification。
- 用 delta pass rate 作为 skill 质量门控，避免把表面合理但实际降低 executor performance 的 skill 写入训练集或 library。
- 同一 composer 支持 offline library building、online task-specific refinement 和 hybrid specialization。
- 证明 skill composition 是 transferable meta-ability：agent-only training 也能改善 code tasks，4B composer 生成的 skill 也能提升 27B executor。

## 实验与结论

实验覆盖 tau2-Bench、LiveCodeBench v6 和 AppWorld。

- 训练与评估设置：code 数据来自 OpenCodeReasoning，agent rejection sampling 使用类似 tau2-Bench Retail domain 的 AutoForge 合成数据；总共约 7,000 条 SFT 数据。Composer 训练为 Qwen3.5-4B，executor 使用 Qwen3.5-4B 和 Qwen3.5-27B。
- 主结果：在 4B executor 上，SkillComposer hybrid 在 tau2-Bench overall 达到 85.7，比 No Skill 79.5 高 6.2；SkillComposer online 在 LiveCodeBench v6 达到 59.1，比 No Skill 56.6 高 2.5。
- 27B executor 迁移：SkillComposer online 在 tau2-Bench 达到 88.0，比 No Skill 83.5 高 4.5；LiveCodeBench overall 达到 83.1，比 79.7 高 3.4。说明小 composer 生成的 skill 能服务更大 executor。
- Baseline 风险：MemP 在 4B executor 的 LiveCodeBench 上从 No Skill 56.6 降到 47.4，说明 uncontrolled procedural memory extraction 会伤害泛化。
- Cross-domain：训练数据只来自 Retail，但在 Airline 和 Telecom 上仍有提升；4B hybrid 在 Airline 达 82.5，比 No Skill 68.8 高 13.7。
- Ablation：offline library 中 `Create/Merge` 最强，tau2-Bench 77.5、LiveCodeBench 59.0；Merge 能合并冗余 skill、提升 generalization。Improve 在 offline 中可能降低 transferability，更适合 online/hybrid。
- Cross-task transfer：agent-only training 在 LiveCodeBench v6 上把 online overall 从 81.4 提到 82.9；code-only 为 84.0；mixed 为 83.1，说明 skill composition 有跨任务类型迁移能力。
- AppWorld：SkillComposer-4B online accuracy 57.7、pass rate 88.3，优于 vanilla 55.2/88.1；未训练 composer 虽提升 accuracy 到 57.1，却把 pass rate 降到 86.0，进一步说明训练和门控的重要性。
- Inference budget：在相同轮数预算下，online skill evolution 优于 repeated sampling；4B executor 到 4 轮时 73.71 vs. 69.71，27B executor 为 91.43 vs. 88.57。

## 局限性与风险

- 训练数据收集很贵：delta pass rate 需要为每个候选 skill 多次执行任务，计算成本高。
- 实验模型族有限，主要是 Qwen3.5-4B/27B；不同 base model、不同 prompt/tool runtime 下是否同样有效还需验证。
- 公开页面没有列出官方代码或模型权重；复现需要自行重建 rejection sampling、skill library、similarity search 和 multi-round evaluation。
- Skill 质量依赖 pass@ 作为 reward，容易偏向 benchmark-visible gains；真实 agent 还需要安全性、稳定性、可解释性和人类可审计性。
- Online / hybrid 模式增加 inference-time compute 和 latency；是否值得取决于任务价值、轮数预算和 skill 是否能长期复用。
- Merge 需要 similarity threshold 和 embedding model，阈值过低会合并不相容技能，阈值过高会导致 library 膨胀。
- Improve 在 offline 构建中可能牺牲 transferability，说明自动 skill refinement 需要部署模式感知的策略。

## 对 LLM Systems 的启发

- Skill library 不能只做 append-only memory。需要 create、merge、improve、reject 和 retire 等 lifecycle operations。
- 对 Codex / Claude Code 类 agent，`SKILL.md` 不只是静态 prompt，而是可以用 trajectory evidence 训练和演化的 external state。
- Generalization 与 specification 要分层处理：全局库更需要 merge 和去冗余，单任务会话更需要 improve 和局部特化。
- Skill 写入应该有 measurable gate。没有 pass@ / verifier / regression test 的 skill extraction 可能像 MemP 一样在新域中伤害表现。
- Small composer + large executor 是实用系统形态：用较小模型维护 skill library，用较大模型执行任务，可以降低长期维护成本。
- 需要把 skill evolution 记录为可审计 artifact：skill diff、source trajectory、评估任务、delta pass rate、merge pair、similarity score、accepted/rejected reason。

## 复现/阅读建议

- 先读 Introduction 和 Section 3，明确 create / merge / improve 的职责边界，再读 Table 1 看三种 deployment mode 的效果。
- 复现不宜一开始做完整 SFT。建议先做 zero-shot composer + small task set，验证 create/merge/improve pipeline 和 skill diff 格式。
- 第二步复现 rejection sampling：对每个候选 skill 固定 executor、seed、pass@ runs 和阈值，记录 accepted/rejected samples。
- 对工程 agent，优先用 repo-local tasks 建一个小型 skill library，测量是否降低重复错误，而不是直接追求论文 benchmark 数字。
- 需要额外报告系统指标：skill library size、merge ratio、retrieval precision、online evolution rounds、token cost、latency、accepted skill rate、regression failures。
- 与 SkillOpt、AutoMem、Self-Harness 一起阅读：SkillComposer 强调 skill lifecycle operations，SkillOpt 强调 bounded text-state optimization，AutoMem 强调 memory operations as actions，Self-Harness 强调 execution harness edits。
