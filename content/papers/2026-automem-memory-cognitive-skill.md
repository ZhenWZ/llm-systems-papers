# AutoMem: Automated Learning of Memory as a Cognitive Skill

## Metadata

- Year: 2026
- Authors: Shengguang Wu, Hao Zhu, Yuhui Zhang, Xiaohan Wang, Serena Yeung-Levy
- Category: Agent Skill Optimization & Procedural Memory
- Priority: P0
- Links: [HF Papers](https://huggingface.co/papers/2607.01224), [arXiv](https://arxiv.org/abs/2607.01224), [project](https://autolearnmem.github.io/), [code](https://github.com/autoLearnMem/AutoMem)
- Keywords: agent memory, procedural memory, metamemory, file-system memory, scaffold optimization, memory specialist, LoRA, BALROG, Crafter, MiniHack, NetHack

## 一句话结论

AutoMem 把 LLM agent 的 memory management 从固定 RAG / prompt engineering 组件提升为可学习的 cognitive skill：agent 把 file-system operations 当作一等 action 来记录、检索和组织外部记忆，再由两个 meta-LLM outer loops 分别优化 memory scaffold 和 memory specialist，使 Qwen2.5-32B-Instruct 在长程游戏任务上获得约 2x-4x progression gain。

## Related Works

- External memory for language models: RAG、MemGPT、Generative Agents、A-MEM、MemoryBank 等把外部 memory 作为 retrieval store、paging system、memory stream 或 long-term memory。AutoMem 的差异是让模型自己通过 file operations 管理 memory，而不是把 memory 作为固定模块。
- Trainable memory behavior: MemLLM、Self-Notes、Memory-R1、MemSearcher 等训练模型使用 read/write memory 或在推理中管理记忆。AutoMem 进一步把 memory 行为拆成 scaffold structure 和 model proficiency 两个优化轴。
- Memory architecture optimization: MemEvolve、EvolveMem、MetaMem 等优化 memory architecture、retrieval configuration 或 meta-memory strategy。AutoMem 的 scaffold loop 直接改写 agent code、prompt、file schema 和 action vocabulary，并且更新信号来自完整长程 trajectory。
- Automated agent optimization: ADAS、AFlow、DSPy、PromptBreeder、APE、SkillOpt、Self-Harness 等都将 agent / prompt / skill / harness 作为可优化对象。AutoMem 与它们相近，但优化对象更聚焦于 memory decisions，并加入 memory-specialist finetuning。
- Long-horizon game agents: Voyager、Reflexion、ExpeL、NetPlay、BALROG 等使用游戏环境评估 planning、skill acquisition 和 long-horizon reasoning。AutoMem 选择 Crafter、MiniHack、NetHack，是为了让 memory mistakes 在随机、长程、部分可观察环境中显性化。

## 问题与背景

LLM 的 context window 类似有限 working memory。长程 agent 任务会远超这个缓冲区：地图、库存、任务目标、失败路径、环境变化和历史行动都需要被压缩到外部 memory。常见做法是加 RAG、summary buffer、vector store 或 scratchpad，但这些方法往往把 memory 作为固定系统组件，不能让模型学习 what to encode、when to retrieve、how to organize。

AutoMem 关注的核心问题是 metamemory：模型是否能学会管理自己的记忆。长程任务中，memory mistake 往往不是立刻失败，而是在数百或数千步后暴露，例如未记录坐标导致迷路、重复写入导致检索噪声、错误 summarization 导致目标丢失。人工 review 完整 trajectory 成本过高，因此需要自动化的 trajectory-level memory optimization。

## 方法与系统设计

AutoMem 由一个 inner-loop agent 和两个 outer-loop optimizer 组成。inner loop 负责执行任务；outer loops 负责从完整轨迹中优化 memory skill。

### Inner-Loop Agent: Memory as File System

AutoMem 给 agent 一个真实 file-system memory directory。每一步 agent 运行两个 routine：

- `LOG`: 判断刚发生的事情是否值得记录，以及应该 append、create、rewrite 或 upsert 到哪个 memory file。
- `PLAN`: 在决定下一步 world action 前，先 search/read 相关 memory files，检索地图、库存、历史事件或策略。

关键设计是把 memory operations 放进和 task actions 同一个 action space。`<|SEARCH|>`、`<|APPEND|>`、`<|UPSERT_MAP|>` 等操作不再是隐藏的 retrieval middleware，而是 trajectory 中可观测、可审计、可优化的 agent actions。

### Outer-Loop 1: Scaffold Optimization

第一个 outer loop 优化 memory structure。Meta-LLM 读取完整 episode traces、memory directory 和 agent scaffold code，诊断 memory usage failure，然后改写 prompt、file schema、validation logic、action vocabulary 或 scaffold code。

每次 revision 都要在 fixed seed set 上重新评估；只有平均 progression 改善才保留。论文报告的收敛版本大致是 Crafter v5、MiniHack v4、NetHack v2。这个 loop 不更新 base model weights，只改 memory scaffold，因此它像 evidence-gated code review + agent scaffold search。

### Outer-Loop 2: Memory-Proficiency Training

第二个 outer loop 优化 model proficiency。Meta-LLM training engine 读取 agent 自身的大量 episode traces，选择值得强化的 LOG / PLAN memory decisions，构造成 supervised data，并同时选择 LoRA 训练配置。

部署时使用 two-model split：LoRA-finetuned memory specialist 处理 LOG 和 PLAN 中的 memory-consultation 部分；未修改的 base gameplay model 负责最终 world action。这样训练信号集中在 memory behavior 上，同时避免破坏 base model 的 task-action competence。

### Evaluation Stack

实验使用 Qwen2.5-32B-Instruct 作为 inner-loop base model，通过 vLLM serving；outer loops 使用强 meta-LLM 审查完整轨迹和驱动训练配置。任务来自 BALROG：Crafter、MiniHack 和 NetHack。官方 repo 提供 scaffold、loop1 scaffold evolution、loop2 training engine、LLaMA-Factory LoRA 训练路径，以及用 Claude Code CLI 驱动 meta-loop 的运行说明。

## 创新点

- 将 memory management 定义为独立可学习 skill，而不是固定 retrieval architecture。
- 用 file-system operations 作为一等 agent actions，使 memory decisions 可追踪、可回放、可审计。
- 把 memory learning 拆成两个互补轴：scaffold structure 通过代码/prompt/file schema 优化，model proficiency 通过 memory specialist LoRA 训练优化。
- 使用完整 trajectory review 作为优化信号，解决 memory mistake 延迟暴露、final reward 难以归因的问题。
- Two-model deployment 保留 base task competence，只把参数更新集中到 memory specialist。
- 公开 project page 和 code，使复现不只停留在论文描述。

## 实验与结论

AutoMem 在三个 procedurally generated long-horizon games 上评估：Crafter、MiniHack、NetHack。指标是 BALROG progression rate。

- Qwen2.5-32B-Instruct 的 basic context baselines 较弱：sliding window 为 Crafter 19.55、MiniHack 2.50、NetHack 0.00；加 chain-of-thought 后为 17.27、10.00、0.00。
- `memory-as-file-system, v0` 已有一定收益：25.00、7.50、0.42。
- Scaffold optimization 后提升到 47.27、27.50、1.57，约为 v0 的 1.89x、3.67x、3.74x。
- 再加入 memory training 后达到 51.36、30.00、1.85，与 Claude Opus 4.5 的 49.5、27.5、2.0 接近，也接近 Gemini 3.1 Pro Thinking 的 55.0、27.5、2.6。
- 相比 Qwen2.5-72B-Instruct BALROG leaderboard 的 27.3、5.0、0.3，AutoMem 显示 well-structured memory management 在这些任务上比单纯扩大同系列模型规模更高杠杆。
- 行为指标也改善：unproductive action rate 下降 32%-65%；redundant memory writes 下降 68%-83%；empty-search rate 下降 13%-50%；per-step input context shrink 3%-30%。
- NetHack case study 中，scaffold 将 append-only dungeon map 改成 coordinate-keyed `<|UPSERT_MAP|>`，并加入 inventory/status auto-sync 与 strategy reference，使 per-step memory growth 从 138 characters 降到 6 characters，减少约 95%。
- Memory specialist 训练后体现出 consult-before-write discipline：LOG-phase memory writes per SEARCH 从 Crafter 0.84 降到 0.39、MiniHack 2.89 降到 0.82、NetHack 4.66 降到 1.31。

## 局限性与风险

- 评估主要在游戏环境，虽然适合暴露 memory skill，但不能直接证明在软件工程、研究助理、企业知识库等真实任务上同样有效。
- 当前 memory 是 episodic：每个 episode 的 file system 重新开始。跨 episode / 跨项目 persistent memory 如何清理、版本化、权限隔离和遗忘仍未解决。
- 每个环境分别优化 scaffold 和 memory specialist；一个通用 scaffold / specialist 能否跨环境迁移仍待验证。
- Outer loops 依赖强 meta-LLM 读取完整轨迹，成本、延迟、隐私和可重复性需要工程治理。
- Loop 2 需要 LoRA training、vLLM serving、LLaMA-Factory、BALROG/NLE 等较重依赖；复现门槛高于纯 prompt/scaffold 方法。
- File-system memory 提高了可审计性，但也引入 prompt injection、敏感信息泄露、stale memory、schema drift 和 memory poisoning 风险。
- 论文结果基于 fixed-seed evaluation 和 progression metric；真实部署还需要 memory regression tests、长时间运行稳定性、成本统计和失败回滚机制。

## 对 LLM Systems 的启发

- Agent memory 不应只是隐藏在 retrieval middleware 后面的黑盒。把 memory operations 暴露为 action，可以让系统记录、审计、回放和优化每一次写入/检索。
- Memory scaffold 是一种重要的 system artifact：file schema、工具 vocabulary、LOG/PLAN routine、validation rules 与 prompt 一样影响 agent 的长期行为。
- 对 Codex / Claude Code 类工程 agent，`AGENTS.md`、skills、workspace notes、task traces 和 generated memories 都可以视为 scaffold + procedural memory，而不是静态说明文档。
- Memory optimization 需要 trajectory-level evidence。只看 final success/failure 不足以定位 memory bug；需要保存 tool traces、memory diffs、search misses、redundant writes 和 stale reads。
- 参数训练和外部状态优化可以互补：优先用 scaffold / skill document / file schema 提升上限，再用 LoRA 或 specialist model 内化稳定的 memory discipline。
- 生产系统应监控 memory 的质量指标：write/search ratio、empty-search rate、duplicate write rate、context token shrink、stale memory hit rate、memory permission violations。

## 复现/阅读建议

- 先读 arXiv Section 2 和 project page 的 framework 图，明确 inner-loop LOG/PLAN 与两个 outer-loop 的职责边界。
- 从官方 repo 的 `scaffolds/inner_agent_v0` 开始，先复现单环境 evaluation，而不是一开始跑完整 AutoMem。
- 优先复现 Loop 1 scaffold optimization，因为它不需要训练模型，能验证 memory schema 和 prompt/code revision 的增益。
- Loop 2 复现需要准备 vLLM、Qwen2.5-32B-Instruct、LLaMA-Factory、LoRA training environment 和 meta-LLM CLI；报告时必须记录 seeds、episode count、meta-LLM、token cost、训练配置和硬件。
- 复现报告不应只给 progression rate，还应输出 memory action statistics：write count、search count、empty-search rate、duplicate writes、memory directory size、per-step context tokens。
- 与 SkillOpt、Self-Harness 一起阅读：三者都把 agent 外部行为面当作可优化对象，但 AutoMem 更强调 memory operations 的 observable action space 和 dedicated memory specialist。
