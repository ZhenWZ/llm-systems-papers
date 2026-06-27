# Self-Harness: Harnesses That Improve Themselves

## Metadata

- Year: 2026
- Authors: Hangfan Zhang, Shao Zhang, Kangcong Li, Chen Zhang, Yang Chen, Yiqun Zhang, Lei Bai, Shuyue Hu
- Category: Agent Skill Optimization & Procedural Memory
- Priority: P1
- Links: [arXiv](https://arxiv.org/abs/2606.09498), [arXiv HTML](https://arxiv.org/html/2606.09498), [ExplainX blog](https://explainx.ai/blog/self-harness-agents-improve-themselves-arxiv-2026)
- Keywords: agent harness, self-improvement, weakness mining, harness proposal, proposal validation, regression testing, Terminal-Bench, DeepAgent

## 一句话结论

Self-Harness 把 agent harness 视为可迭代的系统层 artifact：固定 base model 和 evaluator，只让同一个 agent 基于自身 execution traces 挖掘 failure patterns、提出小范围 harness edits，并通过 held-in/held-out regression gate 接受不会回退的改动。

## Related Works

- Prompt engineering / context engineering: 从 prompt、demonstrations、retrieved evidence、memory 和动态 context 改变 fixed model 行为，但通常没有把完整 harness 当成可审计的系统状态。
- Agent harness engineering: ReAct、SWE-agent、Claude Code、Codex、OpenHands、SemaClaw 等说明 tool interface、runtime policy、verification rule 和 orchestration logic 会显著影响 agent 表现。Self-Harness 直接优化这层非参数化 scaffolding。
- Self-improving agents: Reflexion、Self-Refine、STOP、Agentic Context Engineering、Godel Agent、Darwin Godel Machine 等关注 memory、response strategy、program 或 broader agent evolution。Self-Harness 的边界更窄：不更新 model weights，不依赖 stronger external optimizer，只做 bounded harness edit。
- Automated agent / harness design: ADAS、Language Agents as Optimizable Graphs、Meta-Harness 等使用外部搜索或更强 agent 优化 agent design。Self-Harness 强调由 evaluated model 在当前 harness 下提出自身 harness 修改，并用 regression testing 控制回退。
- SkillOpt: SkillOpt 优化单个自然语言 skill document；Self-Harness 优化 system prompt、tool policy、subagents、middleware、verification guidance 等更宽的 harness surface。两者都把 agent 的外部文本/配置状态纳入训练和验证流程。

## 问题与背景

LLM agent 的能力不仅来自 base model，还取决于包住模型的 harness: system prompt、tools、memory、state management、runtime control、verification rule、failure recovery procedure 等。不同模型会有不同 tool-use habit、prompt sensitivity 和错误模式，因此一个对 MiniMax、Qwen 或 GLM 有效的 harness 不一定能迁移到另一个模型。

传统做法依赖 human engineers 手工读 trace、加 prompt rule、改工具封装和调 orchestration。随着模型迭代速度变快，这种 model-specific harness engineering 很难规模化。外部 optimizer 或 stronger agent 可以帮忙，但会引入额外能力假设和成本，且可能不贴合目标模型自己的 failure modes。

论文的问题是：在 fixed model、fixed evaluator、fixed benchmark protocol 下，能否让同一个 agent 根据自身行为证据改进自己的 harness，并通过可记录、可回滚、可验证的方式提升任务通过率？

## 方法与系统设计

Self-Harness 是一个 iterative propose-evaluate-accept loop。输入包括 fixed model `M`、初始 harness `h0`、held-in split、held-out split、evaluator、proposal width 和 rounds。每一轮只修改 harness，不修改 model weights、evaluator 或 benchmark environment。

### Weakness Mining

系统先用当前 harness 在 held-in split 上运行 agent，收集 messages、tool calls、observations、verifier outcomes 和最终 pass/fail。失败样本不会被当作孤立例子，而是根据 verifier-grounded failure signature 聚类。signature 同时包含 terminal verifier cause、agent behavior 的 causal status，以及 trace 暴露出的 reusable mechanism。

这种设计的关键是把表层失败和可修复机制分开。例如 timeout、missing artifact 或 assertion failure 只是 verifier 结果，真正的 harness edit 应该对应更具体的行为机制：没有及时写 required artifact、反复执行无效命令、丢失 shell session 环境、tool error 后没有恢复文件等。

### Harness Proposal

proposal 阶段仍使用同一个 fixed model，而不是 stronger external agent。模型得到的 context 包括 editable harness surfaces、failure patterns、需要保留的 passing behaviors，以及已经尝试过的 edits summary。每个 proposal 必须绑定一个主要 failure mechanism，指出要改的 harness surface、预期行为变化和 regression risk。

论文强调 diverse yet minimal: 一轮并行生成多个 materially distinct candidates，但每个 candidate 都应只改最小必要 surface，避免把 harness 重写成不可解释的大 prompt 或大 workflow。

### Proposal Validation

候选 harness 不会因为 rationale 看起来合理就被接受。系统重新评估 current harness 和 candidate harness 在 held-in 与 held-out split 上的表现。接受规则是 candidate 至少改进一个 split，且另一个 split 不退化；如果评估有随机性，会重复 evaluation 并聚合 pass counts。未改变 editable surface、运行失败或导致 split trade-off 的候选都会被拒绝。

如果同一轮有多个兼容候选通过 validation，它们会 merge 成下一版 harness；否则沿用原 harness。系统记录 changed surfaces、split-wise outcomes、proposal summary、accept/reject decision，使 harness lineage 可以审计。

### 实验实例化

论文在 Terminal-Bench-2.0 上使用一个 minimal DeepAgent-based baseline harness：简短 system prompt、默认文件/编辑/shell tools、`/AGENTS.md` memory source、无 subagents、无 skills、无 runtime control policy。Self-Harness 只允许改声明过的 harness configuration points，例如 instruction、tools、verification guidance、runtime policy 等。

## 创新点

- 把 agent harness 明确建模为 fixed model 外围的可优化系统层状态，而不是一次性 prompt patch。
- 使用 verifier-grounded failure clustering，把失败样本整理为可定位的 harness weakness，降低从单个 bad trace 过拟合的风险。
- 让同一个 evaluated model 提出自身 harness edits，不依赖 stronger external agent；这更贴近 frontier model 无法找到更强 supervisor 的场景。
- 把 harness 修改纳入 regression-gated state transition：每个 accepted edit 都要有行为证据、目标 surface、split-wise evaluation 和非退化条件。
- 强调 small auditable edits，可以和代码审查、CI、benchmark replay、agent skill repository 等工程流程结合。

## 实验与结论

实验使用 Terminal-Bench-2.0 的固定 64-case subset，排除了外部 web 资源不稳定或初始 harness 不支持的 multimodal tasks。每个 candidate 通常评估两次，指标是 verifier pass percentage。模型包括 MiniMax M2.5、Qwen3.5-35B-A3B 和 GLM-5；模型、decoding configuration、tool set、benchmark environment 和 evaluator 都保持固定。

主要结果：

- MiniMax M2.5 held-out pass rate 从 40.5% 提升到 61.9%，held-in 从 43.0% 到 50.0%。
- Qwen3.5-35B-A3B held-out pass rate 从 23.8% 提升到 38.1%，held-in 从 15.1% 到 36.0%。
- GLM-5 held-out pass rate 从 42.9% 提升到 57.1%，held-in 从 47.7% 到 57.0%。
- 三个模型在 held-in 和 held-out 上都没有因为最终 promoted harness 退化，说明 accepted edits 不是简单牺牲 unseen tasks 换取 seen failures。

定性分析显示，最终 harness 的变化具有 model-specific 特征：

- MiniMax M2.5 的 edits 主要处理 missing required artifacts、schema-invalid tool content、stalled tool-use loops，引导 agent 更早创建并验证 required output。
- Qwen3.5 的 edits 强调 dependency precheck、missing-artifact recovery、retry discipline、tool-error-triggered middleware，避免反复失败命令和文件编辑错误导致 artifact 丢失。
- GLM-5 的 edits 关注 late artifacts、external computation、session-scoped tools、implementation-oriented exploration，让 shell 环境状态更持久，并促使 agent 从长期探索转向实现和测试。

论文结论是：即使从稀疏的 initial harness 出发，只要 proposal 受 execution evidence 约束，并经过 regression gate，fixed model 也能产生有用、可审计、可泛化到 held-out tasks 的 harness 改进。

## 局限性与风险

- 实验是 fixed benchmark 上的 bounded harness edits，不是 open-ended self-improvement；不能直接推断长期在线环境会持续提升。
- 依赖高质量 verifier 和 trace records。如果 evaluator 只能给粗粒度 pass/fail，failure signature 和 proposal quality 会受限。
- Held-out split 能降低但不能消除 benchmark-specific overfitting。Terminal-Bench 的 artifact/retry/session 问题可能不完全代表真实软件工程或研究 agent。
- 当前没有找到官方代码仓库，复现需要自行实现 trace clustering、editable harness interface、proposal prompt、candidate evaluation 和 merge policy。
- 接受规则只看 pass-rate non-regression。生产级 harness 还需要安全、权限、成本、latency、可解释性、用户体验等更强 gate。
- 同一个模型既是 actor 又是 proposer，可能系统性忽略自己无法诊断的 failure modes；必要时仍需要 human audit 或外部 verifier。

## 对 LLM Systems 的启发

- Codex / Claude Code 类 agent 的改进对象不只是 prompt，而是完整 execution harness: memory source、tool wrapper、runtime policy、subagent、skill router、verification rule 都应纳入版本化管理。
- Agent harness update 应该像代码变更一样有 evidence、diff、test result 和 rollback path。只凭一次失败后添加一条 instruction，很容易引入 silent regression。
- 对本仓库已有 `paper-analysis` skill、AGENTS/CLAUDE 规范和前端问答设想，可以借鉴 Self-Harness 的做法：记录失败问答/补充论文过程，聚类常见缺陷，再用小改动更新 skill 或 harness，并通过笔记格式校验和 build 作为最低 regression gate。
- Self-Harness 和 SkillOpt 可以形成两层循环：SkillOpt 优化 domain skill，Self-Harness 优化调度、工具、验证和恢复流程。实际系统应区分 skill artifact 与 harness artifact，避免所有经验都堆进一个长 prompt。
- Harness 的 editable surfaces 需要提前声明。没有清晰边界，agent 可能生成过大的 prompt rewrite 或不可部署的工具改动；边界清楚时，自动化改进才更接近工程可落地流程。

## 复现/阅读建议

- 先读 Abstract、Introduction、Section 3 和 Section 4.1，确认 paper 对 harness、failure signature、accepted edit 的定义。
- 复现时不要从全量 agent framework 开始，优先复制论文的 minimal harness 思路：短 system prompt、文件/编辑/shell tools、一个小型 Terminal-Bench-like task set 和确定性 verifier。
- 每个 candidate edit 都保留 patch、proposal rationale、held-in/held-out pass counts、accept/reject reason，避免只保存最终 prompt。
- 对 Codex/Claude Code 风格环境，优先尝试 artifact reliability、retry discipline、tool-error recovery、session state persistence 四类 failure cluster。
- 官方代码暂未找到；如果后续出现项目页或 repo，应优先核对 proposal prompt、failure clustering 规则、split 构造和 candidate merge 逻辑。
