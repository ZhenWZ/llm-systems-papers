# SkillOpt: Executive Strategy for Self-Evolving Agent Skills

## Metadata

- Year: 2026
- Authors: Yifan Yang, Ziyang Gong, Weiquan Huang, Qihao Yang, Ziwei Zhou, Zisu Huang, Yan Li, Xuemei Gao, Qi Dai, Bei Liu, Kai Qiu, Yuqing Yang, Dongdong Chen, Xue Yang, Chong Luo
- Category: Agent Runtime & Skill Optimization
- Priority: P1
- Links: [project](https://microsoft.github.io/SkillOpt/), [arXiv](https://arxiv.org/abs/2605.23904), [code](https://github.com/microsoft/SkillOpt), [HF Papers](https://huggingface.co/papers/2605.23904)
- Keywords: agent skills, text-space optimization, validation gate, prompt optimization, procedural memory, Codex, Claude Code, tool-use harness

## 一句话结论

SkillOpt 把自然语言 `skill` 文件当作 frozen agent 的可训练外部状态，通过 rollout evidence、minibatch reflection、bounded add/delete/replace edits、held-out validation gate 和 slow/meta update，把 agent skill 从手写 prompt 变成可验证、可复用、部署时零额外模型调用的优化产物。

## Related Works

- Prompt / configuration optimization: TextGrad、GEPA、DSPy、LLM-as-optimizer 等工作把 prompt 或 pipeline 配置作为可优化对象；SkillOpt 的差异是优化长期可复用的 domain skill，而不是一次性 prompt 或完整 agent 配置。
- Trajectory reflection: Reflexion、Self-Refine、Trace2Skill 等从执行轨迹中抽取经验；SkillOpt 更强调训练式控制，包括 batch evidence、textual learning rate、validation gate 和 rejected-edit buffer。
- Agent skill / procedural memory: SkillsBench、SoK on agentic skills、AutoSkill、SkillForge、SkillClaw、SkillX 等研究 skill 的构建、演化、库化和共享；SkillOpt 聚焦单个 compact skill artifact 如何被稳定训练。
- Agent runtime harness: Codex、Claude Code、SWE-agent、Toolformer、ReAct 等提供工具调用、文件操作、verifier 和多步交互背景；SkillOpt 试图让同一优化接口跨 direct chat、Codex 和 Claude Code 生效。

## 问题与背景

Frontier LLM agent 的 domain adaptation 不只发生在 model weights 或 system prompt 中，也发生在 agent 如何收集证据、调用工具、维护状态、处理格式约束、避免循环和验证输出的 procedural layer。现实系统常把这层经验写成 skill 文档或操作手册，但这些文档通常来自人工总结、一次性 LLM 生成，或松散的自我反思迭代，缺少类似深度学习训练中的 batch、step size、validation 和 rejected update 机制。

SkillOpt 要解决的问题是：当目标模型和执行 harness 都固定时，能不能把 skill 文档本身作为 external trainable state，用带反馈的优化过程稳定提升任务表现，同时保持部署简单。这个问题对 LLM systems 很实际，因为很多闭源 frontier model 无法 fine-tune，开源模型 fine-tune 又成本高；相比之下，一个 compact `best_skill.md` 更容易审计、迁移和灰度发布。

## 方法与系统设计

SkillOpt 的核心角色分离为 frozen target model 和 optimizer model。Target model 在当前 skill 下执行任务，harness 记录 task metadata、messages、tool calls、observations、command outputs、verifier feedback 和 final scores。Optimizer model 只在训练阶段读取这些 scored trajectories，并生成对 skill 文档的结构化 patch。

训练循环类似 text-space SGD。每个 step 先采样 rollout batch，然后把成功和失败轨迹拆成 reflection minibatches。失败 minibatch 主要提出 missing / corrective rules，成功 minibatch 主要提炼应保留的有效行为。局部 proposal 经过 failure merge、success merge 和 failure-prioritized final merge，过滤重复、矛盾和过于实例化的建议。

Bounded text update 是论文的关键控制量。SkillOpt 把每步最多应用的 edit 数称为 textual learning rate，默认用 patch mode 的 `append`、`insert_after`、`replace`、`delete` 四类原子操作，并可用 constant、linear、cosine 或 autonomous schedule 控制更新幅度。这样相邻 skill 版本保持连续，避免无约束 rewrite 抹掉已有有效规则。

每个 candidate skill 都必须在 held-out selection split 上用同一个 target model 和 harness 重新评估。只有 selection score 严格提升才接受；tie 也拒绝。被拒绝的 edits 不进入部署 skill，但会进入 rejected-edit buffer，作为同一 epoch 后续 optimizer 调用的 negative feedback。

Epoch 结束时，slow update 会在相同 sampled tasks 上比较前一 epoch 和当前 epoch 的 skill，归纳 improvements、regressions、persistent failures 和 stable successes，再写入受保护的 slow-update 区域。Optimizer-side meta skill 则只面向后续 optimizer 调用，记录哪些编辑方向有效或有害，不随 `best_skill.md` 一起部署。

部署产物是一个 compact `best_skill.md`。论文强调 target model、backend、harness 和 evaluator 在优化期间保持固定；上线时只把最终 skill 放入 agent context，不需要 optimizer model、不改 weights，也不增加额外 inference-time model calls。

## 创新点

- 把 agent skill 明确定义为 frozen agent 的 external trainable state，而不是 prompt 工程副产物。
- 用 rollout batch、reflection minibatch、textual learning rate、validation gate、rejected-edit buffer 和 slow/meta update，把 skill editing 约束成可复现的训练循环。
- 将优化输出限制为 compact、可审计、可迁移的 `best_skill.md`，把训练成本和部署成本清晰分离。
- 同时评估 direct chat、Codex 和 Claude Code，说明方法针对的是 agent procedure，而不是某个单一 prompt 格式。
- 通过 patch representation 和 protected slow-update 区域，让局部快速编辑和跨 epoch 长期记忆分层，减少 skill drift。

## 实验与结论

论文在 SearchQA、SpreadsheetBench、OfficeQA、DocVQA、LiveMathematicianBench 和 ALFWorld 六个 benchmark 上评估，目标模型覆盖 GPT 系列和 Qwen 系列共七个 target models，并包含 direct chat、Codex、Claude Code 三种 execution harness。主要 baselines 包括 no skill、human skill、one-shot LLM skill、Trace2Skill、TextGrad、GEPA 和 EvoSkill。

主结果显示 SkillOpt 在 52/52 个 model x benchmark / harness x benchmark cell 中达到 best 或 tied-best。以 GPT-5.5 为例，project page 报告 direct chat 平均相对 no skill 提升 +23.5 points；Codex harness 和 Claude Code harness 分别提升约 +21.8 和 +18.6 points。arXiv 摘要给出的总体表述是 direct chat +23.5、Codex agentic loop +24.8、Claude Code +19.1；不同聚合口径需要以论文表格为准。

收益最大的一类任务是 procedural workloads，例如 SpreadsheetBench、OfficeQA、LiveMath 和 ALFWorld。它们需要工具调用、格式约束、状态维护、验证和避免循环，正好对应 skill 能编码的 reusable procedure。SearchQA 和 DocVQA 也有正收益，但提升更多来自答案类型、证据绑定和 canonical entity 选择等规则。

消融实验支持几个关键设计。无 bounded learning rate 或动态/无约束更新会降低稳定性；去掉 rejected-edit buffer 会让 optimizer 重复有害方向；去掉 meta skill 和 slow update，尤其会伤害 SpreadsheetBench 这类需要长程程序化约束的任务。Validation gate 还表现为一个实际的 model selection 机制，许多候选 edits 被拒绝，最终部署 skill 保持 compact。

Transfer 实验表明 optimized skill 不是单个 benchmark split 的记忆。论文报告 cross-model、cross-harness 和 cross-benchmark transfer 均保持正迁移：例如 Codex 训练出的 SpreadsheetBench skill 可迁移到 Claude Code，数学 benchmark 上 OlympiadBench skill 对 Omni-MATH 也有正收益。代表性 learned rules 多是程序性约束，例如先检查 workbook 结构、绑定文档视觉区域、维护 ALFWorld 搜索 frontier，而不是硬编码具体样本。

## 局限性与风险

- SkillOpt 依赖 scored trajectories 和 held-out selection split，最适合有自动 verifier、exact match、可执行检查或稳定评分器的任务；开放式任务需要更强的人类或模型评估。
- 训练阶段需要额外 rollout compute 和 optimizer model calls。虽然部署时只有 `best_skill.md`，但一次性任务可能无法摊销训练成本。
- 当前设计优化单个 portable skill，不管理大型 skill library。高度异质的 domain 可能需要多 skill 路由、版本管理和冲突检测。
- Optimized skill 仍可能吸收训练分布中的 domain-specific heuristics，跨模型、跨 harness 或跨任务迁移前必须保留 held-out evaluation。
- 论文结果依赖特定 benchmarks、harness contract 和 optimizer prompt contracts；复现时数据 split、verifier、工具环境和模型版本差异可能显著影响收益。

## 对 LLM Systems 的启发

- Agent systems 的优化对象不只有 weights、prompt 和 runtime scheduler，还包括可审计的 procedural memory。把 skill 当作系统状态，可以给 closed model adaptation 提供一条低侵入路径。
- 对 agent workflow 来说，validation gate 比自我反思本身更关键。没有 held-out gate，reflection 只是生成 plausible text；有 gate 后才变成 propose-and-test optimization。
- Skill artifact 的工程价值来自 compactness、traceability 和 zero inference-time optimizer cost。上线时可以像配置文件一样审计、diff、回滚和 A/B test。
- 训练式控制量可以迁移到 prompt/skill 工程：batch size 控制证据噪声，edit budget 控制 step size，rejected buffer 提供 negative feedback，slow update 类似 momentum 或 long-horizon regularization。
- 对 Codex/Claude Code 这类工具型 agent，skill 里真正有价值的内容往往是 verifier discipline、文件/表格/环境检查顺序、状态 ledger 和 output contract，而不是泛泛的推理建议。

## 复现/阅读建议

- 先读 Introduction 和 Method，抓住 target model / optimizer model / harness / split 四个边界，再看 Appendix C 的 optimizer prompt contracts 和 patch safeguards。
- 复现时优先选择有稳定 evaluator 的任务，例如 SearchQA、SpreadsheetBench 或 LiveMath；先固定 train/selection/test split，再比较 no skill、human skill、one-shot LLM skill 和 SkillOpt。
- 记录的不应只有最终 accuracy，还应包括 accepted/rejected edit 数、selection score 曲线、test score 曲线、训练 token 成本、`best_skill.md` token 数和每步 patch diff。
- 如果用于本仓库的 agent workflow，可以从一个小 domain skill 开始，例如 paper note writing 或 benchmark extraction，把 validation gate 设计成格式校验、引用完整性和人工抽查结合。
