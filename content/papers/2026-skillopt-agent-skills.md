# SkillOpt: Executive Strategy for Self-Evolving Agent Skills

## Metadata

- Year: 2026
- Authors: Yifan Yang, Ziyang Gong, Weiquan Huang, Qihao Yang, Ziwei Zhou, Zisu Huang, Yan Li, Xuemei Gao, Qi Dai, Bei Liu, Kai Qiu, Yuqing Yang, Dongdong Chen, Xue Yang, Chong Luo
- Category: Agent Skill Optimization & Procedural Memory
- Priority: P0
- Links: [arXiv](https://arxiv.org/abs/2605.23904), [project](https://microsoft.github.io/SkillOpt/), [code](https://github.com/microsoft/SkillOpt), [HF Papers](https://huggingface.co/papers/2605.23904)
- Keywords: agent skills, text-space optimization, procedural memory, validation gate, rejected-edit buffer, Codex, Claude Code

## 一句话结论

SkillOpt 把自然语言 skill document 当作 frozen agent 的可训练外部状态，通过 scored rollout、minibatch reflection、bounded add/delete/replace edits 和 held-out validation gate 稳定优化 `best_skill.md`，部署时只加载最终 skill，不引入额外 optimizer model calls。

## Related Works

- Prompt auto tuning / prompt evolution: TextGrad、GEPA、ABSTRAL、EvoTest 等证明 trajectory feedback 可以改进 prompt 或 agent configuration，但主要优化 prompt/system design，而不是可移植的 domain skill artifact。
- Agent skill construction / evolution: SkillsBench、SoK on agentic skills、Trace2Skill、EvoSkill、SkillForge、SkillFoundry、ProcMEM、AutoRefine 等关注 skill discovery、skill library、trajectory lesson distillation 或 evolutionary refinement。SkillOpt 更聚焦一个 compact skill document 如何像训练参数一样被可控优化。
- Reflection-based agents: Self-Refine、Reflexion 等使用语言反馈改进行为；SkillOpt 的区别是把 feedback 收敛到一个可审计、可复用、validation-gated 的 `best_skill.md`。

## 问题与背景

Agent 系统中的 domain adaptation 不只发生在 model weights 或 prompt template 上，也发生在 procedures: 如何查证证据、调用工具、遵守输出格式、处理失败模式。手写 skill 和 one-shot LLM skill 往往静态且脆弱；loosely controlled self-revision 又容易把有用规则覆盖掉，或者把训练 split 上的偶然修复写进部署 artifact。

论文把核心问题定义为：在 target model、execution harness 和 evaluator 固定的情况下，是否能把 skill document 作为 external state，用类似 deep-learning optimizer 的 discipline 来训练它，同时让最终部署仍然只是一个小的文本文件。

## 方法与系统设计

SkillOpt 的输入包括 target domain、initial skill、frozen target model、execution harness，以及 train/selection/test split。训练过程只更新 skill text，不更新 model weights。

### Forward pass: rollout evidence

Target model 在当前 skill 下执行一批训练任务，harness 记录 task metadata、messages、tool calls、observations、command outputs、final answers、verifier feedback 和 scalar score。这个 rollout batch 是后续编辑的 evidence unit。

### Backward pass: minibatch reflection

Optimizer model 将成功和失败 trajectories 分开，按 minibatch 做 reflection。Failure minibatches 产出缺失规则或纠错规则，success minibatches 负责保留已有效的行为。局部 proposals 先分别合并，再在 failure correction 优先的原则下生成候选 edits。

### Bounded text updates

SkillOpt 把 edit budget 当成 textual learning rate。候选 add/delete/replace edits 会被 ranking 后裁剪到预算内，避免一次 rewrite 过大导致已有有效 procedure 被擦掉。论文实现支持 constant、linear、cosine 和 autonomous schedule，默认使用带衰减的 bounded updates。

### Validation gate 和 rejected-edit buffer

候选 skill 必须在 held-out selection split 上严格超过当前 selection score 才会被接受；如果同时超过历史最好结果，就导出为 `best_skill.md`。被拒绝的 edits 不会进入部署 skill，但会进入 rejected buffer，作为同一 epoch 后续 reflection 的 negative feedback。

### Slow/meta update

Epoch 级别的 slow update 比较上一 epoch skill 和当前 skill 在相同训练样本上的表现，把稳定改进、回退、持续失败和稳定成功整理为 protected slow-update guidance。Meta skill 只给 optimizer model 使用，不随 target model 部署，从而把训练时的长程记忆和部署 artifact 分开。

### Harness-agnostic deployment

SkillOpt 通过 adapter 把同一个优化循环接到 direct chat、Codex CLI、Claude Code CLI 等执行环境。最终部署 artifact 是一个 compact `best_skill.md`，target model 在 inference 时只额外读取这个 skill 文本；optimizer model、rejected buffer 和 meta skill 都不在部署路径上。

## 创新点

- 把 agent skill 明确建模为 trainable external state，而不是 prompt 的一次性附属品。
- 将 rollout batch、reflection minibatch、textual learning rate、validation gate、rejected buffer、slow/meta update 组合成一个可控 text-space optimizer。
- 把 target model execution 和 skill optimizer 分离：强 optimizer 只增加训练成本，不增加部署时的 model calls。
- 输出 artifact 是小型、可审计、可跨 model/harness 迁移的 `best_skill.md`，更接近系统工程中可版本化的配置/策略文件。

## 实验与结论

论文在六个 benchmark 上评估：SearchQA、SpreadsheetBench、OfficeQA、DocVQA、LiveMathematicianBench 和 ALFWorld；覆盖七个 target models，并比较 direct chat、Codex 和 Claude Code 三种 execution harness。

主要结果：

- SkillOpt 在 52 个 evaluated `(model, benchmark, harness)` cells 上全部 best 或 tied-best。
- 在 GPT-5.5 direct chat 下，平均比 no-skill 提升 +23.5 points；在 Codex agentic loop 中提升 +24.8 points；在 Claude Code 中提升 +19.1 points。
- GPT-5.5 direct chat 的六个 benchmark 中，SearchQA 从 77.7 到 87.3，SpreadsheetBench 从 41.8 到 80.7，OfficeQA 从 33.1 到 72.1，DocVQA 从 78.8 到 91.2，LiveMathematicianBench 从 37.6 到 66.9，ALFWorld 从 83.6 到 95.5。
- Ablation 显示 bounded textual learning、rejected-edit buffer 和 slow/meta update 都有实际贡献。例如去掉 rejected buffer 会让 SearchQA / SpreadsheetBench / LiveMath 从 87.1 / 77.5 / 61.3 降到 85.5 / 72.9 / 58.9；去掉 meta skill 和 slow update 会让 SpreadsheetBench 明显下降到 55.0。
- Transfer experiments 显示 optimized skill 在 cross-model、cross-harness、cross-benchmark 场景下仍保持正向收益，说明 artifact 不只是记住训练样本格式。
- 论文报告的 GPT-5.5 case studies 中，最终 skills 约 379 到 1,995 tokens，只需要 1 到 4 次 accepted bounded updates，部署成本主要是 prompt token overhead，而不是额外 inference calls。

## 局限性与风险

- 需要可打分的 rollout 和 selection split；对于开放式任务、偏好式目标或长期交互任务，validation gate 的设计会更难。
- 训练期成本不低，特别是长轨迹、多模态 context 或复杂 tool traces 会显著增加 optimizer token cost。
- 虽然 held-out gate 降低了过拟合风险，但 skill 仍可能学到 benchmark-specific answer format 或 harness convention，需要人工审计。
- 当前主要优化单个 domain skill；跨多 domain 的 skill library、skill routing 和技能冲突管理还没有解决。
- "Zero deployment overhead" 指没有额外 model calls 或 weight updates，但最终 skill 仍占用 context budget，并可能和已有 system/developer instructions 发生优先级冲突。

## 对 LLM Systems 的启发

- Agent skill 可以被当成系统层 artifact 管理：有训练数据、validation gate、版本记录、accepted/rejected edits 和最终发布件，而不是靠人工 prompt patch。
- 对 Codex / Claude Code 类执行环境，trace summary、tool output 和 verifier feedback 应作为一等训练信号进入 skill 优化循环。
- Offline optimizer / online deployment 的分离很实用：可以用更强模型训练 skill，再把小文本 artifact 部署到更便宜或更小的 target model 上。
- Skill optimization 的关键不是生成更多规则，而是控制 edit step size 并用 held-out score 阻止 silent drift。

## 复现/阅读建议

- 先读 Introduction、Method 和 Appendix C 的 optimizer prompt contracts，确认 add/delete/replace edits、ranking、slow update 和 meta skill 的实际格式。
- 复现时优先选择 SpreadsheetBench 或 SearchQA：前者能体现 procedural tool-use gains，后者更容易搭建 exact-match scorer。
- 每次实验至少保留 no-skill、human/seed skill、one-shot LLM skill 和 SkillOpt 对比，并同时报告 train、selection、test 分数。
- 重点检查 `best_skill.md`、rejected edits 和 `edit_apply_report.json`，确认收益来自通用 procedure，而不是写入样本特例。
- 如果目标是接入工程 agent，优先做 cross-harness 测试：同一个 skill 文件分别在 Codex 和 Claude Code 风格环境中运行，观察哪些规则是真正 harness-agnostic。
