# CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation

## Metadata

- Year: 2026
- Authors: Weinan Dai, Hanlin Wu, Qiying Yu, Huan-ang Gao, Jiahao Li, Chengquan Jiang, Weiqiang Lou, Yufan Song, Hongli Yu, Jiaze Chen, Wei-Ying Ma, Ya-Qin Zhang, Jingjing Liu, Mingxuan Wang, Xin Liu, Hao Zhou
- Category: Kernel & Operator Systems
- Priority: P0
- Links: [HF Papers](https://huggingface.co/papers/2602.24286), [arXiv](https://arxiv.org/abs/2602.24286), [project](https://cuda-agent.github.io/), [code](https://github.com/BytedTsinghua-SIA/CUDA-Agent), [dataset](https://huggingface.co/datasets/BytedTsinghua-SIA/CUDA-Agent-Ops-6K)
- Keywords: CUDA kernel generation, agentic RL, KernelBench, torch.compile, profiling, reward shaping, CUDA-Agent-Ops-6K, OpenHands-style agent loop

## 一句话结论

CUDA Agent 把高性能 CUDA kernel generation 从 test-time prompt/refinement 问题推进到 large-scale agentic RL 训练问题：通过 6K synthesized ops、skill-augmented CUDA execution environment、robust verification/profiling reward 和 staged actor/critic warm-up，让 agent 学会面向 `torch.compile` baseline 写出正确且更快的 CUDA kernels。

## Related Works

- Training-free CUDA kernel generation: STARK、ReGraphT、EvoEngineer、CudaForge 等使用 planning/search/retrieval/evolution/profiling feedback 做 test-time refinement。CUDA Agent 认为这类方法受限于 base model 的 intrinsic CUDA coding ability，主要靠推理时搜索补救。
- Fine-tuning / RL for kernel generation: Kevin、CUDA-L1、ConCuR、KernelCoder 等尝试用 execution feedback 或合成 trajectory 训练模型。CUDA Agent 的区别是扩展数据规模、隔离 train/test、加入 skill-augmented environment，并专门处理 long-horizon agentic RL 稳定性。
- KernelBench: 论文把 KernelBench 作为核心评测基准，目标不是只生成能编译的 CUDA，而是通过 correctness check 后相对 `torch.compile` 获得真实 runtime advantage。
- Agent skills / procedural memory: 论文把 CUDA coding workflow 写成 `SKILL.md`，让模型在 ReAct-style loop 中按 profiling、implementation、compilation、verification、optimization 的流程迭代。这和 SkillOpt/Self-Harness 的外部 skill/harness artifact 思路相通。
- Kernel & compiler systems: 与 CODA、Triton、CUTLASS/CuTeDSL、TVM、`torch.compile` 等方向相邻。CUDA Agent 不是提出新的 kernel DSL，而是训练 agent 在现有 CUDA/toolchain 下做低层实现和性能调优。

## 问题与背景

GPU kernel optimization 对深度学习系统很关键，但 CUDA performance engineering 需要理解 memory hierarchy、warp/block mapping、coalesced access、shared memory、Tensor Core、profiling toolchain 和数值误差边界。通用 LLM 虽然能写普通代码，但在 CUDA kernel generation 上通常无法稳定超过 `torch.compile`，更难在复杂 operator sequence 或 model block 上做可靠优化。

已有方法主要有两个瓶颈。第一类 training-free workflow 通过多 agent、搜索树、retrieval graph 或 evolutionary editing 在测试时反复改 kernel；它能放大已有能力，但无法从根本上训练模型的 CUDA optimization skill。第二类 fine-tuning/RL 方法通常受限于高质量训练任务不足、训练规模小、固定执行反馈 loop、数据污染或 reward hacking 风险，导致收益难以迁移到干净 test split。

CUDA Agent 的核心问题是：能否构建一个可大规模 rollout 的 CUDA coding environment，让模型在长上下文、多轮工具交互中通过 execution reward 学到真正的 kernel optimization strategy，而不是只学会绕过 verifier 或复用 benchmark 模板？

## 方法与系统设计

CUDA Agent 由三部分组成：scalable training data synthesis、skill-integrated agent loop、stable agentic RL training pipeline。

### 1. Scalable Training Data Synthesis

训练不直接依赖人工写好的 expert CUDA kernels，而是把 PyTorch reference operator 作为 RL task source。数据流水线分三步：

- Seed problem crawling: 从 `torch` 和 `transformers` libraries 挖掘 reference operators，每个 operator 表示为带 initialization 和 forward method 的 Python class。
- Combinatorial synthesis: 用 LLM 从 torch operator classes 中采样最多 5 个算子，顺序组合成 fused multi-operator tasks。组合任务不等价于逐个优化后串联，因为 fusion 会改变 intermediate materialization、register/SMEM pressure、occupancy 和 data layout。
- Execution-driven filtering: 过滤必须同时在 eager 和 compile mode 可运行，排除 stochastic operators；用 anti-hacking 检查去掉 constant 或 indistinguishable outputs；把 eager runtime 限制在 1 ms 到 100 ms；并移除与 KernelBench test cases 高相似的样本。

最终数据集为 `CUDA-Agent-Ops-6K`，包含 6,000 个 curated training samples，用于 RL rollout 和 curriculum。

### 2. Skill-Integrated Agent Loop

执行环境采用 OpenHands/ReAct-style agent loop。Agent 可以使用 shell、glob、multi-file edit、todo 等 coding tools，并在 GPU sandbox 中运行 compile、verify、profile。

CUDA coding workflow 被写成 skill specification：先用 `profile.py` 分析原始 PyTorch implementation，定位 kernel launch、memory access、operator fusion 等机会；再在 `model_new.py` 和 CUDA/C++ binding 中实现 custom operators；随后编译、验证正确性并 profile；持续迭代，直到通过 numerical correctness checks 且至少比 `torch.compile` 快 5%。

这种设计把 CUDA engineering procedure 从隐式 prompt 变成可复用 artifact，同时让训练轨迹包含真实 compiler error、runtime error、accuracy mismatch 和 profiling result。

### 3. Robust Reward 与 Anti-Hacking

论文不用 raw speedup 直接作为 reward，因为不同 task 的优化难度差异很大，raw speedup 会偏向 easy kernels 和 outliers。CUDA Agent 使用离散 milestone reward:

- correctness failed: `-1`
- correct but not faster than eager: `1`
- faster than eager: `2`
- faster than eager and `torch.compile`: `3`

环境还加入了反 reward hacking 机制：保护 verify/profile scripts，禁止 fallback 到原始 PyTorch 路径，使用多输入 correctness checks，同步 warm-up profiling，限制 web retrieval，并通过 process-level isolation 防止 agent 修改 evaluator 或利用环境漏洞。

### 4. Stable Long-Horizon Agentic RL

训练 pipeline 分阶段稳定长上下文 agentic RL：

- Single-turn PPO warm-up: 先提升基础 CUDA generation 能力，避免一开始就在长轨迹中学习崩溃。
- Actor initialization: 从 rollout 中筛选 positive trajectories 做 Rejection Fine-Tuning，去掉低效循环和非法 tool-call pattern。
- Critic initialization: 通过 value pretraining 提高早期 advantage estimation 稳定性。
- Multi-turn agentic RL: 在最高 128k context、150 training turns、evaluation 最多 200 turns 的设置下训练 coding agent。

## 创新点

- 把 CUDA kernel generation 作为 agentic RL 问题系统化处理，而不是只做 inference-time search 或单轮 code generation。
- 用 synthesized fused operator tasks 构造 6K 训练集，并通过 execution filtering 和 contamination control 降低 benchmark leakage 风险。
- 将 `SKILL.md`、compiler/profiler/verifier、GPU sandbox 和 reward schedule 组合成 non-hackable CUDA development environment。
- 使用 staged actor/critic warm-up 解决 long-horizon coding RL 的稳定性问题，让模型能学习 debug、profile、rewrite 和 optimize 的多轮策略。
- 公开项目页、代码仓库和 `CUDA-Agent-Ops-6K` 数据集，使该方向比纯闭源 agent benchmark 更容易复现和审计。

## 实验与结论

论文在 KernelBench 上评估，主要比较 `torch.compile`、强 proprietary models 和 CUDA Agent。项目页报告的 overall 结果为：

- Pass Rate: 98.8%
- Faster Rate vs. Eager: 98.4%
- Faster Rate vs. Compile: 96.8%
- Geomean Speed-up vs. Eager: 2.60x
- Geomean Speed-up vs. Compile: 2.11x

Level-3 面板结果为：Pass Rate 94%，Faster Rate vs. Compile 90%，Speed-up vs. Compile 1.52x。arXiv/HF 摘要中还总结 CUDA Agent 在 KernelBench Level-1、Level-2、Level-3 上相对 `torch.compile` 的 faster rate 为 100%、100%、92%，并称在 hardest Level-3 setting 上比 Claude Opus 4.5、Gemini 3 Pro 等 strongest proprietary baselines 高约 40 个百分点。项目页 main-results 面板和摘要中的 Level-3 数字存在 90%/92% 的展示差异，复现时应以论文表格和评测脚本为准。

Case studies 展示了 agent 学到的典型优化模式：

- Algebraic simplification: 把显式 diagonal matrix multiplication 改写为 row-wise scaling，避免构造 diagonal matrix 和 GEMM。
- Kernel fusion: 把多段 arithmetic、summation、division、scaling 或 residual+activation 合并，减少 intermediate tensors 和 kernel launches。
- Memory access optimization: 使用 coalesced access、shared-memory reductions、vectorized loads 等降低 memory traffic。
- Hardware-aware optimization: 在可接受数值误差范围内启用 TF32/Tensor Core，或调整 cuDNN convolution/bias/activation API。
- Library-aware optimization: 在复杂 block 上不一定全部手写 kernel，而是识别可替换为 fused cuDNN primitive 的路径。

总体结论是：当训练数据、agent environment、reward 和 RL 稳定化都足够系统化时，LLM-based CUDA kernel generation 可以在 KernelBench 上成为 `torch.compile` 的竞争性替代方案，而不仅是 demo 级代码生成。

## 局限性与风险

- Baseline 主要是 `torch.compile` 和 proprietary LLM agents，论文明确没有和 TVM 等更复杂 compiler framework 比较；这限制了对 compiler-tuned search systems 的结论外推。
- 训练依赖大规模 GPU pool、process-level isolation、长轨迹 rollout 和 profiling infrastructure，工程成本高，不适合轻量复现。
- KernelBench 仍是 bounded benchmark；真实生产 kernel 还涉及多 GPU、dynamic shape、mixed precision policy、framework integration、版本兼容和安全审计。
- Reward 与 correctness checks 决定 agent 的学习边界。如果 verifier 不覆盖 corner cases，agent 仍可能学到 unsafe shortcut。
- CUDA Agent 面向 CUDA/NVIDIA ecosystem；迁移到 Triton、ROCm、Ascend C 或其他 backend 需要重新设计 toolchain、profiling 和 reward。
- `SKILL.md` 可以提升流程稳定性，但也可能把当前 environment conventions 固化进模型，需要区分通用 CUDA skill 和 benchmark-specific workflow。

## 对 LLM Systems 的启发

- Kernel generation 的关键不是让模型“一次写对 CUDA”，而是训练它掌握 profile -> implement -> compile -> verify -> optimize 的闭环。
- 对 AI infra 工程 agent，execution environment 和 reward design 与模型同等重要。没有不可篡改 verifier/profile scripts，很容易得到 benchmark hacking 而不是真实性能。
- 数据合成不应只堆 primitive ops；fused multi-operator tasks 更能训练 agent 理解 graph-level fusion、intermediate materialization 和 memory traffic。
- `SKILL.md` 这种 procedure artifact 可以成为 agent RL 的一部分：既提供流程 scaffold，也让 trajectory 更可解释。它和 SkillOpt/Self-Harness 可以结合，用自动化方式持续改进 CUDA coding skill 与 harness。
- 对本仓库的 Kernel & Operator Systems 分类，CUDA Agent 与 CODA 形成互补：CODA 约束 kernel authoring abstraction，CUDA Agent 训练 agent 在真实 CUDA/toolchain 中搜索和调优实现。

## 复现/阅读建议

- 先读 Abstract、Section 3 和项目页 pipeline，确认 data synthesis、agent loop、reward schedule 与 staged RL 的边界。
- 复现优先从 released `CUDA-Agent-Ops-6K` 和 GitHub workflow 入手，不要先尝试完整 RL 训练；完整训练需要大规模 GPU 资源。
- 最小复现可以只做 evaluation agent：固定模型、加载 CUDA skill、在一小批 KernelBench tasks 上运行 compile/verify/profile loop，记录 pass rate 和 speedup vs. eager/compile。
- 对 reward hacking 做硬约束：verify/profile scripts 只读挂载、禁止 fallback 到 PyTorch reference、用多输入测试和 warm-up 同步计时。
- 对 case study 要重点看 agent 为什么选择 algebraic simplification、fusion、shared-memory reduction、TF32/cuDNN path，而不是只看最终 speedup 数字。
