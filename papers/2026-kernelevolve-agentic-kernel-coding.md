# KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta

## Metadata

- Year: 2026
- Authors: KernelEvolve Team / Meta Platforms; arXiv lists Gang Liao, Hongsen Qin, Ying Wang, Alicia Golden, Michael Kuchnik, Yavuz Yetim, Jia Jiunn Ang, Chunli Fu, Yihan He, Samuel Hsia, Zewei Jiang, Dianshi Li, Uladzimir Pashkevich, Varna Puvvada, Feng Shi, Matt Steiner, Ruichao Xiao, Nathan Yan, Xiayu Yu, Zhou Fang, Roman Levenstein, Kunming Ho, Haishan Zhu, Alec Hammond, Richard Li, Ajit Mathews, Kaustubh Gondkar, Abdul Zainul-Abedin, Ketan Singh, Hongtao Yu, Wenyuan Chi, Barney Huang, Sean Zhang, Noah Weller, Zach Marine, Wyatt Cook, Carole-Jean Wu, Gaoxiang Liu
- Category: Kernel & Operator Systems
- Priority: P0
- Links: [arXiv](https://arxiv.org/abs/2512.23236), [Meta Engineering](https://engineering.fb.com/2026/04/02/developer-tools/kernelevolve-how-metas-ranking-engineer-agent-optimizes-ai-infrastructure/), [ISCA 2026](https://iscaconf.org/isca2026/)
- Keywords: agentic kernel coding, heterogeneous accelerators, KernelBench, MTIA, Triton, CuTe DSL, retrieval-augmented prompting, graph-based search, production kernels

## 一句话结论

KernelEvolve 把 production kernel optimization 建模为面向 heterogeneous hardware 的 agentic search problem：给定 operator spec、target hardware 和 performance objective，系统通过 retrieval-augmented prompt synthesis、graph/tree search、profiling-driven feedback 和持久化知识库，在 NVIDIA/AMD/MTIA/CPU 等平台上自动生成并验证 production-grade kernels。

## Related Works

- Traditional kernel optimization: cuBLAS、cuDNN、rocBLAS、TVM、Halide、Triton、CUTLASS/CuTe DSL、TLX、Helion 等降低了手写 kernel 成本，但仍依赖专家为新 operator、新 shape、新 hardware generation 调整 schedule、tiling、pipeline 和 memory movement。
- LLM-based kernel generation: KernelBench、KernelLLM、KernelAgent、AutoTriton、TritonRL、GEAK、Kevin、AlphaEvolve 等证明 LLM 可以用 execution feedback 生成或优化 kernels。KernelEvolve 的区别是面向 Meta production DLRM/ranking stack，覆盖 heterogeneous hardware、proprietary MTIA、production operator long tail 和部署基础设施。
- Agentic code search: tree-of-thought、MCTS、evolutionary search、inference-time scaling 等思想被用于在候选实现空间中搜索。KernelEvolve 把每个 candidate kernel 作为 search node，并把 correctness/performance/profiling feedback 转回 LLM context。
- Procedural memory / skill systems: SkillOpt 和 Self-Harness 优化 agent skill 或 harness；KernelEvolve 的 shared knowledge base 和 self-evolving skill library 则把成功 tiling、debugging 和 hardware-specific patterns 写回后续 search context。
- CUDA Agent: CUDA Agent 通过 agentic RL 训练 CUDA kernel generation 能力；KernelEvolve 更强调 production heterogeneous accelerator coverage、retrieval knowledge injection 和 graph-based search harness。

## 问题与背景

Meta 的 recommendation/ranking infrastructure 面临三维组合爆炸：hardware types and generations、model architectures、operator types。NVIDIA GPU、AMD GPU、MTIA 和 CPU 的 memory hierarchy、instruction set、execution model 不同；推荐模型从 embedding-based DLRM 到 sequence learning、GEM、Meta Adaptive Ranking Model，不断引入新 operator；production workload 还包含 vendor libraries 覆盖不到的 long-tail preprocessing transforms 和 custom model operators，例如 feature hashing、bucketing、sequence truncation、feature interaction 和 specialized attention variants。

如果缺少 accelerator-native kernel，这些 operator 可能回退到 CPU，导致 disaggregated serving 和额外网络开销；或者走未优化路径，无法利用硬件。传统做法让 kernel experts 针对每个 chip generation、model family、operator 和 shape 手写/调优，但这个矩阵已经无法靠人工规模化维护。

论文要解决的问题是：如何让一个 agentic kernel coding system 在 production constraints 下持续生成、验证、优化并复用 kernels，尤其是在 public LLM 从未见过的 proprietary hardware（如 MTIA）上也能工作。

## 方法与系统设计

KernelEvolve 的输入是 kernel specification、target hardware、runtime context 和 performance goals。输出是通过 correctness/performance validation 的 kernel implementation。系统不是一次生成代码，而是长运行的 closed-loop optimization harness。

### LLM Synthesizer

LLM synthesizer 生成候选 kernels，覆盖多层 programming abstractions：Triton、TLX、CuTe DSL、FlyDSL，以及 CUDA、HIP、MTIA C++ 等低层后端。Prompt 不是静态模板，而是由 runtime diagnostics、hardware constraints、prior candidate histories 和 retrieved knowledge 动态构造。

### Graph-Based / Tree Search

KernelEvolve 将 optimization process 表述为 graph-based search：包含 selection policy、universal operator、fitness function 和 termination rule。每个 candidate kernel 是 search tree 的一个 node；search engine 根据历史表现选择 promising nodes，应用 transformations，评估结果，然后决定继续探索、回溯或终止。

关键设计是 memory operator。每个 node 可以继承 parent trajectory、比较 sibling variants、融合 parent/sibling insights，或从 clean slate 重新开始。这使 search 不只是独立采样，而是能在 exploit 已知有效方向和 explore 新策略之间切换。

### Retrieval-Augmented Knowledge Base

为解决 hardware heterogeneity，系统维护分层知识库：

- correctness constraints: 保证生成实现满足 operator semantics 和 target backend 约束；
- platform-agnostic optimization guidance: debugging、tiling、memory coalescing、fusion、profiling interpretation 等通用策略；
- hardware-specific documentation: NVIDIA/AMD/MTIA 的 architecture manuals、instruction sets、memory hierarchy、optimization patterns。

检索由 runtime signal 触发：memory bandwidth bottleneck 会检索 memory hierarchy 文档，compile error 会检索 debugging guidance。对于 MTIA 这种 public LLM 训练语料中不存在的 proprietary accelerator，KernelEvolve 通过知识注入让模型在 inference time 获得架构和编程约束。

### Automated Evaluation and Profiling

候选 kernel 必须通过 correctness 和 performance evaluation。工具栈包括 TritonBench、PyTorch Profiler、NCU、Proton、MTIA Insight 等：既看 end-to-end speedup，也看 kernel launch overhead、host-device sync、occupancy、memory throughput、instruction mix、pipeline behavior、PE utilization、cache behavior 等诊断信号。

系统把 profiling 统一到 compiler-centric abstraction 中：job graph 插入 MLIR-level instrumentation，profiling passes 收集指标，trace synthesis 产生结构化结果。Search engine 不只看到 “kernel A 比 kernel B 快”，还看到瓶颈是 memory-bound、compute-bound 还是 occupancy-limited，并把诊断反馈到下一轮 prompt。

### Shared Data Foundation and Agentic RL

每次 optimization session 都沉淀 search tree、candidate implementations、diagnostics 和高性能轨迹。成功的 tiling、fusion、debugging pattern 会被 distill 成 reusable skills，写回 shared knowledge base。

Meta 工程文还提到这些 trajectory 可以用于 post-training 更小的 specialized models：reward 来自 measured kernel performance，使系统形成 data flywheel。论文主体更强调 graph search + knowledge base + production deployment；RL 属于后续演进方向和工程闭环的一部分。

## 创新点

- 把 production kernel authoring 从一次性 code generation 提升为 graph-based agentic search + evaluation harness。
- 支持 heterogeneous accelerators：同一框架覆盖 NVIDIA、AMD、MTIA、CPU，并用 retrieval knowledge injection 替代每个平台手写 prompt template。
- 将 profiling diagnostics 作为一等反馈信号，而不是只用 scalar latency/speedup 指标。
- 用 persistent knowledge base 和 self-evolving skill library 复用跨 session 的优化经验，使后续任务能继承更好的 starting point。
- 在 proprietary hardware absent from public training corpora 的情况下，通过文档注入和 closed-loop validation 生成可部署 kernels，缓解新硬件 software ecosystem 滞后的问题。

## 实验与结论

论文和 Meta 工程文报告了 benchmark 与 production 两类结果：

- KernelBench: 在 250 个 kernel optimization problems、三个 difficulty levels 上达到 100% pass rate；这里 pass 同时要求功能正确且比 PyTorch reference implementation 更快。
- PyTorch ATen operators: 在 160 个 ATen operators、三个 heterogeneous hardware platforms 上验证 100% correctness，共 480 个 operator-platform configurations。
- Production workloads: 论文结论总结 production workloads 相对 PyTorch baselines 有 1.25-17x speedups。
- MTIA ads model: Meta 工程文报告 KernelEvolve-generated kernels 在 MTIA 上带来超过 25% training throughput improvement。
- NVIDIA ads inference: 工程文报告在已有 `torch.compile` 和 vendor libraries 优化的模型上仍带来超过 60% inference throughput improvement。
- Development velocity: kernel development 从 weeks of expert effort 压缩到 hours of automated search and evaluation。

Case studies 覆盖 convolutional transformer、heterogeneous hardware convolution、WuKong/InterFormer kernel fusion、MTIA preprocessing kernels（如 MapId Transform、MergeBucketizedDense Transform）和 sequence learning Batch Event Truncate。论文特别强调 MTIA v2i/v3 的 preprocessing kernels：在 coverage 不完整的平台上，KernelEvolve 不只是优化性能，还提供 on-device execution path，避免 CPU fallback；在 coverage 更完整的平台上仍可通过 fusion 和硬件特化调优获得 2-3x 等级收益。

## 局限性与风险

- KernelEvolve 本体没有公开官方代码仓库；公开资料主要是 arXiv 论文和 Meta Engineering 解读，复现只能复刻架构思想，不能直接运行原系统。
- 生产结果依赖 Meta 内部 workload、MTIA toolchain、profiling infrastructure、FaaS evaluation、knowledge base 和 serving constraints；外部复现需要替换为公开硬件和公开 benchmark。
- 100% correctness/pass rate 是在定义好的 benchmark/operator set 上成立；真实生产还需要长期 CI、回滚、安全审计、数值容忍度和跨版本稳定性。
- RAG 注入 proprietary hardware docs 能解决知识缺失，但也引入知识库质量、版本漂移、权限隔离和泄露风险。
- Search-based optimization 可能消耗大量 test-time compute；论文提到未来可做 massively parallel search，但成本、调度和结果可复现性需要工程治理。
- Agent 生成的 kernel 可能过度贴合当前 shape/workload；需要明确 shape coverage、dynamic shape policy 和 fallback path。

## 对 LLM Systems 的启发

- Kernel agent 的核心竞争力不是单个模型会不会写 Triton/CUDA，而是 search harness、profiling feedback、knowledge injection 和 production validation 的组合。
- 对 proprietary accelerators，LLM 训练语料缺失不一定是绝对阻塞；把 architecture manuals、ISA、memory hierarchy 和 optimization rules 结构化进 RAG，配合真实编译/性能反馈，可以让 agent 在 inference time 获得硬件知识。
- 对 AI infra 团队，kernel knowledge base 应像代码仓库一样维护：每次成功优化都要沉淀 operator pattern、shape constraints、hardware-specific caveats 和 rejected attempts。
- KernelEvolve 与 CUDA Agent 代表两条互补路线：CUDA Agent 更偏训练出会写 kernel 的 model；KernelEvolve 更偏把 frontier/base model 包进可搜索、可诊断、可部署的 production harness。
- 生产部署必须把 correctness、performance、profiling、cost、CI、rollback 和 ownership 都放进系统边界；否则 agentic kernel generation 只能停留在 benchmark demo。

## 复现/阅读建议

- 先读 arXiv Abstract、Section 3 System Architecture、Section 4 OSS Operator Evaluation、Section 5 Monetization Case Study，再对照 Meta Engineering 文的六组件解释。
- 外部复现建议从公开 KernelBench + TritonBench 开始，构建最小 graph search harness：candidate generation、compile/evaluate/profile、node selection、memory operator、termination rule。
- 不要一开始支持多硬件。先在 NVIDIA + Triton 上做稳定 correctness/performance loop，再扩展 AMD/HIP 或其他 backend。
- Knowledge base 应分层维护：correctness constraints、platform-agnostic tuning rules、hardware-specific docs；每次检索都记录触发信号和使用证据，便于审计。
- 对生产化评估，必须报告 speedup distribution、shape coverage、compile failure rate、search budget、candidate count、profiling cost、fallback policy，而不是只报告最好样本。
