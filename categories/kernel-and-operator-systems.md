# Kernel & Operator Systems

这一类关注 LLM 训练/推理中的底层算子、kernel abstraction、memory movement 和 operator fusion。重点不是单个 model trick，而是如何把 Transformer 计算映射到更高效、更可维护的系统实现。

## 当前论文

| 优先级 | 论文 | 核心问题 | 系统启发 | 笔记 |
| --- | --- | --- | --- | --- |
| P0 | CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation | 通用 LLM 难以稳定生成比 `torch.compile` 更快且正确的 CUDA kernels，test-time refinement 又受限于 base model 能力 | 用 6K synthesized ops、skill-augmented CUDA environment、non-hackable verification/profiling reward 和 staged agentic RL 训练 kernel optimization agent | [notes](../papers/2026-cuda-agent-kernel-generation.md) |
| P0 | KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta | Meta production ranking/ads workloads 的 kernel 数量随 hardware × model × operator 组合爆炸，人工 tuning 无法覆盖 NVIDIA/AMD/MTIA/CPU | 把 kernel authoring 变成 retrieval-augmented graph search，用 profiling diagnostics、persistent knowledge base 和 production validation 生成跨硬件 kernels | [notes](../papers/2026-kernelevolve-agentic-kernel-coding.md) |
| P1 | CODA: Rewriting Transformer Blocks as GEMM-Epilogue Programs | Transformer 中大量 non-GEMM memory-bound operators 导致额外 global memory round trip | 把 RMSNorm、residual、SwiGLU、RoPE、cross-entropy 等重写为 GEMM epilogue programs，形成可组合 kernel abstraction | [notes](../papers/2026-coda-gemm-epilogue.md) |

## 阅读重点

- GEMM mainloop 与 epilogue 的职责边界。
- 哪些 Transformer operations 能安全地 tile-local 化，哪些必须依赖 auxiliary reduction。
- 这种 abstraction 是否能降低手写 CUDA/CuTeDSL 的复杂度。
- 与 `torch.compile`, Triton, CUTLASS/CuTeDSL, Liger Kernels, FlashInfer 的关系。
- Agentic kernel generation 中，profile/compile/verify loop、reward shaping 和 anti-hacking sandbox 如何影响真实性能。
- `SKILL.md` / agent harness 与 CUDA kernel authoring abstraction 的边界。
- Heterogeneous accelerator support 中，RAG knowledge base、profiling diagnostics 和 persistent search state 如何弥补 compiler/library 覆盖不足。

## 后续可补充方向

- GPU kernel DSL 与 LLM-assisted kernel generation。
- Transformer training fused kernels。
- Compiler-level graph rewrite 与 hand-written kernel 的边界。
- Agentic RL for systems programming、KernelBench、Triton/TVM/Ascend C 后端迁移。
- Production kernel search harness：FaaS evaluation、multi-hardware CI、rollback、shape coverage、cost accounting。
