# Kernel & Operator Systems

这一类关注 LLM 训练/推理中的底层算子、kernel abstraction、memory movement 和 operator fusion。重点不是单个 model trick，而是如何把 Transformer 计算映射到更高效、更可维护的系统实现。

## 当前论文

| 优先级 | 论文 | 核心问题 | 系统启发 | 笔记 |
| --- | --- | --- | --- | --- |
| P1 | CODA: Rewriting Transformer Blocks as GEMM-Epilogue Programs | Transformer 中大量 non-GEMM memory-bound operators 导致额外 global memory round trip | 把 RMSNorm、residual、SwiGLU、RoPE、cross-entropy 等重写为 GEMM epilogue programs，形成可组合 kernel abstraction | [notes](../papers/2026-coda-gemm-epilogue.md) |

## 阅读重点

- GEMM mainloop 与 epilogue 的职责边界。
- 哪些 Transformer operations 能安全地 tile-local 化，哪些必须依赖 auxiliary reduction。
- 这种 abstraction 是否能降低手写 CUDA/CuTeDSL 的复杂度。
- 与 `torch.compile`, Triton, CUTLASS/CuTeDSL, Liger Kernels, FlashInfer 的关系。

## 后续可补充方向

- GPU kernel DSL 与 LLM-assisted kernel generation。
- Transformer training fused kernels。
- Compiler-level graph rewrite 与 hand-written kernel 的边界。
