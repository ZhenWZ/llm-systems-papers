# CODA: Rewriting Transformer Blocks as GEMM-Epilogue Programs

## Metadata

- Year: 2026
- Authors: Han Guo, Jack Zhang, Arjun Menon, Driss Guessous, Vijay Thakkar, Yoon Kim, Tri Dao
- Category: Kernel & Operator Systems
- Priority: P1
- Links: [HF Papers](https://huggingface.co/papers/2605.19269), [arXiv](https://arxiv.org/abs/2605.19269), [code](https://github.com/HanGuo97/coda-kernels)
- Keywords: GEMM, epilogue fusion, CuTeDSL, Transformer training, memory-bound operators, LLM-assisted kernel authoring

## 一句话结论

CODA 把 Transformer block 中大量 memory-bound operations 重写成 `GEMM + epilogue` programs，在保留高性能 GEMM mainloop 的同时，把 RMSNorm、residual、SwiGLU、RoPE、cross-entropy 以及 backward 中的局部计算融合到 epilogue 中。

## Related Works

- High-level frameworks: PyTorch/JAX 提供 autograd 和 operator graph，但 operator boundary 往往也是 materialization boundary。
- Compiler systems: `torch.compile`、Triton、TileLang、CuTeDSL 等尝试从图重写、scheduling、code generation 角度优化。
- Kernel libraries: CUTLASS/CuTeDSL 的 GEMM mainloop/epilogue 分离，以及 Epilogue Visitor Trees 为 CODA 提供了直接基础。
- LLM kernels: Liger Kernels、FlashInfer、vLLM、SGLang 等提供特定高性能 kernel，但扩展到新 fusion 或 backward 仍需要大量低层工程。

## 问题与背景

Transformer 训练的 FLOPs 主要来自 GEMM 和 attention，但 end-to-end runtime 中仍有相当比例花在 normalization、activation、residual update、reduction 等 memory-bound operators 上。随着 FP8/FP4 等低精度 GEMM 越来越快，global memory round trip 的相对成本会更突出。

现有框架把 Transformer 写成 operator sequence，易用但会强制 materialize intermediate tensors。生产系统常用 custom kernels 绕开框架边界，但这会牺牲可编程性和可维护性。CODA 要解决的问题是：能否用受限但表达力足够的 abstraction，接近 custom kernels 的性能，同时让人和 LLM 都更容易写 kernel。

## 方法与系统设计

CODA 的核心观察是：很多 Transformer operations 可以 algebraically reparameterize 成 GEMM epilogue work。GEMM mainloop 产生 output tile 后，epilogue 在 tile 仍在 on-chip memory/register/shared memory 生命周期内完成局部变换，避免写回 global memory 后再读出。

CODA 暴露五类 epilogue primitives：

- elementwise/pairwise maps：residual、activation、RoPE、SwiGLU 等。
- vector load/store：加载 RMSNorm weight 等 row/column vector 并 broadcast。
- tile load/store：处理 residual stream、saved activation、backward intermediate。
- tile reductions：产生 partial row/column reductions，再由 lightweight auxiliary kernel 汇总。
- stateful transforms：例如 online log-sum-exp / cross-entropy 的 running state。

关键 reparameterization 包括：

- `GEMM -> residual -> RMSNorm -> GEMM`：把 residual 和 RMSNorm weight scaling 放进第一个 epilogue，partial RMS 用 auxiliary reduction 汇总，row-wise scale 延迟到第二个 GEMM epilogue。
- `GEMM + SwiGLU/RoPE`：把 pairwise activation 或 rotary transform 放到 GEMM output tile 上处理。
- `GEMM + cross-entropy`：在 epilogue 中做 target logit selection 和 partial log-sum-exp。
- backward pass：局部 backward rule 仍然可表达为 GEMM-with-epilogue，非局部 reduction 用 partial reductions 处理。

## 创新点

- 把 epilogue 从简单 bias/scale post-processing 提升成 Transformer-specific programmable interface。
- 用一组受限 primitives 覆盖标准 Transformer 中大部分 non-attention、non-embedding forward/backward computation。
- 通过 algebraic reparameterization 跨越传统 module boundary，例如把 RMSNorm scale 延迟到后续 GEMM epilogue。
- 把 LLM-assisted kernel authoring 从“生成任意 CUDA”约束为“组合已有 epilogue primitives”，降低搜索空间和出错面。

## 实验与结论

论文在单 H100 上评估 CODA kernels，比较对象包括 cuBLAS with `torch.compile`、Liger Kernels、FlashInfer，以及 raw GEMM ceilings。实验覆盖 hidden size 2048/4096/8192，对应约 1B/7B/70B scale。

结论层面，CODA 在 kernel-level 和 block-level benchmarks 上都能接近 raw GEMM ceiling，并在多种 reparameterized Transformer sequences 中优于或接近优化库组合。更重要的是，LLM-authored CODA kernels 与 human-authored kernels 都能达到较高性能，说明 abstraction 本身对 authoring 有帮助。

## 局限性与风险

- 目前主要针对常见 Transformer architecture，泛化到 MoE、non-standard normalization、特殊 attention variants 需要额外工作。
- 论文聚焦 single-GPU kernels，尚未解决 distributed execution 下的通信/计算重排。
- reparameterization 会模糊 framework module boundary，使调试、autograd 集成、semantic mapping 变难。
- 对 CuTeDSL/CUTLASS 生态依赖较强，工程门槛仍不低。

## 对 LLM Systems 的启发

- 对训练系统而言，下一阶段瓶颈不只是 GEMM/attention 本身，而是围绕 GEMM 的 memory-bound glue operations。
- “受限 DSL + 专家 mainloop + 可组合 epilogue primitives”可能比通用 kernel generation 更适合 LLM-assisted systems programming。
- 论文提供了一个判断 fusion 价值的框架：只要 operation 能 tile-local 或 partial-reduction 化，就应考虑放进 GEMM epilogue 生命周期。

## 复现/阅读建议

- 先读 Section 3.2 的 reparameterization，再读 Appendix C 的 kernel list。
- 对照代码仓库看 epilogue primitive 如何组合，而不是从 benchmark 图开始。
- 如果做实践，建议先复现 `GEMM-RMSNorm-GEMM` 和 `GEMM-SwiGLU` 两类，再扩到 backward。
