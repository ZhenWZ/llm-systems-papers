# Long-Context Attention & KV Cache

这一类关注长上下文推理中的 attention complexity、prefill latency、decode bandwidth 和 KV cache footprint。重点是如何在保持 accuracy 的同时降低 attention 计算与 memory traffic。

## 当前论文

| 优先级 | 论文 | 阶段 | 核心问题 | 系统启发 | 笔记 |
| --- | --- | --- | --- | --- | --- |
| P0 | Stem: Rethinking Causal Information Flow in Sparse Attention | prefill | uniform top-k sparse attention 忽略 causal information flow，容易剪掉早期关键 token | sparse budget 应随 token position decay，selection metric 应引入 value/output awareness | [notes](../papers/2026-stem-sparse-attention.md) |
| P0 | SparDA: Sparse Decoupled Attention for Efficient Long-Context LLM Inference | decode | KV cache offload 的 PCIe 传输无法与计算 overlap，且 sparse selection 绑定 query、O(T²) 开销大 | 用 Forecast projection 把 selection 与 query 解耦并提前一层，overlap CPU→GPU prefetch；GQA 下每组一个 Forecast head 降 selection 成本 | [notes](../papers/2026-sparda-decoupled-sparse-attention.md) |
| P0 | VeriCache: Turning Lossy KV Cache into Lossless LLM Inference | decode | KV cache compression 提升吞吐但输出会随 decode 累积偏离 full-KV path，影响 code/tool calling 等功能正确性 | 用 compressed KV draft、full KV verify，把 lossy compression 包装成 lossless serving path，并用 BW/HBM ring 调度 full-KV reload | [notes](../papers/2026-vericache-lossless-kv-cache.md) |
| P1 | SALS: Sparse Attention in Latent Space for KV Cache Compression | decode | KV cache 低秩压缩受 RoPE 和 full reconstruction overhead 制约 | 在 latent space 做 token selection，只重建重要 token，把 low-rank compression 与 sparse attention 结合 | [notes](../papers/2025-sals-latent-kv-cache.md) |

## 阅读重点

- prefill 和 decode 的瓶颈不同：prefill 更偏 attention compute，decode 更偏 KV cache memory bandwidth。
- sparse attention 的关键不是只降 sparsity，而是 token selection 是否保留 task-relevant context。
- KV cache compression 需要同时考虑 footprint、reconstruction cost、RoPE 影响和 fused kernel 实现。
- Lossy KV compression 在长输出、code generation、tool calling 中可能累积 functional correctness 风险；可用 full-KV verification 把近似路径变成 draft path。

## 后续可补充方向

- MInference / FlexPrefill / XAttention / Quest / H2O / StreamingLLM。
- KV cache quantization: KIVI 等。
- Lossless KV compression serving、remote prefix caching verification、compressed-KV drafting。
- Training-based sparse attention 与 post-training sparse attention。
