# LLM Systems Papers

面向 LLM systems 的论文库。当前先收录 Notion database 中的 3 篇论文，按系统层能力分类整理，并为每篇论文提供中文笔记。专业术语保留英文，便于后续检索和与原文对照。

## 分类导航

| 系统层分类 | 关注问题 | 论文 |
| --- | --- | --- |
| [Kernel & Operator Systems](categories/kernel-and-operator-systems.md) | GEMM epilogue fusion、算子重写、训练 kernel authoring | CODA |
| [Long-Context Attention & KV Cache](categories/long-context-attention-and-kv-cache.md) | sparse attention、prefill latency、KV cache compression、decode throughput | Stem, SALS |

## 阅读优先级

| 优先级 | 论文 | 分类 | 为什么先读 | 链接 | 笔记 |
| --- | --- | --- | --- | --- | --- |
| P0 | Stem: Rethinking Causal Information Flow in Sparse Attention | Long-Context Attention & KV Cache | 有明确 kernel 链接，直接面向 long-context prefill sparse attention，工程可复现性最高 | [arXiv](https://arxiv.org/abs/2603.06274), [Stem kernel](https://github.com/Tencent/AngelSlim/blob/main/angelslim/compressor/sparsity/stem/ops/stem_kernel.py), [Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention) | [notes](papers/2026-stem-sparse-attention.md) |
| P1 | CODA: Rewriting Transformer Blocks as GEMM-Epilogue Programs | Kernel & Operator Systems | 代表训练侧 kernel abstraction 方向，把 Transformer block 重写成 GEMM-plus-epilogue programs | [HF Papers](https://huggingface.co/papers/2605.19269), [arXiv](https://arxiv.org/abs/2605.19269), [code](https://github.com/HanGuo97/coda-kernels) | [notes](papers/2026-coda-gemm-epilogue.md) |
| P1 | SALS: Sparse Attention in Latent Space for KV Cache Compression | Long-Context Attention & KV Cache | 结合 low-rank KV cache compression 和 sparse token selection，适合跟踪 decode 阶段 memory bandwidth 优化 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2025/hash/00a0ebcad584c59dbc439c2af8793638-Abstract-Conference.html), [arXiv](https://arxiv.org/abs/2510.24273) | [notes](papers/2025-sals-latent-kv-cache.md) |

## 当前收录

| 年份 | 论文 | 主题标签 | 阶段 | 代码状态 |
| --- | --- | --- | --- | --- |
| 2026 | CODA: Rewriting Transformer Blocks as GEMM-Epilogue Programs | GEMM, epilogue fusion, CuTeDSL, training kernels | training | public repo |
| 2026 | Stem: Rethinking Causal Information Flow in Sparse Attention | sparse attention, prefill, block sparse kernel, long context | inference prefill | linked implementation |
| 2025 | SALS: Sparse Attention in Latent Space for KV Cache Compression | KV cache compression, low-rank projection, sparse attention, RoPE | inference decode | paper says code forthcoming |

## 笔记格式

每篇论文笔记统一包含：

- Metadata
- 一句话结论
- Related Works
- 问题与背景
- 方法与系统设计
- 创新点
- 实验与结论
- 局限性与风险
- 对 LLM systems 的启发
- 复现/阅读建议

## 维护约定

- 新论文优先放到最贴近的系统层分类；如果横跨多个方向，在 README 中只放主分类，在笔记中列出交叉分类。
- 阅读优先级使用 `P0/P1/P2`：`P0` 表示近期最值得复现或精读，`P1` 表示重要方向，`P2` 表示跟踪或背景阅读。
- 中文为主，保留英文术语，例如 `prefill`, `decode`, `KV cache`, `GEMM`, `epilogue`, `sparse attention`。
