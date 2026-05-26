# Stem: Rethinking Causal Information Flow in Sparse Attention

## Metadata

- Year: 2026
- Authors: Lin Niu, Xin Luo, Linchuan Xie, Yifu Sun, Guanghua Yu, Jianchen Zhu, S. Kevin Zhou
- Category: Long-Context Attention & KV Cache
- Priority: P0
- Links: [arXiv](https://arxiv.org/abs/2603.06274), [Stem kernel](https://github.com/Tencent/AngelSlim/blob/main/angelslim/compressor/sparsity/stem/ops/stem_kernel.py), [Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention)
- Keywords: sparse attention, causal information flow, prefill, long context, token position decay, output-aware metric

## 一句话结论

Stem 认为 causal attention 中早期 token 是跨层 information flow 的 “stem”，因此 sparse attention 不应对所有位置使用 uniform top-k，而应给早期 token 更高预算，并用 output-aware metric 保留真正影响 attention output 的 token。

## Related Works

- Training-free sparse attention: MInference、FlexPrefill、XAttention 等通过 pattern selection 或 block pruning 降低 prefill cost。
- Block sparse kernels: 论文实现依赖 Block-Sparse-Attention，说明方法本身更接近可落地 kernel strategy，而不只是算法指标。
- Training-based sparse models: DeepSeek-V3.2 的 DSA、MiniCPM-4.1 的 InfLLMv2 已经训练 sparse selection；Stem 作为 post-training module 还能进一步压缩 budget。
- Long-context benchmarks: LongBench 用于真实长上下文任务，RULER 用于可控 retrieval stress testing。

## 问题与背景

长上下文 prefill 阶段需要并行处理完整 prompt，standard self-attention 的 quadratic complexity 会导致 latency 和 memory overhead 激增。现有 sparse attention 通常按 attention scores 做 uniform top-k selection，但这忽略了 causal architecture 的不对称性。

在 causal attention 中，第一个 value vector 会参与后续所有位置的 aggregation，而末尾 token 只影响局部后续位置。跨层之后，这种依赖会被继续放大。剪掉早期 token 会造成 global distortion，剪掉尾部 token 往往只是 local error。

## 方法与系统设计

Stem 包含两个核心组件。

### Token Position-Decay (TPD)

TPD 根据 token position 动态调整 sparse budget。早期 query/block 分配更高 budget，后期 token 更激进 sparsify。直觉是保护 recursive dependency chain，避免早期 token 被剪掉后错误跨层传播。

### Output-Aware Metric (OAM)

传统 sparse selection 主要看 routing/attention score。Stem 进一步引入 value magnitude，近似衡量 token 对最终 output magnitude 的影响。其 metric 可以概括为 routing score 加上 value magnitude term，避免保留“注意力分高但 value 信息弱”的 token。

### Kernel execution

算法先对 Q/K 做 block pooling，计算 coarse routing metric；对 value magnitude 做 block-level max pooling；随后按 position-decay schedule 选择 top-k blocks，再 gather sparse K/V 做 exact sparse attention。论文用 Block-Sparse-Attention 执行 sparse computation。

## 创新点

- 从 causal information flow 而不是单层 attention score 解释 sparse attention 的 token importance。
- 提出 position-dependent budget，使 sparse budget 成为 position schedule，而不是全局常数。
- OAM 把 value 信息引入 token selection，使 selection 更接近 output error minimization。
- 方法 training-free，同时还能叠加到 training-based sparse models 上继续压缩 budget。

## 实验与结论

LongBench 上，Stem 在低 budget 下接近 dense：

- Qwen3-8B: dense 32.01，Stem 31.64，budget 25%。
- Llama-3.1-8B-Instruct: dense 42.02，Stem 41.48，budget 31%。

RULER 上，Stem 也保持 near-lossless：

- Llama-3.1-8B-Instruct: dense 88.86，Stem 88.47，budget 25%。
- Qwen3-8B: dense 87.66，Stem 87.15，budget 25%。

与 training-based sparse models 结合后：

- DeepSeek-V3.2 + Stem 在平均 budget 降低约 15% 后，LongBench 平均分略升。
- MiniCPM-4.1 + Stem 在 budget 降低约 18% 后，平均分基本不变。

Latency 方面，在 Llama-3.1-8B-Instruct、128K context 上，Stem 将 dense latency 从 1540ms 降到 420ms，约 3.7x speedup。

## 局限性与风险

- OAM 的 `beta` 和 TPD 的 decay ratio 需要调参；论文默认 `beta=0.2`, `mu=0.7`，但不同模型/任务可能不同。
- 对 block size、initial/local windows、minimum budget 的工程选择较敏感。
- 主要解决 prefill；decode 阶段 KV cache bandwidth 不是 Stem 的核心目标。
- 对早期 token 的高预算是一种结构性 bias，在某些非自然语言或特殊 prompt 分布下可能不是最优。

## 对 LLM Systems 的启发

- sparse attention 的核心不是“越稀疏越好”，而是 budget allocation 和 selection metric 是否符合 causal dependency。
- prefill sparse kernel 可以把 algorithm insight 和 block sparse kernel implementation 紧密结合，避免只停留在 Python-level pruning。
- 对长上下文系统，应该分别设计 prefill 和 decode 优化：Stem 更像 prefill-side scheduling/selection 策略。

## 复现/阅读建议

- 优先读 theoretical information flow 和 Algorithm 1，再看 LongBench/RULER 表格。
- 复现时先固定 Block-Sparse-Attention 后端，只替换 selection schedule 和 metric。
- 对比 baseline 时要同时报告 accuracy、average budget、kernel latency 和 total latency。
