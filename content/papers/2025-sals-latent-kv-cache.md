# SALS: Sparse Attention in Latent Space for KV Cache Compression

## Metadata

- Year: 2025
- Authors: Junlin Mu, Hantao Huang, Jihang Zhang, Minghui Yu, Tao Wang, Yidong Li
- Category: Long-Context Attention & KV Cache
- Priority: P1
- Links: [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2025/hash/00a0ebcad584c59dbc439c2af8793638-Abstract-Conference.html), [arXiv](https://arxiv.org/abs/2510.24273)
- Keywords: KV cache compression, latent space, low-rank projection, RoPE, sparse attention, decode throughput

## 一句话结论

SALS 把 KV cache 压缩到 low-rank latent space，在 latent space 中做 top-k token selection，只重建少量重要 tokens，再执行 sparse attention，从而同时降低 KV cache footprint、memory traffic 和 full reconstruction overhead。

## Related Works

- KV cache compression: Palu、Eigen Attention 等利用 KV cache 的 low-rank characteristics，但常在 accuracy 与 reconstruction cost 之间折中。
- KV cache quantization: KIVI 等通过低比特量化减少 footprint，但不直接减少 attention token count。
- Sparse attention / sparse decoding: Double Sparse、HShare、Loki、Quest、StreamingLLM 等通过 token/page selection 降低 decode bandwidth 和 compute。
- Training-based sparse attention: NSA 等通过训练获得 sparse module；SALS 是 post-training/calibration based。

## 问题与背景

长上下文 decode 阶段的主要瓶颈是 KV cache size 和 memory bandwidth。低秩压缩看起来适合 KV cache，但现代 LLM 普遍使用 RoPE。论文指出，RoPE 会增加 key vectors 的 variance 和 effective rank，导致 post-RoPE low-rank compression 精度差。

如果选择 pre-RoPE low-rank compression，可以保留较好 representation，但每次 attention 需要把完整 key cache 从 latent space reconstruct 回 full dimension 再应用 RoPE，reconstruction overhead 会变成新瓶颈。

## 方法与系统设计

SALS 分三阶段：

### Stage 1: KV cache compression

离线 calibration 得到 projection matrix，把 pre-RoPE keys/queries 投影到 shared latent space。Key cache 以低秩形式存储。Value 因为更接近 full rank，论文中采用 group quantization 而不是简单低秩投影。

### Stage 2: token selection in latent space

对于当前 query，SALS 在 latent space 计算 approximate query-key score，选择 top-k critical tokens。这个步骤避免了完整 reconstruct key cache。

### Stage 3: selective reconstruction and sparse attention

只对选中的 tokens reconstruct full-rank key/value，随后应用 RoPE 并执行 sparse attention。这样 full reconstruction 的 cost 从所有 tokens 降到 top-k subset。

## 创新点

- 明确分析 RoPE 对 key rank/variance 的影响，解释为什么 post-RoPE low-rank compression 会掉精度。
- 用 pre-RoPE latent representation 做 token importance estimation，绕开 full reconstruction。
- 把 low-rank KV cache compression 与 token sparse attention 结合，而不是二选一。
- 将 memory-bound speedup 建模为 `sr* + 2kr` 级别的数据移动，直接面向 decode bandwidth。

## 实验与结论

在 LLaMA2-7B-chat 和 Mistral-7B-v0.2 上，SALS 在 LongBench 中接近 baseline，同时显著降低 memory access：

- LLaMA2-7B-chat: baseline avg 32.65；SALS-25% avg 32.26，memory access 0.11；SALS-12.5% avg 31.97，memory access 0.06。
- Mistral-7B-v0.2: baseline avg 43.12；SALS-25% avg 42.79，memory access 0.11；SALS-12.5% avg 40.99，memory access 0.06。

RULER 上，Llama3.1-8B-Instruct:

- baseline avg 81.60。
- SALS-25% avg 80.81，基本接近 baseline。
- SALS-12.5% avg 75.86，在更强压缩下 retrieval-heavy tasks 出现明显下降。

Efficiency 方面：

- bs=8, 4K attention latency: FlashAttention 1.630ms，SALS-12.5% 0.439ms。
- end-to-end throughput: 32K sequence 下 GPT-Fast 19.8 tokens/s，SALS-12.5% 89.47 tokens/s。
- 论文总结为 6.4x KV cache compression、5.7x attention operator speedup，以及 4K/32K 上相对 GPT-Fast 的 1.4x/4.5x end-to-end improvement。

## 局限性与风险

- 论文中代码状态为 future public availability，工程复现不如 Stem 直接。
- 需要 offline calibration 得到 projection matrix，生产系统要考虑模型版本、任务分布和 calibration set drift。
- 12.5% compression 在 RULER 上已有明显 accuracy drop，尤其是强 retrieval dependency 任务。
- selective reconstruction 本身需要 fused kernel，否则 framework overhead 可能吞掉收益。
- Value cache 未完全低秩化，而是依赖 quantization，系统实现会更复杂。

## 对 LLM Systems 的启发

- KV cache compression 不能只看 memory footprint，还要把 reconstruction path 纳入 latency model。
- RoPE 会改变 low-rank structure；做 KV cache compression 时应区分 pre-RoPE 与 post-RoPE representation。
- decode-side optimization 可以采用“latent scoring + selective reconstruction + sparse attention”的三段式 pipeline。
- 对 serving 系统，SALS 适合作为 long-context decode backend 的候选模块，但落地前必须先验证 fused reconstruction kernel。

## 复现/阅读建议

- 先读 Section 3 对 RoPE 和 latent space 的分析，再读 Section 4 的 pipeline。
- 复现时先固定 sequence length 4K/8K，比较 baseline、KIVI、Palu、SALS-25%。
- 必须同时记录 accuracy、memory access、attention latency、end-to-end throughput。
