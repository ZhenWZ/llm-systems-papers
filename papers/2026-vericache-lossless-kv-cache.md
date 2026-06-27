# VeriCache: Turning Lossy KV Cache into Lossless LLM Inference

## Metadata

- Year: 2026
- Authors: Jiayi Yao, Samuel Shen, Kuntai Du, Shaoting Feng, Dongjoo Seo, Rui Zhang, Yuyang Huang, Yuhan Liu, Shan Lu, Junchen Jiang
- Category: Long-Context Attention & KV Cache
- Priority: P0
- Links: [arXiv](https://arxiv.org/abs/2605.17613), [HF Papers](https://huggingface.co/papers/2605.17613)
- Keywords: KV cache compression, lossless inference, speculative verification, full-KV verification, remote prefix caching, vLLM, LMCache, token dropping, quantization, decode throughput

## 一句话结论

VeriCache 把 lossy KV cache compression 变成 lossless serving framework：用 compressed KV cache 高速 draft token，再周期性把 full KV cache 从 CPU/storage reload 到 GPU 做 verification，从而在输出与 full-KV decoding 保持一致的同时，保留大部分 KV compression 的吞吐收益。

## Related Works

- KV cache compression: KIVI、KVQuant、KVzip、SnapKV、DuoAttention、StreamingLLM 等通过 quantization 或 token dropping 降低 KV footprint 与 memory bandwidth，但通常会改变 decode distribution；VeriCache 的目标是把这些 lossy compressors 放进 draft-then-verify 框架中恢复 full-KV output。
- Speculative decoding: 普通 speculative decoding 用小模型 draft、target model verify；VeriCache 的 drafter 不是小模型，而是同一个 target model 在 compressed KV 上 decode，因此 acceptance length 通常更长。
- Sparse/compressed-KV self speculation: MagicDec、QuantSpec、SparseSpec 等也从 sparse/compressed KV draft，但通常把 full KV 留在 GPU HBM、面向单一 compressor，且不覆盖 remote prefix caching；VeriCache 重点解决 full KV 不常驻 GPU 时的 reload scheduling。
- KV cache serving systems: vLLM 与 LMCache 提供 serving engine、persistent KV storage 和跨层级 KV movement 背景；VeriCache 构建在这类系统之上，把 compression、reload、verification 和 admission control 接入 runtime。

## 问题与背景

长上下文 LLM serving 的瓶颈不只在 prefill。Decode 阶段每一步都要从 HBM 读取 KV cache，context 越长，单 request 的 KV footprint 与 bandwidth cost 越高；同时，KV cache 越大，GPU 上能同时容纳的 batch 越小。跨请求 prefix reuse 也会把瓶颈转移到 KV cache 从 CPU/storage/remote store 传到 serving GPU 的带宽和延迟上。

现有 KV compression 方法可以显著降低 footprint 与 transfer size，但大多是 lossy。短输出时语义质量可能变化不明显；长输出、code generation、tool calling 和 structured output 中，per-token distribution shift 会逐步累积，最终造成格式、函数参数、代码语法或任务正确性失败。

VeriCache 要解决的是 accuracy-throughput dichotomy：full KV 保证输出正确但吞吐低，compressed KV 吞吐高但输出会偏离。论文的问题定义很明确：能否利用 KV compression 的吞吐优势，同时保持与 full-KV decoding 相同的输出。

## 方法与系统设计

VeriCache 使用 compressed KV cache 作为 drafter。请求 decode 时，GPU HBM 常驻 compressed KV，模型先基于 compressed KV 连续 draft 多个 tokens。随后 runtime reload 对应 full KV cache 到 GPU，用 full KV 对 draft tokens 做 verification；被接受的 token 保留，被拒绝时回退到 full-KV 正确路径。由于 verification 使用 full KV，最终输出与 full-KV decoding 保持一致。

关键系统挑战是 full KV 不常驻 GPU。VeriCache 需要把 full KV 放在 CPU memory、storage 或远端 KV store 中，只在 verification window 前异步加载到 GPU。论文的两个主要 insight 是：compressed-KV decoding 通常是 HBM-bandwidth-bound，而 full-KV reload 是 PCIe/network-bound，可以跨资源 stagger；并且 compressed KV 与 full KV 往往输出相近，所以每轮 verification 可以接受较长 draft，从而 amortize reload 和 verification cost。

Runtime 维护两个资源 ring：BW ring 跟踪 interconnect / storage transfer window，HBM ring 跟踪 full KV reload 临时占用的 GPU memory。每个 request admission 或 verification 后，scheduler 会寻找未来可行的 verify iteration，提前预留 full-KV transfer 和 HBM capacity。若资源不足，请求进入 waiting queue。

VeriCache 支持两个主要场景。Pipeline 1 是 long-context decoding：context KV 预计算后保存在 CPU/storage，compressed KV 常驻 GPU 用于 draft，full KV 按 verification 需要 reload。Pipeline 2 是 remote prefix caching：远端实例更快读取 compressed KV 并 draft，本地 GPU 在更快链路上加载/缓存 full KV 做 verification。

论文还定义 uniform compressor interface，覆盖 token dropping 与 quantization。实现中基于 vLLM AsyncScheduler 管理 compressed/transient reload KV allocation，基于 LMCache 做 KV lookup、compressed lookup 和跨层级 movement；online compressors 通过 forward hook 更新 per-layer attention activations，offline compressors 可在 serving 前或 idle compute 上执行。

## 创新点

- 把 lossy KV compression 转化为 lossless inference：compressed KV 只负责 draft，full KV 负责最终 verification。
- 将 full KV 从 GPU HBM 移出，只在 verification 前 reload，释放 HBM 给 compressed-KV batching，而不是像部分 self-speculative 方法一样把 full KV 固定驻留在 GPU。
- 提出 cross-resource staggering，用 HBM-bound compressed decode overlap PCIe/network-bound full-KV reload。
- 利用 compressed-KV drafter 与 full-KV target 共享模型权重和主要 attention pattern 的特点，获得比小模型 speculative decoding 更长的 accepted run。
- 提供 compressor interface，统一接入 token dropping 与 quantization 方法，避免为每个 compressor 重写 runtime。

## 实验与结论

arXiv 摘要报告，VeriCache 在保持 identical outputs 的同时，相比 full-KV inference 最高可达 4x throughput。论文评估覆盖 long-context decoding 与 remote prefix caching 两条 pipeline。

实验模型与硬件包括 Mistral-24B、Qwen-32B 和 Llama-70B；Mistral/Qwen 使用 NVIDIA RTX PRO 6000，Llama-70B 使用 2H100 NVL。Compression methods 包括 KVzip、KVZap、ExpectedAttention、SnapKV、KIVI、KVQuant、RotateKV。Baselines 包括 Full KV、traditional speculative decoding 的 EAGLE3，以及 self-speculative SparseSpec。

质量评测包括 LMCache-trace 的 KL divergence、ComplexFuncBench 的 exact function-call argument matching、PISanitizer 的 prompt-injection defense success rate、LongGenBench 的 completion rate 和 GSM8K-Long 的 exact answer accuracy。效率指标是 end-to-end latency 与 throughput。

论文给出的关键现象包括：compressed-KV drafter 的 acceptance run 明显长于小模型 drafter，VeriCache 在不同模型和 compression methods 上保持接近 full-KV 的输出分布，并能在 function calling、long generation 和 math workloads 中避免 lossy KV 的功能性失败。与 EAGLE/SparseSpec 的对比表明，VeriCache 主要通过缩小每个 request 的 KV footprint 提升 batch/throughput，而 traditional drafter 主要减少每个 request 内的 decode steps；两者可以组合。

需要注意的是，HTML 版部分 figure 数字渲染缺失。除摘要中的最高 4x 外，笔记不复述无法从文本直接核对的具体 speedup 数字；复现时应以 PDF 图表或原始实验为准。

## 局限性与风险

- 需要额外保存 full KV。VeriCache 将 compressed KV 放在 GPU，同时在 CPU/storage 保留 full KV，因此总存储占用高于单纯 compression。
- 依赖高 acceptance length。若 compressor 让 compressed-KV output 与 full-KV output 分歧明显，verification 太频繁会吃掉 throughput gain。
- 当前 draft length 是 workload-level static policy；异质 prompt/context/compressor 下，需要 per-request adaptive policy。
- Full-KV reload 依赖 PCIe/network/storage 带宽与调度窗口。链路拥塞、remote prefix store 抖动或多租户负载会直接影响 tail latency。
- 论文没有在 HF/arXiv 页面给出官方代码仓库；复现需要自行基于 vLLM/LMCache 实现 scheduler、compressor interface 和 verification path。

## 对 LLM Systems 的启发

- KV compression 的正确性不能只用短输出 perplexity 或平均任务分衡量；code、tool calling 和 structured output 需要关注 distribution drift 的累积效应。
- Lossy optimization 可以作为 draft path，只要最终 commit path 有 full-fidelity verifier。这个设计模式可推广到其他近似 serving 技术。
- 长上下文 serving 的关键资源不止 GPU compute，还包括 HBM capacity、HBM bandwidth、PCIe/network transfer 和 remote KV store latency；runtime 需要显式建模这些资源。
- 与 SparDA/SALS 的关系：SparDA 解决 KV offload prefetch 与 sparse selection，SALS 在 latent space 降低 KV traffic，VeriCache 则用 compressed-KV draft + full-KV verification 把 lossy compression 包装成 lossless path。
- 对生产系统，VeriCache 的最大价值在于把 quality risk 从 compressor 质量转移到 verification scheduler 和 resource model 上，风险更可测、更可控。

## 复现/阅读建议

- 先读 Section 3 的 lossy KV failure motivation，确认为什么 semantic similarity 不能保证 functional correctness。
- 再读 Section 4/5，重点理解 compressed-KV draft、full-KV reload/verification、BW ring、HBM ring 和 request admission。
- 读 Section 6 时关注 compressor interface：token dropping 与 quantization 如何被统一为 deployment-time substitution。
- 复现实验时至少保留 Full KV、lossy KV、VeriCache 三组，并分别报告 exact output match、KL divergence、functional accuracy、throughput、latency 和 full-KV reload overhead。
- 若要工程落地，优先从单 compressor、单 pipeline、static draft length 开始，再扩展到 remote prefix caching、adaptive draft length 和 multi-tenant scheduling。
