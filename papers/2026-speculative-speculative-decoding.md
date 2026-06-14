# Speculative Speculative Decoding

## Metadata

- Year: 2026
- Authors: Tanishq Kumar, Tri Dao, Avner May
- Category: Inference Serving & Speculative Decoding
- Priority: P0
- Links: [arXiv](https://arxiv.org/abs/2603.03251), [code](https://github.com/tanishqkumar/ssd), [HF Papers](https://huggingface.co/papers/2603.03251)
- Keywords: speculative decoding, asynchronous drafting, Saguaro, speculation cache, verification outcome, fallback speculator, PagedAttention, CUDAgraphs, H100, inference decode

## 一句话结论

Speculative Speculative Decoding 继续拆解 ordinary speculative decoding 的串行依赖：当 target verifier 正在验证上一轮 draft 时，draft model 在独立 GPU 上预测可能的 verification outcomes，并提前为这些 outcomes 生成下一轮 draft；如果实际 outcome 命中 speculation cache，就可以立即进入下一轮 verification，从而隐藏 draft overhead。

## Related Works

- Speculative decoding: Leviathan et al. 和 Chen et al. 的方法用 draft model 提前生成多个 token，再由 target model 一次 forward 验证，保持 target distribution 不变；SSD 保留 correctness，但把 draft 与 verify 从串行改成异步并行。
- Multi-token / tree-based draft methods: Medusa、EAGLE、SpecInfer、token tree decoding 等主要提高每轮 proposal 数量或 acceptance rate；SSD 目标不同，关注 ordinary SD 内部 drafting 和 verification 之间的 pipeline bubble，并可与 tree-based 方法组合。
- LLM inference engines: vLLM、SGLang、PagedAttention、continuous batching、CUDAgraphs、FlashAttention/FlashInfer 构成 serving backend 背景；SSD 的贡献需要在真实 engine path 中衡量，而不只是算法级 acceptance。
- Existing repo relevance: DFlash 优化 drafter 类型，用 diffusion language model 做 parallel proposal；SSD 优化 draft/verifier 调度，用异步 speculation cache 隐藏 draft latency，两者都属于 serving-side speculative decoding。

## 问题与背景

Autoregressive decoding 的每个 token 都依赖前一个 token，decode 阶段难以充分利用硬件并行度。Speculative decoding 已经通过 draft model 先猜多个 token、target model 一次验证，减少 target model forward 次数，并用 rejection sampling 保证输出仍来自 target distribution。

但 ordinary speculative decoding 仍有一段关键串行依赖：verification 完成前，draft model 不知道这一轮接受了几个 tokens，也不知道 rejection 后 sampled bonus token 是什么，因此不能开始下一轮 draft。结果是 target verifier 和 draft model 轮流等待，尤其在 draft overhead 不能忽略时，会限制端到端 latency。

SSD 的核心问题是：能否消除 speculation 与 verification 之间的 sequential dependence。论文把这件事类比成 speculative execution：在分支结果出来前，先为多个可能结果预计算后续路径，命中时直接复用。

## 方法与系统设计

SSD framework 将 draft model 放到独立硬件上。每轮 target verifier 正在验证当前 draft 时，draft model 同步预测下一轮可能出现的 verification outcomes，并为这些 outcomes 预先生成候选 tokens 与 logits，构成 speculation cache。

Verification outcome 不只是 accepted length。对于长度为 `K` 的 proposal，outcome 包含接受了 `0..K` 个 tokens 以及 rejection 后的 bonus/recovery token。这个空间很大，无法全量缓存。论文提出 Saguaro，把 verification outcome prediction 变成受 fan-out / cache budget 约束的优化问题，优先覆盖最可能命中的 accepted-prefix 与 bonus-token 分支。

Saguaro cache 的 key 通常包括 sequence id、accepted-prefix length 和 recovery token。Cache hit 时，draft side 可以立即把下一轮 `K` 个 speculative tokens 与对应 logits 返回 target，target 继续验证；cache miss 时则走 fallback。

Saguaro sampling 解决另一个 tradeoff：ordinary SD 只关心 draft distribution 与 target distribution 的接近程度，以提高 acceptance rate；SSD 还关心 residual/bonus token 是否容易被 cache 预测。如果 sampling 让 bonus token 分布过于分散，cache hit rate 会下降。Saguaro sampling 在 acceptance rate 与 cache hit rate 之间做平衡。

Fallback strategy 是第三个关键点。小 batch 时，cache miss 可以用更准但更慢的 neural backup speculator；大 batch 或较高 temperature 时，任一 sequence miss 都可能拖慢整批请求，此时 fast backup speculator 可能更合适。论文强调 optimal fallback 随 batch size 改变。

实现上，官方 repo 提供自定义 inference engine，支持 Qwen3 与 Llama3 model families、tensor parallelism、PagedAttention、CUDAgraphs、torch compilation 和 prefix caching。论文附录说明 target model split across 4 GPUs，draft model 在单独 GPU 进程运行，二者每轮通过 NCCL 交换 outcome key、sequence length、block table、temperature、cache hit bitmap、speculative tokens 与 logits；不会在 target 和 draft 之间传输 KV cache。

## 创新点

- 提出 SSD framework，将 speculative decoding 内部的 speculation 与 verification 进一步并行化。
- 把 verification outcome prediction 明确建模为 cache construction 问题，而不是只优化 draft token accuracy。
- 提出 Saguaro cache / Saguaro sampling / Saguaro fallback 三个组件，分别对应 outcome prediction、acceptance-hit-rate tradeoff 和 cache miss handling。
- 在 serving implementation 中将 target 与 draft 分离到不同 GPU，并通过 NCCL 交换极少量控制信息与 logits，避免跨设备 KV cache 传输。
- 说明 SSD 可与 EAGLE-3、token-tree speculative decoding 等方法组合，属于对 SD runtime pipeline 的正交优化。

## 实验与结论

arXiv v3 摘要报告，Saguaro 平均比 optimized speculative decoding baselines 快约 30%，最高比 autoregressive decoding 快 5x。arXiv HTML v1 的正文/摘要表述为最高可比 optimized SD 快 2x。可核查的 appendix numerical results 给出更具体的端到端 decode throughput。

实验设置：Llama-3.1-70B target 搭配 Llama-3.2-1B draft，Qwen3-32B target 搭配 Qwen3-0.6B draft；数据集包括 HumanEval、Alpaca、GSM8K、UltraFeedback；每个数据集取 128 prompts，每个 prompt 生成 512 tokens；AR/SD baselines 使用 4x H100，SSD 使用 4x H100 target + 1x H100 draft。

Llama-3.1-70B 平均结果：AR 为 54.7 tok/s，SD 为 161.8 tok/s，SSD 为 255.8 tok/s；SSD/SD 为 1.58x，SSD/AR 为 4.68x。按数据集看，HumanEval、UltraFeedback、Alpaca、GSM8K 上 SSD/SD 分别约为 1.60x、1.55x、1.55x、1.60x。

Qwen3-32B 平均结果：AR 为 88.8 tok/s，SD 为 136.8 tok/s，SSD 为 203.8 tok/s；SSD/SD 为 1.49x，SSD/AR 为 2.29x。按数据集看，SSD/SD 约在 1.43x 到 1.54x 之间。

论文还报告 cache hit rate 随 fan-out 增加而提升，并讨论 batch size scaling：SSD 严格改善 throughput-latency Pareto frontier，但收益依赖 cache hit rate、fallback latency、batch size、temperature、draft compute budget 和硬件配置。

## 局限性与风险

- SSD 默认增加额外 draft GPU。论文的主实验使用 5 GPUs 对比 4 GPUs baselines，因此实际部署需要报告 cost-normalized latency/throughput，而不能只看 tok/s。
- SSD 用更多 draft-side FLOPs 换更低 latency。对于 offline batch generation、RL rollout 或高 batch throughput workloads，额外 fan-out compute 未必划算。
- Cache miss 和 fallback 会影响尾延迟。大 batch 下一个 sequence miss 可能拖住整批 decode，fallback policy 需要随 batch size、temperature、workload 调整。
- 实现复杂度高于 ordinary SD：需要异步 draft process、NCCL 通信、speculation cache、page table reconciliation、rollback rejected tokens、custom attention mask 和多 GPU 调度。
- Saguaro 的收益依赖 draft model、target model、sampling temperature、prompt distribution 和 output length；跨模型与线上混合 workload 仍需实测。

## 对 LLM Systems 的启发

- Speculative decoding 不只是 drafter 质量问题，也是 pipeline scheduling 问题。普通 SD 的 draft/verify bubble 可以用异步 speculation cache 继续压缩。
- Serving 系统可以把 draft model 当成独立 accelerator service：用额外便宜算力与显存换 target verifier 的空闲时间。
- 评价 speculative decoding 需要同时看 acceptance length、cache hit rate、fallback rate、tokens/s、TTFT、tail latency、额外 GPU 成本和 batch scaling。
- SSD 与 DFlash、EAGLE、Medusa、token-tree draft 是正交方向：前者优化异步 runtime pipeline，后者优化 proposal quality 或 proposal width。
- 对工程实现，真正难点在 scheduler/KV/page-table/communication 细节，而不仅是论文算法；官方 repo 值得作为 studying inference engine 的代码样本。

## 复现/阅读建议

- 先读 Section 3 的 SSD framework，再读 Section 4 的 Saguaro cache、sampling 和 fallback；最后读 Appendix B 的 systems design 与 Appendix B.3 numerical results。
- 复现优先使用官方 [tanishqkumar/ssd](https://github.com/tanishqkumar/ssd)，从 Llama 70B/1B 或 Qwen 32B/0.6B 的 greedy decode benchmark 开始。
- Benchmark 至少同时保留 AR、ordinary SD、SSD 三组，并明确 GPU 数量、batch size、temperature、lookahead `K`、fan-out `F`、output length、dataset 和是否包含 prefill。
- 如果目标是线上 serving，需要额外测 tail latency、cache miss fallback path、batch size sweep、mixed prompt lengths、temperature sampling 和多租户调度影响。
- 与 DFlash 对比阅读时，可把 DFlash 看作 better parallel drafter，把 SSD 看作 better draft/verifier pipeline；二者的系统瓶颈和复现风险不同。
