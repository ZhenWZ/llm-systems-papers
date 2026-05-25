# DFlash: An Efficient Speculative Decoding Framework using Diffusion Language Models

## Metadata

- Year: 2026
- Authors: Peixin Zhang, Jiawei Liu, Xu Guo, Xinhao Duan, Boyuan Pan, Jinbo Bi, Xipeng Qiu
- Category: Inference Serving & Speculative Decoding
- Priority: P0
- Links: [project](https://z-lab.ai/projects/dflash/), [arXiv](https://arxiv.org/abs/2602.06036), [code](https://github.com/z-lab/dflash), [models](https://huggingface.co/collections/z-lab/dflash)
- Keywords: speculative decoding, block diffusion, diffusion language model, drafter, verifier, SGLang, vLLM, inference decode

## 一句话结论

DFlash 把 diffusion language model 用作 speculative decoding 的 parallel drafter，并通过 block-wise generation、confidence-based remasking 和 joint SFT training 提高 acceptance length，在不改 target autoregressive LLM 输出分布的前提下降低 decode latency。

## Related Works

- Speculative decoding: Medusa、EAGLE、SpecInfer、Lookahead 等方法通过小模型或多头 drafter 生成候选 token，再由 target LLM 验证；DFlash 的差异在于 drafter 是 diffusion language model，可以并行生成多个 token。
- Diffusion language models: LLaDA、Dream 等工作证明 masked diffusion 可以用于 text generation，但直接 serving 仍受迭代去噪成本限制；DFlash 把它们放到 drafter 位置，避免替代主模型的质量风险。
- Block diffusion / parallel decoding: 通过一次预测多个位置降低 sequential dependency，但需要处理 token 之间的 intra-block 依赖；DFlash 用 block schedule 和 remasking 让 proposal 更适合 verifier 接受。
- LLM serving systems: SGLang、vLLM 提供 runtime integration 背景，DFlash 的价值不只在算法 acceptance rate，也在能否接进实际 serving path。

## 问题与背景

Autoregressive LLM decode 每步通常只产生一个 token，计算图高度串行。即使 prefill 和 attention kernel 已经高度优化，decode 仍会被 per-token target forward、KV cache 访问和调度开销限制。Speculative decoding 用一个更便宜的 drafter 提前提出多个 token，再由 target model 一次验证，从而把多个 sequential steps 合并。

传统 speculative decoding 的核心瓶颈是 proposal quality 和 parallelism 的折中：如果 drafter 太弱，acceptance length 低；如果 drafter 太重，drafting overhead 吃掉收益。DFlash 的目标是利用 diffusion model 的 parallel token generation，让 drafter 一次生成一个候选 block，并让这些候选更容易被 autoregressive target 接受。

## 方法与系统设计

DFlash 采用 standard speculative decoding 框架：draft model 先提出 token block，target autoregressive LLM 计算这些 token 的条件概率并执行 rejection sampling / verification。只要验证逻辑正确，最终采样分布仍与 target LLM 一致；系统收益来自一次 target forward 验证多个候选 token。

核心是 DLM drafter。论文把 draft 过程组织为 block-wise denoising：给定 prefix 后，drafter 在一个 block 内并行预测多个 masked token。相比逐 token autoregressive drafter，DLM 的 block proposal 可以暴露更多 parallelism，但也带来 block 内条件依赖不足的问题。

为提升 proposal quality，DFlash 引入 confidence-based remasking。drafter 会根据 token confidence 保留高置信 token、重新 mask 低置信 token，并多轮细化候选 block。这个机制在 draft 阶段用额外 denoising steps 换取更高 acceptance length，系统上需要平衡 denoising cost 与 verifier 节省的 target forward。

训练上，DFlash 使用 joint supervised fine-tuning，把 diffusion drafter 与 target LLM 的行为对齐。论文强调只有 diffusion pretraining 或 vanilla SFT 不足以获得稳定高 acceptance rate；需要针对 speculative decoding 的接受目标改造训练数据与目标。

工程实现上，项目提供 SGLang backend，并在 roadmap 中列出 vLLM backend。系统集成关注点包括 draft/verifier batching、KV cache 复用、block size、verification kernel path，以及多轮 remasking 对 serving scheduler 的影响。

## 创新点

- 把 diffusion language model 定位为 speculative decoding drafter，而不是替代 autoregressive target LLM。
- 用 block-wise parallel proposal 扩大每轮 decode 的候选 token 数，从系统角度直接攻击 sequential decoding bottleneck。
- 通过 confidence-based remasking 在并行生成和 proposal correctness 之间做动态折中。
- 设计面向 speculative decoding 的 joint SFT，使 drafter 的训练目标更贴近 verifier acceptance，而不只是普通 language modeling loss。
- 提供 project/code/models，说明方法已经开始面向实际 serving 框架落地。

## 实验与结论

论文在 MT-Bench、HumanEval、GSM8K、Alpaca 等任务上评估 DFlash。核心结论是 DFlash 在保持 target LLM 输出质量的同时显著提升推理速度；project page 报告 Qwen3-8B 上最高约 6.1x speedup，arXiv 摘要报告多类语言任务超过 6x speedup。

Acceptance length 是关键指标。论文报告在 greedy decoding 下平均 acceptance length 为 4.9，在 temperature = 1.0 sampling 下为 4.1，说明 DLM drafter 能让 verifier 在多数轮次接受多个 token。

消融实验显示，joint SFT 对 DFlash 的 acceptance length 和速度收益很重要。confidence-based remasking、block size 和 denoising steps 则决定了 draft overhead 与 proposal quality 的平衡。

需要注意的是，端到端速度高度依赖模型组合、batch size、sampling setting、prompt/output length、serving backend 和硬件配置。具体线上收益应以复现实验或本地 benchmark 为准。

## 局限性与风险

- DLM drafter 需要额外训练与部署，生产系统要承担模型管理、显存占用、版本匹配和 fallback 复杂度。
- Draft denoising steps 会引入额外 compute；当 target model 较小、batch 较大或 acceptance length 下降时，speedup 可能被抵消。
- 论文主要展示特定 model families 和 benchmark，跨模型、长输出、多轮对话、工具调用和严格格式生成场景仍需验证。
- Confidence-based remasking 的超参数与 block size 可能对 workload 敏感，serving 系统需要 runtime tuning。
- vLLM backend 在 project page 中标记为 roadmap，当前可复现路径应优先以已公开的 SGLang backend 为准。

## 对 LLM Systems 的启发

- Speculative decoding 的关键不是“drafter 越小越好”，而是 draft overhead、acceptance length 和 verifier batching 的系统平衡。
- Diffusion/parallel decoding 模型可以作为 serving-side accelerator，而不必直接替换主力 autoregressive LLM。
- Acceptance length 应与 end-to-end latency、吞吐、batching 行为和显存占用一起报告；只看 per-token accuracy 或 draft loss 不足以判断系统价值。
- 面向 serving 的训练目标可以直接优化系统指标，例如 verifier acceptance，而不是只追求 standalone generation quality。

## 复现/阅读建议

- 先读 Method 中 DLM drafter、confidence-based remasking 和 joint SFT，再看 speculative decoding correctness 相关部分。
- 复现时从官方 SGLang backend 和 Hugging Face models 开始，先固定 target/draft pair，再扫 block size、denoising steps、temperature 和 batch size。
- Benchmark 至少报告 acceptance length、tokens/s、端到端 latency、draft overhead、显存占用和输出质量；不同 sampling setting 要分开记录。
