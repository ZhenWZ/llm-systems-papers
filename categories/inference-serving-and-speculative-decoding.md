# Inference Serving & Speculative Decoding

这一类关注 LLM inference serving 中的 decode latency、throughput、speculative decoding、batching、scheduler/runtime integration 和 verifier/drafter 协同。重点是如何在保持 target model 输出分布或质量的同时，减少串行 decode 步数和 serving overhead。

## 当前论文

| 优先级 | 论文 | 阶段 | 核心问题 | 系统启发 | 笔记 |
| --- | --- | --- | --- | --- | --- |
| P0 | DFlash: An Efficient Speculative Decoding Framework using Diffusion Language Models | decode | autoregressive decode 串行生成 token，传统 drafter 在 proposal quality 与 overhead 之间难平衡 | 用 diffusion drafter 并行生成候选 block，并通过 confidence-based remasking 与 joint SFT 提高 verifier acceptance length | [notes](../papers/2026-dflash-speculative-decoding.md) |
| P0 | Speculative Speculative Decoding | decode | ordinary speculative decoding 仍然有 draft/verify 串行依赖，draft model 在 verifier 运行时空闲 | 在独立 GPU 上预先预测 verification outcomes 并缓存下一轮 drafts，用 Saguaro cache/sampling/fallback 隐藏 draft overhead | [notes](../papers/2026-speculative-speculative-decoding.md) |

## 阅读重点

- Speculative decoding 的正确性依赖 verifier/rejection sampling，系统收益依赖 acceptance length 与 draft overhead 的平衡。
- Drafter 训练目标应贴近 verifier acceptance，而不只是 standalone language modeling loss。
- Draft/verifier pipeline 也可以继续优化：当 verifier 正在跑时，draft-side hardware 可以提前为多个可能 outcome 做 speculation。
- Serving backend 集成需要同时关注 KV cache、batching、scheduler、block size、sampling setting 和显存占用。

## 后续可补充方向

- EAGLE / EAGLE-2 / Medusa / SpecInfer / Lookahead decoding / SSD variants。
- SGLang、vLLM、TensorRT-LLM 中的 speculative decoding runtime support。
- Parallel decoding、block diffusion 和 diffusion language model serving。
