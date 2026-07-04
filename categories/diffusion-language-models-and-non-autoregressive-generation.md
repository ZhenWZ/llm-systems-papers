# Diffusion Language Models & Non-Autoregressive Generation

这一类关注 diffusion / flow-based language models、continuous vs. discrete generation、non-autoregressive decoding、sampling-step efficiency、quality-diversity trade-off，以及这些模型形态对 inference serving runtime 的影响。

## 当前论文

| 优先级 | 论文 | 阶段 | 核心问题 | 系统启发 | 笔记 |
| --- | --- | --- | --- | --- | --- |
| P1 | ELF: Embedded Language Flows | diffusion generation | continuous DLM 长期落后于 discrete DLM，关键瓶颈可能不是语言离散性本身，而是 embedding representation、decoder interface、sampler 和 guidance 设计不足 | 把生成过程保持在 continuous embedding space，用 Flow Matching、shared-weight final decoder、CFG 和 SDE sampler 改善少步生成质量；serving 需要从 AR token decode 转向 step-wise full-sequence denoising | [notes](../papers/2026-elf-embedded-language-flows.md) |

## 阅读重点

- Continuous embedding space 与 discrete token space 的建模差异，以及 final discretization 是否成为质量瓶颈。
- Sampling steps、sequence length、batch size、CFG scale、sampler 选择对 latency/quality/diversity 的共同影响。
- Non-autoregressive generation 是否真的带来系统收益，必须用 wall-clock latency、throughput、memory 和 hardware utilization 验证。
- 与 DFlash / speculative decoding 的关系：diffusion LM 可以作为 block proposal drafter，但 serving 收益取决于 proposal quality 与 verifier acceptance。

## 后续可补充方向

- MDLM、Duo、E2D2、LLaDA、Dream 等 discrete diffusion language models。
- FLM/FMLM、LangFlow、BitstreamDiffusion、Cola-DLM 等 continuous / flow-based language models。
- Diffusion LM serving runtime：step batching、sampler scheduling、early exit、distillation、quality/diversity monitoring。
