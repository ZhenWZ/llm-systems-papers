# Vision Generation & Understanding Systems

这一类关注 generative vision models 如何成为统一的 visual understanding backend。重点不是单个 segmentation 或 depth 模型，而是如何把多种视觉任务表达为统一的 generative interface，并在模型服务、prompt contract、output decoding、latency/cost 和下游消费之间形成可维护的系统边界。

## 当前论文

| 优先级 | 论文 | 阶段 | 核心问题 | 系统启发 | 笔记 |
| --- | --- | --- | --- | --- | --- |
| P1 | Image Generators are Generalist Vision Learners | multimodal vision | vision tasks 输出格式分裂，dense perception 难以用纯文本接口统一 | 把 segmentation、depth、surface normal 等任务输出参数化为 RGB image，用 image generator 作为统一 perception backend | [notes](../papers/2026-vision-banana-generalist-vision.md) |

## 阅读重点

- Image generation pretraining 是否能像 LLM pretraining 一样产生可迁移的 understanding capability。
- Dense vision tasks 如何被重新表达为 RGB image generation。
- Prompt contract、color mapping、output decoding 和 evaluator 如何成为系统接口的一部分。
- Generalist image generator 与 specialist perception model 在 latency、cost、精度、稳定性和可部署性上的 tradeoff。

## 后续可补充方向

- Painter / SegGPT / Perceptio / VisionFoundry 等 unified vision 或 generative perception 方向。
- Open-source image editing models 的 zero-shot vision learners 复现工作。
- Multimodal agents 中 segmentation/depth/normal map 作为中间状态的 grounding 与 planning 系统。
