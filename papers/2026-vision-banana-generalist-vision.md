# Image Generators are Generalist Vision Learners

## Metadata

- Year: 2026
- Authors: Valentin Gabeur, Shangbang Long, Songyou Peng, Paul Voigtlaender, Shuyang Sun, Yanan Bao, Karen Truong, Zhicheng Wang, Wenlei Zhou, Jonathan T. Barron, Kyle Genova, Nithish Kannen, Sherry Ben, Yandong Li, Mandy Guo, Suhas Yogin, Yiming Gu, Huizhong Chen, Oliver Wang, Saining Xie, Howard Zhou, Kaiming He, Thomas Funkhouser, Jean-Baptiste Alayrac, Radu Soricut
- Category: Vision Generation & Understanding Systems
- Priority: P1
- Links: [project](https://vision-banana.github.io/), [arXiv](https://arxiv.org/abs/2604.20329), [HF Papers](https://huggingface.co/papers/2604.20329)
- Keywords: Vision Banana, image generation, visual understanding, generative pretraining, instruction tuning, dense prediction, semantic segmentation, instance segmentation, metric depth, surface normal

## 一句话结论

Vision Banana 证明 image generator 可以被改造成 generalist vision learner：通过对 Nano Banana Pro 做轻量 instruction-tuning，并把 segmentation、depth、surface normal 等任务输出统一参数化为 RGB images，模型可以用 image generation interface 处理 2D/3D visual understanding 任务。

## Related Works

- Generative vision pretraining: 论文类比 LLM 通过 next-token generation 学到语言理解能力，提出 image/video generator 也可能通过生成式预训练学到可迁移的视觉表征。
- Unified vision interface: Painter 等工作把不同 vision tasks 重新表达为 image-to-image prediction；Vision Banana 的重点是用强 image generator 的生成接口承载 dense perception，而不是为每个任务增加独立 head。
- Dense prediction specialists: SAM 3、SegMan、OpenSeeD、X-Decoder、Depth Anything、Depth Pro、MoGe、UniK3D、Marigold、StableNormal、DSINE、Lotus 等提供 segmentation、metric depth 和 surface normal 的专业 baseline。
- MLLM + specialist pipelines: Referring segmentation 和 reasoning segmentation 常把 MLLM 与 SAM 类模型组合；Vision Banana 更强调让生成模型直接输出视觉任务结果图，减少任务特定 pipeline。

## 问题与背景

传统 computer vision 系统通常按任务拆分：semantic segmentation、instance segmentation、referring segmentation、metric depth、surface normal 等任务各自有模型结构、输出格式、训练数据和后处理逻辑。这种方式在单任务上高效，但系统复杂度高，也不容易形成类似 LLM 的统一任务接口。

Multimodal LLM 可以用文本回答视觉问题，但 dense visual outputs 很难只用文本表达。像 segmentation mask、depth map、normal map 这样的结果天然是 image-like tensor。论文的核心背景判断是：如果 image generation model 已经学会生成视觉内容，那么它可能也学会了可用于视觉理解的表征；关键是把 perception task 的输出空间重新组织成生成模型擅长的 RGB image。

## 方法与系统设计

Vision Banana 基于 Nano Banana Pro。论文没有公开训练细节到可完全复现的工程级别，但 arXiv 摘要和项目页说明，其做法是在原始 image generation training data 之外，加入少量 vision task data 进行 lightweight instruction-tuning。目标是在不牺牲基础 image generation capability 的前提下，让模型按 prompt 生成视觉理解结果图。

系统接口上，输入是 image + textual instruction，输出仍是 RGB image。Semantic segmentation 可以被描述为按指定 color mapping 生成 per-pixel class labeling；instance segmentation 可以要求不同实例使用不同颜色；referring expression segmentation 通过自然语言指定目标区域；metric depth 和 surface normal 则被输出为可视化的 dense map。

这个设计把 perception 任务统一为 image generation problem。对系统实现来说，任务差异从 architecture/head 迁移到 prompt contract、color mapping、output decoding 和 metric evaluation。模型服务侧可以复用 image generator 的通用 inference path，但评测和下游消费仍需要稳定的 prompt 模板与后处理规范。

在 3D 场景中，项目页展示 metric depth prediction 可以结合 camera intrinsics unproject 成 point cloud。论文同时强调 Vision Banana 在 depth 训练和推理中不使用 camera intrinsics；intrinsics 是从 depth map 转成 3D point cloud 时的外部几何信息。

## 创新点

- 把 image generation training 明确定位为 visual understanding pretraining，而不只是内容生成能力来源。
- 将 dense perception outputs 统一表达为 RGB images，用同一个 image generator interface 覆盖 segmentation、depth 和 surface normal。
- 用 lightweight instruction-tuning 在保留生成能力的同时引入视觉任务能力，避免为每个任务训练独立 specialist head。
- 把 prompt、color map 和 output image 变成视觉系统的主要接口，形成类似 text generation 在 NLP 中的统一任务表达。
- 在 2D 与 3D understanding 上同时展示结果，说明该范式不局限于语义 mask，也能覆盖 geometric perception。

## 实验与结论

项目页报告 Vision Banana 在 zero-shot transfer setting 下覆盖 2D 和 3D vision tasks，并在多个指标上达到或接近 SOTA。需要注意，这里的 zero-shot 更偏向目标 benchmark/dataset 的 transfer；模型本身经过了少量 vision task data 的 instruction-tuning，不是完全未接触视觉任务的 base generator。

2D understanding 方面，项目页展示 Cityscapes semantic segmentation mIoU 为 69.9，高于 SAM 3 的 65.2，但低于 non-zero-shot SegMan-L 的 84.2。ReasonSeg val gIoU 中，Vision Banana + Gemini 2.5 Pro setup 达到 79.3。RefCOCOg val 和 SA-Co/Gold 等任务也被用于展示 referring/instance segmentation 能力。

3D understanding 方面，项目页给出 metric depth 在 6 个 benchmark 上平均 delta1 为 0.882，高于 Depth Pro、MoGe-2 和 UniK3D 的对应结果。Surface normal 任务中，Vision Banana 平均 angular error 为 15.549，优于 Marigold、StableNormal、DSINE 和 Lotus-2 的项目页对比结果。

论文结论是：image generation pretraining 可以像 LLM pretraining 一样形成 generalist capability；当 vision tasks 被重新参数化为 image generation，生成模型可以作为统一的视觉理解后端。

## 局限性与风险

- 官方页面和 arXiv 未提供可复现实验代码或模型权重，复现实验受 closed model 和未公开训练细节限制。
- RGB image 作为 dense output interface 易于统一任务，但会引入颜色编码、量化、采样随机性和后处理一致性问题。
- 生成模型可能输出视觉上合理但数值上不精确的结果；对 metric depth、几何测量或 safety-critical perception，需要额外校准和不确定性评估。
- Image generator inference 通常比 specialist perception model 更重；在实时 perception、移动端或高吞吐场景中，latency/cost 可能成为主要障碍。
- Vision task data 虽然较少，但仍是 instruction-tuning 的关键组成部分；结论不能简单等价为 base generator 具备完整 zero-shot dense perception 能力。

## 对 LLM Systems 的启发

- 统一接口不一定必须是 text；对 multimodal systems，选择与输出结构匹配的 modality 作为 serialization format 可能更有效。
- 许多 task-specific heads 可以被重新思考为 prompt contract + generative output + postprocessor 的组合，这会改变模型服务、评测和工具调用边界。
- 对 multimodal agent，生成的 segmentation/depth/normal map 可以作为中间状态进入后续 planning、robotics、UI grounding 或 scene reasoning。
- 系统评估不能只看 benchmark score，还要报告 prompt sensitivity、sampling stability、output decoding robustness、latency、cost 和下游可消费性。
- 这类模型提示未来的 foundation vision model 可能同时承担 generation backend 和 perception backend，但部署上仍需要 specialist fallback 与质量监控。

## 复现/阅读建议

- 先读 arXiv Abstract 与项目页 Overview，确认论文主张是 generative vision pretraining 和 RGB output interface，而不是新 segmentation/depth architecture。
- 阅读项目页 Capabilities 时重点看 prompt contract：不同任务如何通过自然语言与颜色映射表达输出约束。
- 对实验结果要区分 specialist baseline、non-zero-shot baseline、MLLM-assisted setup 和 Vision Banana 自身输出，避免把不同设置的指标直接混合比较。
- 若要复现开放版本，可跟踪 open-source image editing/generation models 的 zero-shot dense prediction 工作，并优先测试 semantic segmentation、metric depth、surface normal 三类任务。
- 工程验证至少记录 prompt、random seed/sampling setting、post-processing script、latency、cost、metric score 和失败样例；否则很难判断收益来自模型能力还是 prompt/后处理偶然性。
