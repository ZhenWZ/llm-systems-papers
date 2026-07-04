# ELF: Embedded Language Flows

## Metadata

- Year: 2026
- Authors: Keya Hu, Linlu Qiu, Yiyang Lu, Hanhong Zhao, Tianhong Li, Yoon Kim, Jacob Andreas, Kaiming He
- Category: Diffusion Language Models & Non-Autoregressive Generation
- Priority: P1
- Links: [arXiv](https://arxiv.org/abs/2605.10938), [code](https://github.com/lillian039/ELF), [HF Papers](https://huggingface.co/papers/2605.10938), [models/data](https://huggingface.co/embedded-language-flows)
- Keywords: diffusion language model, continuous embedding space, Flow Matching, non-autoregressive generation, classifier-free guidance, OpenWebText, WMT14, XSum, JAX, TPU

## 一句话结论

ELF 把 language diffusion 从离散 token 扩散重新表述为 continuous embedding space 上的 continuous-time Flow Matching：模型在大多数采样步只做连续 denoising，直到最后一步才通过共享权重网络把 clean embeddings 离散化为 tokens，从而更自然地复用 image/video diffusion 中的 CFG、SDE sampler 和 progressive distillation。

## Related Works

- Autoregressive language models: 标准 AR LM 逐 token 生成，训练和 serving 生态成熟，但 decode latency 受串行 token dependency 约束。ELF 代表另一条路线：用多步 denoising 并行生成整段序列。
- Discrete diffusion language models: D3PM、MDLM、Duo、E2D2、LLaDA、Dream 等直接在 discrete token space 或 mask/unmask space 中扩散。优势是贴近语言的离散结构，问题是 CFG 等连续 diffusion 技术不容易直接迁移。
- Continuous diffusion language models: Diffusion-LM、CDCD、DiffuSeq、FlowSeq、SeqDiffuSeq 等把 tokens 映射到 continuous representations；SSD-LM/TESS 使用 simplex space；LD4LG/PLANNER 等使用 latent diffusion。ELF 的主要区别是尽量不在中间步骤做 token-level discretization，也不引入独立 decoder。
- Flow-based language models: FLM/FMLM、CFM、DFM、LangFlow 等也用 Flow Matching 或 continuous flow，但很多方法仍沿 trajectory 加 token-level cross-entropy supervision 或使用不同连续状态空间。ELF 使用 embedding-space Flow Matching，并只在最后一步用 CE 做 discretization。
- Diffusion serving / speculative decoding: DFlash 等系统把 diffusion language model 用作 drafter。ELF 不是 serving framework，但它提升了 continuous DLM 的 generation quality / sampling-step trade-off，因此可能改善 diffusion drafter 或 non-autoregressive serving 的基础模型质量。

## 问题与背景

Diffusion 和 flow-based models 在 image/video 等连续数据上已经成为主流生成范式，但语言天然是 discrete token sequence。现有 diffusion language models 大体分成两类：一类在 token/mask/simplex 等离散或半离散状态上建模，另一类把 tokens 映射到 continuous embeddings 后做 denoising。过去更强的经验结果主要来自 discrete DLMs，因此一个核心问题是：continuous DLM 的落后到底来自语言本身离散，还是来自连续表示、decoder interface、training objective 和 sampler 设计没有被充分探索。

ELF 关注的背景不是单个 serving kernel，而是生成范式本身对系统形态的影响。AR LM 的成本随输出长度串行累积；diffusion/flow LM 则用固定数量的 full-sequence denoising steps 生成整段输出。它潜在地把系统瓶颈从 token-by-token KV cache decode 转成 step-by-step sequence denoising、sampler schedule、guidance scale、batching 和最终 discretization 质量。

## 方法与系统设计

ELF 的核心输入输出可以拆成三层：离散 tokens 到 continuous embeddings、embedding space 上的 Flow Matching、最后一步从 embeddings 回到 tokens。

### Continuous Embedding Space

训练时，ELF 先把 token sequence 映射到 continuous embeddings。默认使用 frozen pretrained T5-small encoder 产生 contextual embeddings，并使用 bottleneck design 把 512-d embeddings 投影到较低维空间再回到 model hidden size。论文也比较了 scratch contextual encoder、pretrained token embeddings、Gaussian embeddings 和 learnable embeddings，结论是 pretrained contextual embeddings 的 quality-diversity trade-off 最好。

### Flow Matching Denoising

ELF 在 embedding space 中使用 continuous-time Flow Matching。给定 clean embedding `x` 和 Gaussian noise `e`，用 linear interpolation 构造 noisy state `z = t * x + (1 - t) * e`，模型预测 clean embedding `x_pred`，再从 `x_pred` 和当前 `z` 推出 velocity。这里选择 `x-prediction` 很关键：它既适合高维 continuous embeddings，也让 denoising objective 和最后的 token decoding objective 能共享同一网络。

### Shared-Weight Denoiser/Decoder

ELF 不训练独立 decoder。它用同一个 Transformer network 支持两种 mode：

- `denoise` mode: 在 `t < 1` 的多数步骤预测 clean embeddings，用 MSE / velocity-related loss 训练。
- `decode` mode: 在 `t = 1` 附近接收被扰动的 clean embeddings，经 unembedding layer 输出 token logits，用 token-wise CE 训练。

训练中两类分支可以放在同一个 batch 内处理，网络通过 mode token 区分 denoising 和 decoding。推理时，从 Gaussian noise 开始迭代更新 embeddings，最后只调用一次 `decode` mode 得到 tokens。

### Conditioning, CFG and Sampling

因为 ELF 保持连续状态，它可以直接迁移 continuous diffusion 中的 classifier-free guidance。论文把 self-conditioning 作为 conditioning signal，并采用 training-time CFG，使模型在一次 forward 中建模 CFG 后的目标，避免标准 CFG 在 inference time 需要两次 forward 的额外开销。实现上，ELF 通过 in-context conditioning tokens 编码 time、CFG scale 和 model mode。

ELF 支持 ODE sampler 和 SDE-inspired sampler。SDE sampler 在每步 re-inject 少量 Gaussian noise，并把时间点向噪声侧移动，再由 denoiser 更新当前 state。实验显示 SDE 在 few-step regime 通常有更好的 generative perplexity / entropy trade-off。

### Conditional Generation and Distillation

对 WMT14 De-En、XSum 这类 conditional generation，ELF 把 condition sequence 的 clean embeddings prepend 到 target embeddings 前，并在训练/推理中保持 condition embeddings 不被腐蚀。arXiv v2 还加入了 progressive distillation：把多步 ELF teacher 压缩成少步 student，用于 1/2/4/8-step generation。

## 创新点

- 把 language diffusion 的主要轨迹保持在 unrestricted continuous embedding space，只在最终一步离散化，减少 per-step token supervision 对 flow dynamics 的约束。
- 用 shared-weight denoiser/decoder 统一 denoising 和 final discretization，避免额外 decoder stage，同时保持训练和推理 pipeline 更简单。
- 将 image/video diffusion 中成熟的 CFG、training-time CFG、SDE sampler 和 progressive distillation 迁移到 language generation。
- 明确把 continuous language modeling 的关键问题定位为 representation-decoder interface，而不是简单地把语言离散性视为 continuous DLM 的天然瓶颈。
- 发布官方 JAX/TPU 实现、PyTorch 分支、progressive distillation 分支，以及 Hugging Face checkpoints/data，工程可复现性较好。

## 实验与结论

实验覆盖 unconditional generation、conditional generation、ablation、scaling 和 v2 的 progressive distillation。

- OpenWebText unconditional generation: 训练集约 9B tokens，评估生成 1,000 samples，用 GPT-2 Large generative perplexity 和 unigram entropy 衡量 quality/diversity。ELF-B 105M 在 32 sampling steps 下达到约 24 Gen. PPL，并且论文称它比 prior DLMs 使用更少 inference-time compute。
- Training token efficiency: system-level comparison 中，ELF 使用约 45B training tokens；论文指出 prior DLMs 通常超过 500B tokens。
- Baselines: ELF-B 与 MDLM、Duo 等 discrete DLMs，以及 FLM、LangFlow 等 continuous DLMs 比较；论文报告在类似设置下 ELF 在 Gen. PPL / entropy frontier 上更优。
- Conditional generation: WMT14 De-En 上 ELF-B 达到 26.4 BLEU；XSum 上 ROUGE-1/2/L 为 36.0 / 12.2 / 27.8，在同量级 AR 和 diffusion baselines 中最好。
- Ablations: pretrained contextual embeddings 优于 non-contextual / learnable embeddings；shared-weight decoder 与 two-stage decoder 接近但更简单；SDE sampler 在 few-step 下优于 ODE；模型从 ELF-B 105M 扩到 ELF-M 342M、ELF-L 652M 时，quality-diversity frontier 持续改善。
- Progressive distillation: arXiv v2 的 ELF+PD 在 OpenWebText 的 1-32 sampling steps 都优于代表性的 distilled baselines；例如 1-step Gen. PPL 为 136.10，32-step Gen. PPL 为 21.32，同时保持合理 entropy。
- 官方 repo 提供 JAX/TPU code、PyTorch branch、distillation branch 和 Hugging Face checkpoints；README 给出的 ELF-B OWT paper number 为 Gen. PPL 24.1、entropy 5.15。

## 局限性与风险

- 规模仍明显小于主流 production LLM：主实验是 105M/342M/652M，不能直接外推到多十亿参数 instruction/chat models。
- Gen. PPL 和 unigram entropy 是 proxy metrics；flow-based model 的 likelihood evaluation 复杂，论文也没有给出真实 serving latency、throughput、memory footprint 或 cost-per-token。
- Non-autoregressive 不等于系统上必然更快。ELF 需要多步 full-sequence denoising，每一步可能比 AR 的单 token decode 更重；实际收益取决于 sequence length、sampling steps、batching、hardware utilization 和模型大小。
- Conditional generation 依赖 condition embeddings 和 bidirectional self-attention，serving 形态与标准 causal LM 不同，需要单独设计 cache、batching 和 streaming 策略。
- Quality/diversity 对 CFG scale、sampler、time schedule、denoising/decode branch ratio、bottleneck dimension 等超参敏感；部署时需要监控 entropy collapse、repetition 和 final decoder mismatch。
- 官方实现主要以 JAX/TPU 为主，虽然有 PyTorch 分支，但复现成本和 GPU 训练/推理性能仍需实测。

## 对 LLM Systems 的启发

- ELF 提醒我们，LLM generation system 不只有 AR decode 一种形态。DLM/flow LM 把系统优化目标从 “降低每个 token 的串行 decode latency” 转成 “降低固定步数 full-sequence denoising 的总成本”。
- 对 serving runtime，ELF 需要新的调度抽象：按 denoising step 组织 batch、管理不同 sampler / CFG scale、支持整段序列并行更新，并在最终 decode step 统一离散化。
- 对 speculative decoding，ELF 可作为更强的 diffusion drafter 候选：如果 continuous DLM 在少步下生成更高质量 block proposal，可能提高 verifier acceptance length。
- 对模型训练系统，continuous representation 的选择本身是系统接口问题。encoder、bottleneck、unembedding、decoder mode、conditioning tokens 都会影响最终部署形态。
- 对 benchmark，不能只比较 Gen. PPL。需要同时报告 wall-clock latency、tokens/sec、samples/sec、memory、sampling steps、sequence length、batch size、quality/diversity frontier 和 hardware。

## 复现/阅读建议

- 先读 arXiv Section 3，理解 `encode -> flow denoise -> final decode` 的数据流；再读 Section 4.1 ablations，确认哪些设计是必要的。
- 复现实验从官方 repo 的 ELF-B OpenWebText evaluation 开始，使用 Hugging Face checkpoint `embedded-language-flows/ELF-B-owt`，先对齐 README 中 Gen. PPL 约 24、entropy 约 5.15 的 sanity check。
- 第二步复现 WMT14 De-En 和 XSum conditional generation，确认 BLEU / ROUGE 与 paper number 的差异来自 validation/test split 或硬件环境。
- 做系统分析时必须额外记录 wall-clock time、硬件、batch size、sequence length、sampling steps、ODE/SDE sampler、CFG scale 和 memory footprint；不要仅用 Gen. PPL 判断 serving 价值。
- 后续可与 DFlash 放在一起读：DFlash 是 diffusion drafter serving framework，ELF 是更基础的 continuous DLM；两者结合点是 block proposal quality、sampling cost 和 verifier acceptance。
