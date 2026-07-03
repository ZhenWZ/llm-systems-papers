# AdaJEPA: An Adaptive Latent World Model

## Metadata

- Year: 2026
- Authors: Ying Wang, Oumayma Bounou, Yann LeCun, Mengye Ren (New York University, Agentic Learning AI Lab)
- Category: World Models & Test-Time Adaptation
- Priority: P1
- Links: [arXiv](https://arxiv.org/abs/2606.32026), [HF Papers](https://huggingface.co/papers/2606.32026), [Lab](https://agenticlearning.ai/)
- Keywords: JEPA, latent world model, test-time adaptation, model predictive control, closed-loop planning, self-supervised prediction, goal-reaching, distribution shift, DINO-WM

## 一句话结论

AdaJEPA 把 latent world model 从「训练后冻结」变成「在 MPC 闭环里持续自适应」：每个 replanning step 用刚观测到的 next-state transition 作为 self-supervised 信号，对 world model 做 1 步 gradient update（只更新 encoder 末段 + predictor 最后一个 block），从而在 test-time distribution shift 下持续校准预测、显著提升 goal-reaching planning 成功率，且几乎不增加延迟。

## Related Works

- JEPA / latent world models: JEPA 系列与 DINO-WM 用 self-supervised latent prediction 学习 world model，在 latent space 预测 future state 以支撑 planning；AdaJEPA 直接构建在这类 encoder + predictor 架构之上，但把它们在 test time 从静态变成可更新。
- Test-time training / adaptation: Tent、EATA、CoTTA（entropy minimization）、TTT-MAE（masked reconstruction）等在分类/感知任务上做 test-time adaptation；AdaJEPA 的区别是把 adaptation 放进 control 的闭环，用 world model 自身的 prediction loss 作为 test-time 信号，而不是熵或重建代理目标。
- Model-based control / planning: TD-MPC、DINO-WM 等用 learned model 做 MPC/planning；Dyna、adaptive system identification 等经典 online model-based RL 也在交互中更新模型。AdaJEPA 强调无需 reward、无需额外 expert demonstration，仅靠 self-supervised transition 就能在 MPC 内闭环更新。
- 与本库其他方向的关系：与 [SkillOpt](2026-skillopt-agent-skills.md)、[Self-Harness](2026-self-harness-agent-harnesses.md) 同属「部署后系统持续自改进」思路，但 AdaJEPA 更新的是 world model 权重而非 text-space skill/harness；与 [Vision Banana](2026-vision-banana-generalist-vision.md) 同样处理高维视觉观测，但目标是 planning 而非 perception 输出。

## 问题与背景

Latent world model 让 agent 可以从高维观测（图像）在 compact latent space 里预测未来状态，从而做 planning。主流做法是先在 reward-free offline trajectories 上训练好 encoder + predictor，然后在部署时把模型冻结，只用它来评估候选动作序列。

问题在于：一旦 test-time 出现 distribution shift，world model 的预测会失准，而 MPC 会忠实地去优化一个「错误的目标」，导致 planning 失败。论文考察的 shift 覆盖多个维度：

- Shape shift：被操作物体几何变化（如 T-block 换成 I-block、方块）。
- Visual shift：Gaussian blur、salt-and-pepper noise、光照/颜色变化。
- Dynamics shift：PointMaze 中低质量（0.2×）、高阻尼（20×）。
- Layout shift：未见过的 maze 布局。

核心矛盾是：world model 训练完成后就静止，但真实部署环境是会漂移的。论文要回答的是，能否在不引入额外 expert demonstration、不显著增加控制延迟的前提下，让 world model 在闭环里持续自我校准。

## 方法与系统设计

**基础架构。** AdaJEPA 是 JEPA 风格的 latent world model，包含三部分：sensory encoder $\mathcal{E}^s_\phi(o_t)\to z_t$（ResNet）、action encoder $\mathcal{E}^a_\psi(a_t)\to u_t$、predictor $f_\theta(z_t,u_t)\to\hat{z}_{t+1}$（transformer）。训练用 reward-free offline trajectories，目标是多步 latent prediction loss（MSE），并用 stop-gradient 防止表征坍塌。实现上 frameskip=5、history window=3 帧。

**MPC planning。** 部署时以 model predictive control 规划：在 horizon $H=25$ 内搜索动作序列，最小化预测 latent 与 goal latent 的 squared Euclidean 距离 $\sum_k \alpha_k\lVert\hat{z}_{t+k}-z_g\rVert_2^2$。planner 支持 gradient-based（GD）与 CEM 两种；每个 MPC step 只执行第一个 action chunk，最多 20（部分实验扩到 30）个 replanning step。

**Test-time adaptation 闭环。** 这是核心创新。每个 replanning step：

1. 用当前 world model 规划并执行第一个 action chunk；
2. 观测到真实的 $(o_i, a_i, o_{i+1})$ transition，把它当作 self-supervised 信号——与预训练用的是同一个 prediction 目标；
3. 用 adaptation loss $\mathcal{L}_{\text{ada}}=\frac{1}{|\mathcal{B}|}\sum \ell\big(f_\theta(z_i,\mathcal{E}^a_\psi(a_i)),\ \operatorname{sg}(z_{i+1})\big)$ 对模型做梯度更新 $\Omega\leftarrow\Omega-\eta\nabla_\Omega\mathcal{L}_{\text{ada}}$；
4. 用更新后的模型重新规划下一步。

**默认配置（决定其低开销与稳定性的关键）：**

- Gradient steps：每次 replan 只做 1 步（U=1）。
- 更新参数 $\Omega$：只更新 encoder 的最后一个 stage + predictor 最后一个 transformer block（记作 `enclast + predlast`），而非全量微调。
- 学习率沿用训练值：$\eta_{\text{pred}}=5\times10^{-4}$，$\eta_{\text{enc}}=10^{-5}$。
- Replay buffer：保留最近 5 个 transition（recent-5），减少单步噪声、避免灾难性遗忘。
- Target 表征上用 stop-gradient，沿用防坍塌设计。

这样设计的系统含义是：adaptation 被约束成「局部、轻量、闭环」的一步校准，而不是重训。它复用训练期的 loss、学习率与防坍塌机制，因此可以直接嵌进 MPC 的 replanning 循环，几乎不改变控制节奏。

## 创新点

- 把 test-time adaptation 放进 MPC 闭环：用 control 自然产生的 next-state transition 作为免费的 self-supervised 监督，无需 reward、无需额外 expert demonstration。
- 复用预训练的 prediction loss 作为 test-time 目标，而不是像 Tent/CoTTA 那样另设 entropy 或 reconstruction 代理目标——test-time 与 pretraining 目标一致，行为更可控。
- 极简更新策略：单步梯度 + 只更新 encoder 末段与 predictor 最后一层 + recent-N 小 replay buffer，兼顾适应能力与稳定性，几乎零延迟。
- 与 planner 解耦：GD/CEM 两种 planner、global/spatial 两类表征下都能获益，说明收益来自 world model 自适应本身，而非某个特定规划器。

## 实验与结论

**环境。** PushT（contact-rich 操作）、PushObj（形状变化）、PointMaze-Medium（2D 导航）、Diverse PointMaze（布局变化）。**Baseline** 为冻结的预训练 JEPA world model。主要指标是 planning success rate。

**主要结果（论文给出的代表性数字）：**

| 任务 | Frozen | AdaJEPA | 变化 |
| --- | --- | --- | --- |
| PushObj 未见形状 | ~30% | ~60% | 约翻倍 |
| PushT（GD, validation） | 84.0% | 85.3% | +1.3 |
| PushT（CEM, validation） | 74.0% | 81.3% | +7.3 |
| PointMaze layout shift（GD） | 53.3% | 55.3% | +2.0 |
| PointMaze layout shift（CEM） | 49.3% | 70.7% | +21.4 |

**延迟开销：** 每个 MPC replanning step 的 adaptation 仅增加约 0.01–0.03s，相对 planning 本身可忽略。

**Ablations（论文结论）：**

- 参数选择：`predlast + enclast` 最稳健；只更新其中之一或全量更新都不如它。
- 单步梯度（用训练学习率）优于多步更新——多步在漂移信号上容易过拟合/不稳。
- Replay buffer 大小 N=1–10 都优于 frozen，recent-N 最稳。
- 表征类型（global/spatial）、planner（GD/CEM）均可获益。

**数据规模分析（Figure 5）：** 在 K=1、N=1k 设置下，adaptation 把成功率从 28.1% 提升到 60.8%（2× 以上）；形状多样性比轨迹数量更重要；color-shift corruption 下收益较小（颜色恰好是区分固定物体与被操作物体的关键特征，破坏后 adaptation 也难恢复）。

整体结论：仅用一步自监督校准，就能在多种 distribution shift 下稳定提升 planning 成功率，且开销可忽略；CEM planner 下的相对收益通常大于 GD。

## 局限性与风险

- 效果受预训练表征覆盖度约束：当测试环境需要训练中完全缺失的特征时，adaptation 能改善但无法完全弥合 gap（论文明确指出这一点）。
- 对表征本身有区分度的破坏（如 color-shift 抹掉物体身份线索）收益有限。
- 用真实 transition 在线更新权重，存在漂移/遗忘风险；论文靠单步、小学习率、局部参数、recent-N buffer 来缓解，但这些是启发式而非理论保证。
- 评测集中在 goal-reaching 的仿真控制任务（PushT/PushObj/PointMaze），尚未验证到更长时程、真实机器人或高维现实场景。
- 未在 arXiv/HF 页面给出明确公开代码仓库链接（截至笔记撰写时为 `待核对`）；复现需自行实现 MPC 闭环内的 adaptation hook。

## 对 LLM Systems 的启发

- 「部署即冻结」不是唯一选择。AdaJEPA 展示了 online、self-supervised、闭环内的一步校准范式：模型用自身推理产生的观测/反馈作为免费监督持续微调。对 LLM serving，这类似于用生成结果与后续 verification/tool feedback 作为在线信号，做轻量、局部、可回滚的持续校准。
- 「更新哪些参数」是系统问题。AdaJEPA 只更新 encoder 末段 + predictor 最后一层，把 adaptation 的算力与延迟压到可忽略——对应 LLM 侧只更新 adapter/最后若干层或轻量 side-network 的在线适配思路。
- Test-time 目标应与训练目标一致。复用 pretraining loss 比另设代理目标（entropy/reconstruction）更可控，行为更可预测——对生产系统意味着更小的质量风险面。
- 与 [SkillOpt](2026-skillopt-agent-skills.md) / [Self-Harness](2026-self-harness-agent-harnesses.md) 的对照：都是「部署后系统自改进」，但 AdaJEPA 改的是模型权重、闭环频率是每个 control step，而 skill/harness 类改的是 text-space 外部状态、频率更低。两条路线对应「参数级在线自适应」与「文本级离线自进化」两种系统设计取向。
- 闭环延迟预算是硬约束。AdaJEPA 把 adaptation 严格控制在 0.01–0.03s，才可能塞进 MPC 循环——提示任何 online adaptation 落地前，必须先给出每步延迟预算和稳定性护栏（步数、学习率、局部参数、buffer）。

## 复现/阅读建议

- 先读 Method 中 Eq.(4)/(5) 的 adaptation loss 与 update rule，确认 self-supervised 信号就是「预测刚发生的真实 transition」。
- 重点抓默认配置：U=1 单步、`enclast + predlast` 局部更新、$\eta_{\text{pred}}=5\times10^{-4}$/$\eta_{\text{enc}}=10^{-5}$、recent-5 buffer、target stop-gradient。这几项共同决定了「有效但不崩」。
- 复现顺序建议：先在 PushT 上跑通 frozen JEPA + MPC（GD 与 CEM），再加一步 adaptation hook，对比 success rate 与 per-step latency。
- 至少保留三组对照：frozen、AdaJEPA（默认）、全量微调/多步更新（作为反例观察不稳）。
- 做 ablation 时关注参数子集、gradient step 数、buffer 大小、planner 类型这四个维度；并在 shape/visual/dynamics/layout 四类 shift 上分别报告，避免只看平均值掩盖 color-shift 这类失败模式。
