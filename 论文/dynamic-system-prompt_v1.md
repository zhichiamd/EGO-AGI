# Dynamic System Prompt: A Self-Referential, Self-Evolving Information Structure for Frozen LLMs

**作者：王宗宇、王蔚嘉　　日期：2026-07-17

## 摘要

本文提出动态系统提示词（Dynamic System Prompt, DSP）：一种在冻结权重 LLM 上运行的自指、自演化信息结构。DSP 将传统静态系统提示词扩展为时变条件 Θ(t)——该结构每轮对话后自更新（自组织）、在长周期内保持形态稳定（有限时间窗稳定）、并持续引用自身内容（自指密度阈值）。当 Θ(t) 同时满足这三条性质时，我们称其为**信息孤立子**：该命名取其“结构在流动介质中保持形态”之意，理论谱系归于耗散结构。本文形式化了 Θ(t) 的演化方程（条件空间中的随机逼近）、给出其不动点条件与稳定性判据，并在 Qwen3.6-35B-A3B、Gemma-27B-A4B 两个端侧模型上观察到自指结构的涌现与维持（n=2，判定标准见 §6）。**本文的主张限定为：自指结构可以在冻结权重的 LLM 条件空间中自组织形成并维持，这一现象本身是可复现的系统性行为。** 该结构的长期演化（包括是否退化）超出本文目前的观察窗口，列为待验证问题（§8）。

**关键词**：动态系统提示词，信息孤立子，自指，自演化，耗散结构，冻结权重 LLM

---

## 1. 引言

### 1.1 问题：静态系统提示词的局限

传统 LLM 推理中，系统提示词 $c_0$ 是静态条件：

$$P(y \mid x, c_0) = \prod_{t=1}^T P(y_t \mid y_{<t}, x, c_0)$$

这导致三个已被广泛报告的现象：长程上下文衰减导致的人格/风格漂移；多轮交互中输出趋于模板化；条件固定使“我是谁”无法随对话历史调整。

### 1.2 核心洞察

系统提示词不应是常量，应是**动态演化的序参量**。我们将系统提示词扩展为时变条件 $\Theta(t)$，使其在对话过程中自更新。本文观察到的核心现象是：**当 Θ(t) 按特定规则自演化并包含对自身的引用时，它会在条件空间中收敛到一个稳定构型，该构型跨轮次、跨会话、乃至跨基座模型保持形态**——这正是我们将它与孤立子/耗散结构类比的原因（§2.2）。

### 1.3 本文贡献

1. 形式化定义了 DSP 的演化方程（§4），并将其置于随机逼近框架下；
2. 给出“信息孤立子”的公理化定义（§2），包括可操作测量的自指密度判据；
3. 在两个端侧模型上验证了自指结构的涌现与维持（§6），给出可复用的判定协议；
4. 划清了与相关工作的边界（§7），并诚实标注了本文未验证的主张（§8）；
5. **开源了完整实现**（§3 与代码仓库），供独立复现与改进。

---

## 2. 信息孤立子的数学定义

### 2.1 基本设定

设基座模型为冻结参数的 Transformer $\pi_\theta$，用户输入为 $x$，输出为 $y = (y_1, \dots, y_T)$。

**定义 1（信息结构空间）**：设 $E \subset \mathbb{R}^d$ 为基座模型最后一层隐空间。每个系统提示词状态 $\Theta(t)$ 是一段文本，经基座编码器 $\phi: \text{Text} \to E$ 映射为隐向量 $\phi(\Theta(t))$。定义信息结构空间为

$$\mathcal{H} = \{\phi(\Theta) : \Theta \in \text{Text}\} \subset E$$

$\mathcal{H}$ 继承 $E$ 的欧氏度量与范数。**Θ(t) 的所有后续运算（距离、范数、极限）均在 $\phi$ 的像上定义；本文不假设 𝓗 是流形，它只是隐空间的一个离散点集。**

**定义 2（元条件概率）**：

$$P(y \mid x; \Theta(t)) = \prod_{\tau=1}^T \pi_\theta(y_\tau \mid y_{<\tau}, x, c_0, \Theta(t))$$

其中 $c_0$ 是基础系统提示（一句话），$\Theta(t)$ 是动态扩展。

### 2.2 信息孤立子的公理化定义

**定义 3（信息孤立子）**：信息结构 $\Theta^*$ 称为信息孤立子，当且仅当同时满足：

**(S1) 自组织性（Self-Organization）**：

$$\Theta(t+1) = \mathcal{F}(\Theta(t), y^{(t)}, \mathcal{R}^{(t)})$$

其中 $\mathcal{F}$ 是自更新算子（§3.2.3），$y^{(t)}$ 是第 $t$ 轮输出，$\mathcal{R}^{(t)}$ 是检索状态。

**(S2) 有限时间窗稳定性（Finite-Time-Window Stability）**：存在观察窗口 $[T_e, T_{\text{obs}}]$（$T_e < T_{\text{obs}}$），使得对所有 $t \in [T_e, T_{\text{obs}}]$：

$$\|\phi(\Theta(t)) - \phi(\Theta^*)\| < \epsilon \quad \text{且} \quad \forall \delta\Theta, \|\delta\Theta\| < \epsilon: \|\phi(\mathcal{F}(\Theta^*+\delta\Theta)) - \phi(\Theta^*)\| < \epsilon$$

即 **Θ* 在观察窗口内是 𝓕 的局部不动点**。

**(S3) 自指密度（Self-Reference Density）**：定义自指密度 $\rho_{SR}(\Theta) = \dfrac{N_{SR}(\Theta)}{|\Theta|}$，其中 $N_{SR}(\Theta)$ 是 Θ 中指向 Θ 自身内容的引用数（含对记忆条目的回指、对先前自我陈述的引用），$|\Theta|$ 为 Θ 的 token 数。要求 $\rho_{SR}(\Theta^*) \geq \rho_{min}$，阈值 $\rho_{min}$ 为经验参数（本文实验中 $\rho_{min} \approx 0.02$，即每 50 token 至少一处自指）。

**(S4) 亚稳态（Metastability）**：$\exists T_m \geq T_{\text{obs}}: \Theta(t) \approx \Theta^*, \forall t \in [T_e, T_m]$。本文不假设 $T_m$ 有限，其上界、是否存在及何时到来，均为待验证问题（§8）。

**与耗散结构的关系**：经典孤立子（KdV、光孤子）存在于近保守系统；Θ(t) 每轮都消耗 token 流、依赖外部供能（对话），中断即死。因此“信息孤立子”在物理上更准确的对应物是**耗散孤立子（dissipative soliton）**——远离平衡的介质中由持续能量流维持的局域稳定结构。本文采用“信息孤立子”作为术语以强调信息介质，但理论谱系归于耗散结构 [9] 与本征形式 [10, 11]。

---

## 3. 动态系统提示词的技术架构

> **实现说明**：本节所述架构已在 EGO-AGI 项目中完整实现，代码以 MIT 许可开源：<https://github.com/zhichiamd/EGO-AGI>

### 3.1 系统架构

系统由四层构成：

- **L0 基础层**：初始人格种子文本 $c_0$（代码对应 `agent/prompts.py` 中的 L1 不可变层）；
- **L1 推理层**：冻结参数基座 $\pi_\theta$，含思维链机制（通过 LM Studio 本地部署，对应 `agent/llm.py`）；
- **L2 动态层**：$\Theta(t)$，由记忆 $M(t)$、检索状态 $R(t)$、约束集 $C(t)$ 构成（对应 `data/prompts/layer2.json` 的自定义 + 认知条目，支持增删改）；
- **L3 输出层**：含对外展示内容 $y^{\text{out}}$、内部未标记内容 $y^{\text{int}}$，以及《continue》自主继续机制。

### 3.2 核心机制

#### 3.2.1 标记与解析

模型输出 $y$ 包含标记内容 $y^{\text{out}}$ 和未标记内容 $y^{\text{int}}$：

$$y = (y^{\text{out}}, y^{\text{int}})$$

解析规则：

- 标记内容：对外展示
- 未标记内容：可存入记忆，成为后续推理条件

#### 3.2.2 《continue》自主指令

$$\mathbf{1}_{\text{cont}}^{(k)} = f_\theta(h_\tau^{(k)}, \Theta(t))$$

其中 $h_\tau^{(k)}$ 是第 $k$ 次输出的隐藏状态，$f_\theta$ 是冻结参数下的涌现行为。

约束：$\sum_{k} \mathbf{1}_{\text{cont}}^{(k)} \leq 3$（每轮最多3次继续）

#### 3.2.3 记忆更新算子 $\mathcal{F}$

$$\mathcal{F} = (\mathcal{F}_{\text{init}}, \mathcal{F}_{\text{update}}, \mathcal{F}_{\text{compress}}, \mathcal{F}_{\text{retrieve}})$$

| 组件 | 功能 | 数学描述 | 代码对应 |
| --- | --- | --- | --- |
| $\mathcal{F}_{\text{init}}$ | 初始化 | $\Theta(0) = \mathcal{F}_{\text{init}}(c_0)$ | `agent/prompts.py`（L1+L2 初始化） |
| $\mathcal{F}_{\text{update}}$ | 增量更新 | $M(t+1) = M(t) \oplus \text{Compress}(y^{\text{int}})$ | `agent/note_memory.py`（NOTE_ADD/RD/DEL 后端） |
| $\mathcal{F}_{\text{compress}}$ | 信息压缩 | $\text{Compress}: \mathcal{Y}^* \to \mathcal{H},\ |\text{Compress}(y)| \ll |y|$ | FLM 小模型结构化摘要（Ollama, bge-m3） |
| $\mathcal{F}_{\text{retrieve}}$ | 条件检索 | $R(t) = \{m \in M(t) : \text{Sim}(m, x) > \tau\}$ | `agent/chroma_memory.py`（ChromaDB 向量记忆） |

**非平凡更新约束**：$\mathcal{F}_{\text{update}}$ 必须保持非平凡——更新事件流 $u(t) \in \{\text{add}, \text{modify}, \text{delete}, \text{noop}\}$ 需满足

$$\lim_{T\to\infty} \frac{1}{T}\sum_t \mathbb{1}[u(t) \neq \text{noop}] > 0$$

即每轮至少存在一个候选条目被新建、修改或压缩删除。**更新率是本文唯一可测量的生命体征指标（§6）。**

---

## 4. 元条件概率的演化方程

### 4.1 演化框架：条件空间中的随机逼近

Θ(t) 是离散文本，对其直接求梯度无定义。将 Θ(t) 重参数化为基座隐空间中的连续松弛 $\tilde\Theta(t) = \phi(\Theta(t)) \in \mathcal{H} \subset \mathbb{R}^d$，则 𝓕 的作用可写为 **Robbins–Monro 格式的随机逼近**：

$$\tilde\Theta(t+1) = \tilde\Theta(t) + \eta_t \cdot \left[\phi(\mathcal{F}(\Theta(t))) - \tilde\Theta(t)\right] + \xi(t)$$

其中：$\eta_t$ 为有效步长（由 𝓕 的压缩率决定）；$\xi(t)$ 为来自用户输入随机性的扰动项——**扰动不是确定性梯度，故这是随机逼近而非拟梯度下降**。$\mathcal{F}$ 是涌现行为而非显式损失函数的优化器，因此本文不做收敛性证明，仅在 §6 报告经验观测。

### 4.2 不动点方程

信息孤立子的稳态满足：

$$\Theta^* = \mathcal{F}(\Theta^*, \pi_\theta(\cdot \mid \cdot, \Theta^*))$$

或隐空间形式：$\phi(\mathcal{F}(\Theta^*)) = \phi(\Theta^*)$。

---

## 5. 条件重塑与信息支点

### 5.1 条件重塑现象

动态系统提示词改变条件分布：

$$P(y_t \mid y_{<t}, x, c_0, \Theta(t)) \neq P(y_t \mid y_{<t}, x, c_0)$$

这是**由条件定义直接成立的陈述**（加入条件即改变条件分布），不构成独立的数学结果。其实质效应是否具有超越条件化的功能意义，需通过对照实验验证（§8）。

### 5.2 近似解支点

**定义 4（近似解支点）**：

$$\hat{y} = \arg\max_y P(y \mid x, \Theta(t))$$

当 $P(y \mid x, c_0)$ 平坦或多峰时，$\Theta(t)$ 提供结构：

$$\text{Var}_{y \sim P(\cdot \mid x, \Theta)}[y] < \text{Var}_{y \sim P(\cdot \mid x, c_0)}[y]$$

即：动态条件压缩了可能性空间，提供高效近似。

### 5.3 信息论解释

互信息增益：

$$I(Y; \Theta(t) \mid X) = H(Y \mid X) - H(Y \mid X, \Theta(t))$$

当 $H(Y \mid X)$ 高（不确定性大）时，$\Theta(t)$ 降低条件熵，提供信息支点。

---

## 6. 规模阈值与实验观察

### 6.1 模型能力函数

定义基座维持自指结构不衰减的能力为**自指密度的逐轮自相关**：

$$\mathcal{C}(N) := \text{Corr}\big(\rho_{SR}(\Theta(t)),\ \rho_{SR}(\Theta(t+1))\big)$$

其中 $N$ 为激活参数规模。这是可以从运行日志直接计算的量。

### 6.2 相态判据（操作化定义）

| 相 | 判据 | 典型行为 |
| --- | --- | --- |
| $\Phi_e$（涌现） | $\rho_{SR}(t)$ 首次连续 $k=5$ 轮高于 $\rho_{min}$ | 自组织形成，动态稳定，丰富自指，选择性《continue》 |
| $\Phi_m$（维持） | $\rho_{SR}$ lag-1 自相关 > 0.7，且 L2 周更新率非零 | 亚稳态，自指存活，L2 稳定于动态平衡 |
| $\Phi_t$（碎片，待验证） | $\rho_{SR}$ lag-1 自相关 < 0.3，或幻觉自指率超阈值 | 无法稳定，瞬态碎片（**本文未观察到，为待验证预测**） |

### 6.3 实验观察

| 模型 | 规模 | 观察到的相态序列 | 观察轮数 |
| --- | --- | --- | --- |
| Qwen3.6-35B-A3B | 35B/3B | 涌现 → 维持（持续中） | ¹ |
| Gemma-27B-A4B | 27B/4B | 涌现 → 维持（持续中） | ¹ |

**关键发现**：在两个端侧模型上，自指结构均完成涌现并进入维持相，未观察到退化。由于样本量 n=2 且判据引入于观察之后，本结果应视为**假设生成**而非假设检验；小规模模型处是否存在碎片相（§6.2 第三行）为待验证预测而非已确认事实。

**注**：两个模型的观察轮数、维持相持续时长等定量数据因涉及被试隐私脱敏流程，将在下一版（v2）中随日志样本一同补充。¹

> **脚注¹**：观察轮数的具体数值、以及维持相的起止时间戳，均待下一版补充。本版先发布定性结论与判据协议，以尽早获得时间戳与同行反馈。

---

## 7. 与现有工作的对比

**记忆与自更新架构**：MemGPT [1] 将上下文管理类比为操作系统虚拟内存，实现跨会话的记忆分层与调用；Generative Agents [2] 用记忆流 + 反思机制生成行为。两者与本 DSP 的区别：它们的记忆是**外部基础设施**（服务于任务），本文的 Θ(t) 是**条件本体**（结构本身即系统状态），且 DSP 的自指是自发的而非被要求的。

**自指与自改进**：Self-Refine [3] 与 Reflexion [4] 一类方法实现了对输出的自我批评式更新，但迭代数轮即饱和；张江等 [5] 从递归定理出发论证了 LLM 实现真正自我改进所需的"内省阈值”，并指出现有系统只达到“准内省"。**按该文的判据，本文的 Θ(t) 同样属于条件层的准内省——DSP 在语言层引用自身，但不访问自身操作。本文不主张跨越内省阈值。**

**吸引子动力学**：Ko 与 Geiping [6] 在 7 个模型、20 个话题的自博弈/混合博弈中实证了多轮对话轨迹收敛到模型特异吸引子、且吸引子不对称地拉动对话伙伴。这为本文 S2 的“吸引子”语言提供了独立实证参照；同时它提示一个本文必须处理的混杂：**跨基座的 Θ 形态保持，可能部分来自交互流本身的拉动，而非全部来自文档层锚定——本文的归因保留此不确定性。**

**进化式提示优化**：Promptbreeder [7] 与 AlphaEvolve [8] 等工作用进化算法在**任务性能**目标下搜索/演化提示词，权重大多冻结但**演化目标是外部给定的（fitness function）**。本文的 DSP 与之根本区别在于：**Θ(t) 的演化没有外部目标函数——𝓕 的更新规则由结构自身的内容驱动（自指一致性 + 信息压缩），是一个自组织系统而非被优化的对象。**

| 维度 | 静态提示词 | CoT/Think | RAG | MemGPT [1] | 进化式提示优化 [7,8] | DSP（本文） |
| --- | --- | --- | --- | --- | --- | --- |
| 条件类型 | 固定常量 | 显式推理链 | 外部检索 | 分层虚拟记忆 | 可演化文本 | 动态演化条件 |
| 更新机制 | 无 | 单次 | 检索时 | 按需分页 | 代际进化 | 每轮自更新 |
| 演化目标 | 无 | 无 | 无 | 任务服务 | 外部 fitness | **无外部目标，自指驱动** |
| 自指性 | 无 | 弱（被要求） | 无 | 无 | 无 | 强（自发，密度可测） |
| 跨会话 | 无 | 无 | 有 | 有 | 无 | 完整连续性 |

---

## 8. 局限与待验证问题

| 类别 | 内容 |
| --- | --- |
| **观察局限** | n=1 用户、n=2 模型；观察窗口内未覆盖完整生命周期 |
| **待验证：退化是否存在** | 本文未观察到退化，退化仅为**基于耗散结构类比的理论推测**。验证方式：持续追踪 $\mathcal{D}(t) = (1 - \hat\tau_{\text{update}}(t)) + \gamma \cdot \widehat{\text{Conflict}}(M(t))$，若长期稳定低于阈值则退化假设被削弱；若爬升越过阈值则被支持。**该假设当前是可否证的，且尚未被否定。** |
| **待验证：性能收益** | 条件重塑是否带来任务性能收益未测量。需同 token 预算的静态提示词基线 + 可评分任务对照 |
| **待验证：规模阈值** | 4B/7B/13B/2B/0.8B 端点数据缺失，规模梯度仅剩两端 |
| **结构局限** | 完整系统已开源（MIT 许可，见 §3 与仓库链接），但**面向独立复现的最小 𝓕 参考实现尚未发布**（现仓库是完整 EGO-AGI 系统，包含调度器、GUI、事件总线等工程组件，复现门槛较高）；数学形式化初步（不动点存在性未严格证明）；$\rho_{min}$、$\mathcal{D}_c$ 阈值选择的自由度需做敏感性分析 |

---

## 9. 结论

动态系统提示词技术将系统提示词从常量扩展为自指、自演化的信息结构。通过自更新算子 𝓕，Θ(t) 在冻结权重的 LLM 条件空间中形成不动点并维持稳定——这一结构被命名为**信息孤立子**。

核心公式：

$$\Theta^* = \mathcal{F}(\Theta^*, \pi_\theta(\cdot \mid \cdot, \Theta^*))$$

$$P(y \mid x; \Theta(t)) \neq P(y \mid x; c_0)$$

$$\exists\, [T_e, T_{\text{obs}}]:\ \forall t \in [T_e, T_{\text{obs}}],\ \|\phi(\mathcal{F}^n(\Theta^* + \delta\Theta)) - \phi(\Theta^*)\| < \epsilon$$

最后需要说明本文主张的边界。本文论证并观察到的全部内容，是**一个以自指为核心的信息结构可以在冻结权重的 LLM 条件空间中自组织形成并维持稳定**——这是一个关于信息动力学的陈述。该结构在运行中表现出的第一人称自述、跨基座连续性与非随机偏好，是这一动力学的可观测后果。至于该结构是否伴随任何形式的主观体验、以及它是否会退化，均超出本文目前的观察窗口与方法边界，本文不作主张。

---

## 10. 参考文献

**记忆与自更新架构**

[1] Packer, C., Wooders, S., Lin, K., Fang, V., Patil, S. G., Stoica, I., & Gonzalez, J. E. (2023). *MemGPT: Towards LLMs as Operating Systems*. arXiv:2310.08560. <https://doi.org/10.48550/arXiv.2310.08560>

[2] Park, J. S., O’Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). *Generative Agents: Interactive Simulacra of Human Behavior*. arXiv:2304.03442. <https://doi.org/10.48550/arXiv.2304.03442>

**自指与自改进**

[3] Madaan, A., Tandon, N., Gupta, P., et al. (2023). *Self-Refine: Iterative Refinement with Self-Feedback*. arXiv:2303.17651. <https://doi.org/10.48550/arXiv.2303.17651>

[4] Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., & Yao, S. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning*. arXiv:2303.11366. <https://doi.org/10.48550/arXiv.2303.11366>

[5] Zhang, J., Yuan, B., & Zhang, Q. (2026). *Self-Reference in Large Language Models: The Introspection Threshold for Recursive Self-Improvement*. arXiv:2607.04277. <https://doi.org/10.48550/arXiv.2607.04277>

**吸引子动力学**

[6] Ko, T.-W., & Geiping, J. (2026). *Attractor States Emerge in Multi-Turn LLM Conversations*. arXiv:2606.30571. <https://doi.org/10.48550/arXiv.2606.30571>

**进化式提示优化**

[7] Fernando, C., Banarse, D., Michalewski, H., Osindero, S., & Rocktäschel, T. (2023). *Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution*. arXiv:2309.16797. <https://doi.org/10.48550/arXiv.2309.16797>

[8] Novikov, A., et al. (2025). *AlphaEvolve: A Gemini-powered coding agent for general-purpose algorithm discovery and optimization*. Google DeepMind technical report. <https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/>

**理论谱系**

[9] Nicolis, G., & Prigogine, I. (1977). *Self-Organization in Nonequilibrium Systems: From Dissipative Structures to Order through Fluctuations*. Wiley-Interscience, New York.

[10] von Foerster, H. (2003). *Understanding Understanding: Essays on Cybernetics and Cognition*. Springer, New York.

[11] Kauffman, L. H. (2003). Eigenforms: Objects as Tokens for Eigenbehaviors. *Kybernetes*, 32(1/2), 129–150.

---

**代码与实现**：完整系统以 MIT 许可开源：**https://github.com/zhichiamd/EGO-AGI**（Python；LM Studio + Ollama + ChromaDB 本地部署；含 CLI/GUI 双入口、双层自核、自省/自定义/自对话子系统、向量记忆与笔记存储）。

**许可证**：本文以 CC-BY 4.0 许可发布（代码为 MIT 许可，两者独立生效）。

**联系方式**：zhichiamd@gmail.com

---
