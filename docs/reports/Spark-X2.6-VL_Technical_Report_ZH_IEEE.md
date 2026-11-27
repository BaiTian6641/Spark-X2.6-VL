# Spark-X2.6-4B-Loop48-En1B-VL 阶段性里程碑技术报告（Stages 7–13）：多阶段视觉锚定、反事实因果验证与后续扩展方向

**阶段性里程碑技术报告（中文版）——标准 IEEE 引文格式**  
**公开模型仓库：** `TassP0lar/Spark-X2.6-4B-Loop48-En1B-VL`（Stage-13 BF16、FP8+q4g128 及 Stage-13A 预览权重）  
**报告日期：** 2026年11月27日

---

## 摘要

本里程碑报告系统总结了 **Spark-X2.6-4B-Loop48-En1B-VL** 在 Stage-7 至 Stage-13 多日迭代中的架构状态、实证评测结果、反事实因果验证与现存局限。该系统在 36 层纯文本推理骨干 `Spark-X2.6-4B` 的基础上构建了总参数量为 6.0477B（含 CPU 锁页稀疏条件记忆）的多模态推理架构，集成 48 虚拟层权重共享 `Dual-mHC` 循环推理、0.999B 分层 MoE-Engram 查表记忆子系统（`En1B`），以及经由 FP32 `RadioMultiLayerGELUProjector`（附带二维换行符行断标记与单侧视觉 RMS 上界正则化）桥接的 `nvidia/C-RADIOv4-H`（651.6M）视觉编码器。全部训练均在单台异构工作站（1× NVIDIA RTX PRO 4000 Blackwell 24 GB + 1× Intel Arc Pro B50 16 GB）上完成，线性谱系累计处理 188,756 条样本（估算约 114.2M tokens），耗时 100.11 小时；其中 Stage-13 专项精调耗时 18.43 小时（新增约 26.37M tokens），融合了 `TassP0lar/PRISM-Omni-Dataset-v1`、零重叠 `COCO` 全 80 类物体存在性与空间布局描述、`DOCCI`、`ChartQA`、`ScienceQA`、`ViRL39K`、反事实颜色×形状网格，以及在线视觉因果门控强化学习（`VC-GRPO`）。在严格零重叠基准上，Stage-13 将标准 8 基准多模态宏平均由 Stage-12 的 37.75%（BF16）/ 37.68%（FP8）提升至 **44.15%（BF16）/ 44.83%（FP8+q4g128）**，并在 **POPE**（62.78% → 86.11% 终版；Stage-13A 达 89.44%）、**MMBench-CN**（32.50% → 50.83% / 51.67%）、**AI2D**（63.33% → 67.50%）、**BLINK 六项空间任务**（43.52% → 47.22% / 49.07%）、**MMStar**（28.33% → 32.22% / 33.33%）、**CV-Bench 3D**（55.83% → 60.00%）、**3DSRBench**（44.17% → 51.67%）与 **MMMU-Pro**（20.83% → 22.50%）上均取得一致提升。独立的真实图像 vs. 纯灰空白图像 vs. 随机错配图像反事实审计表明，Stage-12 存在的模态解耦现象已在全部视觉审计赛道得到消除，`POPE`（+36.11%）、`RealWorldQA`（+11.25%）、`BLINK`（+5.56%）、`AI2D`（+3.75%，对称 KL 散度由 0.0005 增至 0.9487 nats）、`颜色×形状`（100.0%）与未见合成词 OCR（66.67% 精确匹配 / 86.67% 字符相似度）均呈现正向视觉净增益。同时，诚实的定性与定量误差分析表明当前仍存在明确短板：开放式多句视觉解释偏简短且偶发细粒度类别混淆，5 组全量 `OCRBench`（5.00%）在密集文档与手写数学公式上仍受限，`MMMU (val)`（-4.63%）、`GPQA-Diamond`（-2.02%）与 `Terminal-Bench 4.1`（-9.52%）出现小幅权衡回撤，且累计多模态 Token 规模较工业级开源 VLM 仍存在 $10^{3}\times$ 至 $10^{4}\times$ 差距。本文客观陈述了当前已具备的能力、由实验数据合理推断出的机理规律，以及面向下一阶段单机与集群扩展的可执行路线。

**关键词—** 视觉语言模型，反事实视觉评测，循环 Transformer，稀疏条件记忆，可验证奖励强化学习，多模态扩展分析。

---

## 一、 引言与报告范围

将紧凑型纯文本推理大语言模型（LLM）扩展为视觉语言模型（VLM），在公开文献中通常依赖 $10^{9}$ 至 $4 \times 10^{12}$ 级别的多模态 Token 预算 [1]–[6]，这与计算最优扩展规律一致 [7]。当研发受限于单台工作站算力时，需要重点回答两个基础问题：（1）在显存与吞吐约束下，能否通过稳定的优化与数据配比建立真实、像素驱动的视觉感知而不严重损伤语言推理能力；（2）如何通过反事实对照手段区分真实的视觉理解与基于语言先验的快捷作答 [20], [21]。

本报告作为 **Spark-X2.6-4B-Loop48-En1B-VL** 完成 **Stage-13** 后的阶段性里程碑总结，基于多日连续训练与评测所记录的原始机器日志，采用客观审慎的学术叙述，围绕三个核心部分展开：（1）**当前已有成果**（模型拓扑、算力账本、零重叠基准与反事实实测数据）；（2）**可合理推断的规律与局限**（层深度分工、RLVR 门控特性、解释能力短板与工程部署发现）；（3）**下一阶段可行方向**（单机 Stage-14 精调与多卡集群扩展方案）。

---

## 二、 当前已有成果：系统架构、训练账本与数据治理

### A. 精确参数审计（总参数量 6.0477B）

公开版本（`TassP0lar/Spark-X2.6-4B-Loop48-En1B-VL`）以 36 层 `Spark-X2.6-4B` 为语言骨干（原生 1M RoPE 上下文；27/36 层采用 `sliding_window=512` 滑动窗口注意力，每第 4 层采用全局注意力），并集成四项子系统（表 I 为基于导出 BF16 权重的直接参数审计）：

1. **48 虚拟层循环机制（`Loop48` Dual-mHC）：** 第 $[12, 24)$ 层通过双随机流形约束残差路由与秩-16 轮次适配器执行两轮循环推理 [11]–[13]（+3.07M 参数）。
2. **分层 MoE-Engram 条件记忆（`En1B`）：** 在 8 个深度锚点（$[3, 7, 11, 15, 19, 23, 27, 31]$）接入 16 专家（`top_k=4`）稀疏查表记忆，拆分为 GPU Tier-1A 热桶与主机 CPU 锁页 Tier-1B 冷桶 [14]（合计 998.82M 参数）。
3. **聚合视觉编码器（`nvidia/C-RADIOv4-H`）：** 32 层 ViT-H/16 编码器（651.65M 参数），融合蒸馏自 SigLIP 2、DINOv3 与 SAM3 [8]–[10]。单图 patch 数量约束在 $\le 1{,}600$（$2\times 2$ 空间重排后 $\le 400$ 个视觉 token），以确保视觉前缀完整落入 512 滑动窗口内。
4. **FP32 `RadioMultiLayerGELUProjector` 与二维行断标记：** 提取 `C-RADIOv4-H` 第 24 与第 32 层特征（$\mathbf{F}^{(24)}, \mathbf{F}^{(32)}$），经独立归一化、拼接与 $2\times 2$ 空间像素重排得到 $\mathbf{U} \in \mathbb{R}^{N_g \times 10240}$，在严格 FP32 下结合活性 GELU 初始化（$\tau_{\text{init}} = 0.035$）与单侧 RMS 上界惩罚完成投影：
   $$\mathbf{V}_{\text{grid}} = \mathbf{W}_{2}\,\mathrm{GELU}\!\left(\mathbf{W}_{1}\,\mathrm{RMSNorm}(\mathbf{U}) + \mathbf{b}_{1}\right) + \mathbf{b}_{2}, \qquad \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda_{\text{rms}} \left[\max\!\left(0,\; \mathrm{RMS}(\mathbf{V}_{\text{grid}}) - \tau_{\text{cap}}\right)\right]^{2}$$
   其中 $\lambda_{\text{rms}} = 5.0$，$\tau_{\text{cap}} \in \{0.08, 0.09, 0.10\}$。每个特征行末尾显式附加原生换行符嵌入 $\mathbf{e}_{\text{nl}}$ 以维持二维阅读顺序。

**表 I**  
*`Spark-X2.6-4B-Loop48-En1B-VL`（Stage-13 BF16）模块级参数审计*

| 模块 / 子系统 | 精确参数量 | 占比 (%) | 设备部署与 Stage-13B 训练角色 |
| :--- | ---: | ---: | :--- |
| `Spark-X2.6-4B` 语言骨干（36层） | 4,112,079,360 | 67.99% | Blackwell CUDA（`APOLLO+COAT`，可训） |
| `nvidia/C-RADIOv4-H` 视觉编码器（32层） | 651,645,440 | 10.78% | Intel Arc B50 XPU（顶层 8 层 `Muon8bit`，157.4M 活跃） |
| `En1B` Tier-1B CPU 锁页冷缓冲 | 578,813,952 | 9.57% | 主机 CPU 锁页内存（稀疏查表） |
| `En1B` Tier-1A 热桶与路由 | 420,004,136 | 6.94% | Blackwell CUDA（可训） |
| `GatedVisualCrossAttention`（3 组） | 196,723,206 | 3.25% | Blackwell CUDA（可训） |
| `RadioMultiLayerGELUProjector` | 65,558,018 | 1.08% | Blackwell CUDA FP32（可训） |
| 辅助空间 / Alpha / 相机位姿连接器 | 19,811,840 | 0.33% | Blackwell CUDA（可训） |
| `Loop48` Dual-mHC 控制器 | 3,072,600 | 0.05% | Blackwell CUDA（可训） |
| **总参数量（含 CPU Tier-1B 缓冲）** | **6,047,708,552** | **100.00%** | **BF16 12.69 GB / FP8+q4g128 10.69 GB** |

### B. 异构训练算力账本（Stages 7–13）与 Stage-13 课程设计

全部训练运行于单台异构双卡工作站：**1× NVIDIA RTX PRO 4000 Blackwell SFF（24 GB）**运行 `APOLLO`（`rank=64`）[15] 与 `COAT` FP8 分组-16 激活压缩 [16]，协同 **1× Intel Arc Pro B50（16 GB）**通过双锁页共享内存执行 8-bit `Muon8bit` 视觉微调 [17]。中间检查点 Stage-13A 另在远程 **NVIDIA A100-SXM4-80GB** 洁净环境中完成了独立复核评测。

针对 Stage-12 审计发现的通用 VQA 模态解耦与 `POPE-Random` 长尾误报问题，**Stage-13** 从 `final-model-stage12-bf16` 热启动（`reinit_vision_projector=False`），基于 **57,485 条严格审计的商用合规数据**（含 54,741 对图像样本，`failures_count = 0`）构建训练语料，涵盖：（a）`TassP0lar/PRISM-Omni-Dataset-v1`（锁定修订 `8e5ac6197b`，8,257 条纯视觉-文本样本，覆盖 `scenes3d`、`charts`、`paths`、`mazes`、`gvsr`、`pixmo_count`、`pixmo_points` 与下采样 `shapes`，统一转为 Spark XML 空间原语）；（b）`COCO train2017` 全 80 类物体存在性与空间布局描述（13,000 条样本，负采样严格按 35% 全 80 类均匀随机 + 45% 共现对抗 + 20% 热门配比，并经 500-ID `POPE val2014` 守卫自动剔除 34 张重叠图像，实现 0/500 重叠）；（c）本地缓存的 `DOCCI` 稠密图像描述（1,494 条）[34]、`ChartQA`（2,600 条）[35]、`ScienceQA` 含图科学问答（2,400 条）[36] 与 `ViRL39K`（3,600 条，自动过滤 53 条含评测集名称来源样本）[37]；（d）2,400 条反事实 `颜色×形状×空间关系` 网格；（e）`DocVQA`、`TextVQA`、`A-OKVQA`、`SynthDoG-EN`、`LaTeX`、`GQA-open`、`Nemotron-v3`（含 1,200 条中文 `zhwiki`）、`SyncCraft v38/v28` 及前沿推理回放数据。

Stage-13 分为三个连续阶段（共 18.43 小时，估算新增 26.37M tokens；见表 II）：
- **Phase 13A（9.03 小时，7,948 步，$B=8$，63,584 条样本，15.12M tokens）：** 分块特征预计算（512 px 上限，每 100 步主机峰值内存 $\le 2.5$ GiB），联合更新 FP32 GELU 投影器与前 12 层 `self_attn`，每微批次固定 8 路均衡槽位（`ema_ce` 由 `2.7495` 降至 `0.7045`）。
- **Phase 13B（6.29 小时，950 步，`grad_accum=12`，11,400 条样本，约 9.69M tokens）：** 全 36 层 `APOLLO+COAT` 联合 Arc B50 顶层 8 块 `C-RADIOv4-H`（`Muon8bit`，151.0 步/小时），恢复 896 px 完整分辨率并引入真图-空白视觉边际加权（3,517 次探针均值 `mean_adv = 0.9269 nats`）。
- **Phase 13C（`VC-GRPO`，2.73 小时，1,200 题 × $K=4$ = 4,800 条在线采样，156 步，1.57M 计分 tokens）：** 在 29,537 条可验证多模态池上执行在线分组相对策略优化（GRPO）[32]，结合逐条采样的真图-空白对数概率差（$\Delta\log p = \overline{\log p}(y \mid I_{\text{real}}, q) - \overline{\log p}(y \mid I_{\text{blank}}, q)$）进行视觉因果门控，并预留了面向人类视觉纠错的直接偏好优化（DPO）[33] 接口（`collect_user_vl_feedback.py`，本次自动运行中人类反馈条数为 0）。

**表 II**  
*Stages 7–13 单机双卡工作站线性演化算力与 Token 账本*

| 训练阶段 | 视觉编码器与投影器配置 | 更新步数 | 处理样本数 | 估算 Token 量 | 实测耗时 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Stages 7–7.5 | `SigLIP2-SO400M`（基座与对齐） | 3,231 | 38,772 | ~36.6M | 38.10 h |
| Stages 8–11.5 | `C-RADIOv4-H` + `SwiGLU` DeepStack | 4,510 | 54,120 | ~41.8M | 36.21 h |
| Stage-12（`12A–12C`） | `C-RADIOv4-H` + 新建 FP32 `GELU` + `\n` | 1,740 | 20,880 | ~9.4M | 7.37 h |
| **Stage-13（`13A+13B+13C`）** | **热启动 `GELU` + `PRISM-Omni` + `VC-GRPO`** | **9,054** | **74,984*** | **~26.4M** | **18.43 h** |
| **线性谱系累计（7→13）** | — | **18,535** | **188,756** | **~114.2M** | **100.11 h** |
| **现行 `GELU` 投影器累计（12→13）** | **Stage-12/13 有效投影器谱系** | **10,794** | **95,864** | **~35.8M** | **25.80 h** |

*\*表内为 13A（63,584）与 13B（11,400）的监督样本总数；13C 另生成 4,800 条在线 RLVR 采样轨迹。*

---

## 三、 严格零重叠定量基准评测结果

表 III 与表 IV 汇总了锁定不可变 SHA 版本的标准 8 基准多模态套件（`scale=0.6`，每模型 1,056 题）及三维空间与前沿推理套件的实测结果。需要说明的是，导出的 `final-model-stage13-{bf16,fp8}` 对应 `post_stage13b` 权重，原因在于 Phase 13C 触发了自动 SFT 交叉熵门控回滚（`loss_ratio = 1.0513 > 1.04`，详见第五节 B）；表中一并列出在 Colab A100 上独立评测的 Stage-13A 中间权重以呈现完整演化轨迹。

**表 III**  
*冻结标准 8 基准多模态套件跨阶段评测结果（`scale=0.6`，$n=1{,}056$/模型，0% 训练重叠）*

| 模型阶段 / 精度 | 8 基准宏平均 | AI2D [22] | RealWorldQA [31] | BLINK 六空间 [23] | MMBench-CN [26] | MMMU Val [24] | MMStar [20] | POPE 准确率（`F1`）[18] | OCRBench 5 组 [19] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Stage-8 BF16 | 37.69% | 56.67% | 44.17% | 39.81% | 37.50% | 30.56% | 24.44% | 50.00%（`0.667*`） | 1.67% |
| Stage-11.5 BF16 | 36.80% | 60.00% | 37.50% | 40.74% | 35.83% | 31.48% | 32.78% | 52.78%（`0.536`） | 2.50% |
| Stage-12 BF16 | 37.75% | 63.33% | 40.83% | 43.52% | 32.50% | **31.48%** | 28.33% | 62.78%（`0.579`） | 4.17% |
| Stage-12 FP8+q4g128 | 37.68% | 64.17% | 40.00% | 43.52% | 32.50% | 28.70% | 28.89% | 64.44%（`0.595`） | 4.17% |
| Stage-13A BF16（A100） | 44.74% | 61.67% | **44.17%** | 43.52% | 50.83% | 30.56% | **33.89%** | **89.44%（`0.891`）** | 4.17% |
| **Stage-13 终版 BF16** | 44.15% | **67.50%** | 38.33% | 47.22% | 50.83% | 26.85% | 32.22% | 86.11%（`0.852`） | **5.00%** |
| **Stage-13 终版 FP8+q4g128** | **44.83%** | **67.50%** | 40.00% | **49.07%** | **51.67%** | 26.85% | 33.33% | 86.11%（`0.852`） | **5.00%** |
| *$\Delta$ Stage-13 较 Stage-12（BF16 / FP8）* | **+6.40% / +7.15%** | **+4.17% / +3.33%** | -2.50% / 0.00% | **+3.70% / +5.55%** | **+18.33% / +19.17%** | -4.63% / -1.85% | **+3.89% / +4.44%** | **+23.33% / +21.67%** | **+0.83% / +0.83%** |

*\*Stage-8 的 POPE F1 系全答 `"Yes"` 所致退化值（`yes_ratio = 1.000`）。Stage-13A（`yes_ratio = 0.472`）与 Stage-13 终版（`yes_ratio = 0.439`）在 Random（88.33%）、Popular（81.67%–88.33%）与 Adversarial（88.33%–91.67%）三个子集上均保持均衡的精确率与召回率。*

**表 IV**  
*三维空间、前沿推理与 SyncCraft 套件演进结果（BF16，相同全量切片，0% 重叠）*

| 基准赛道（冻结版本） | Stage-11.5 | Stage-12 | **Stage-13 终版** | $\Delta$（13 较 12） | 关键说明 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **CV-Bench 3D**（`n=120`）[21] | 49.17% | 55.83% | **60.00%** | **+4.17%** | 深度 59.0%，距离 61.0%，4 原语使用率 99.2% |
| **3DSRBench**（`n=120`）[27] | 48.33% | 44.17% | **51.67%** | **+7.50%** | 4 原语使用率 97.5%；平行 80%、近景/邻近 70% |
| **MMMU-Pro 10 选项**（`n=120`）[25] | 20.00% | 20.83% | **22.50%** | **+1.67%** | 超 10 选项随机水平 2.2 倍 |
| **GPQA-Diamond**（`n=198`）[28] | 34.85% | **38.89%** | 36.87% | -2.02% | 物理 45.35%，化学 31.18%，生物 26.32% |
| **Terminal-Bench 4.1**（`n=21`，语法/实质） | **66.67%** | 61.90% / 57.14% | 52.38% / 33.33% | -9.52% | `<think>` 健康闭合率 47.62% |
| **SimpleQA-Verified**（`n=120`）[30] | 3.33% | 1.67% | **3.33%** | **+1.66%** | `En1B` 稀疏事实记忆仍处于早期填充阶段 |
| **SyncCraft Gate D 提议器**（`n=4`） | **100.0%** | **100.0%** | **100.0%** | 0.00% | 平均求解器调用 1.0 次（243 tokens） |
| **SyncCraft 未见 `v28` 严格编译**（`n=4`） | — | 0.00% | 0.00% | 0.00% | 严格 DSL 语法偏差仍未解决 |

*注：为将自动评测控制在 24 小时窗口内，本次 Stage-13 后置脚本未重跑 `HLE-Diamond` [29]（Stage-12 为 8.00%）与 `AIME 2026`（Stage-12 为 0.00%）。*

---

## 四、 反事实视觉因果复核与开放式图像解释实测

### A. 真实图像 vs. 纯灰空白图像 vs. 随机错配图像反事实审计

为检验 Stage-13 的分数提升是否源自真实像素输入而非文本先验，我们在 Stage-12、Stage-13A（A100）与 Stage-13 终版（Blackwell）上执行了完全一致的独立反事实脚本（`scripts/verify_stage12_visual_causality.py`），记录真图、纯灰空白对照（`RGB(128,128,128)`）、错配真图对照（POPE）及首 token 对称 KL 散度 $D_{\text{KL}}^{\text{sym}}$（见表 V）。

**表 V**  
*独立反事实视觉因果演进对比：Stage-12 vs. Stage-13A vs. Stage-13 终版（BF16）*

| 反事实审计赛道 | Stage-12 真图 / 空白（$\Delta_{\text{vis}}$） | Stage-13A 真图 / 空白（$\Delta_{\text{vis}}$） | **Stage-13 终版 真图 / 空白（$\Delta_{\text{vis}}$）** | Stage-13 补充信号与 KL 指标 |
| :--- | :---: | :---: | :---: | :--- |
| **POPE 全集（`n=180`）** | 62.78% / 55.00% (+7.78%) | **89.44% / 50.00% (+39.44%)** | **86.11% / 50.00% (+36.11%)** | 错配真图 53.33%（**较错配 +32.78%**；空白误报 0/90） |
| **AI2D（`n=80`）** | 71.25% / 73.75% (-2.50%) | 67.50% / 67.50% (0.00%) | **71.25% / 67.50% (+3.75%)** | $D_{\text{KL}}^{\text{sym}}$：0.0005 (12) → 0.0007 (13A) → **0.9487** (13) |
| **BLINK 六空间（`n=72`）** | 47.22% / 45.83% (+1.39%) | 41.67% / 44.44% (-2.78%) | **48.61% / 43.06% (+5.56%)** | 选项翻转率：2.78% (12) → **30.56%** (13) |
| **RealWorldQA（`n=80`）** | 36.25% / 35.00% (+1.25%) | 42.50% / 31.25% (+11.25%) | **36.25% / 25.00% (+11.25%)** | 预测翻转率：11.25% (12) → **36.25%** (13) |
| **合成颜色×形状开放题（`n=8`）** | 颜色 25.0% / 形状 50.0% (0.0%) | **100.0% / 100.0%** (+75%/+50%) | **100.0% / 100.0%** (+75%/+50%) | `"red circle"` 模板先验已完全消除 |
| **OCRBench 快速探针（`n=32`，原问）** | 31.25% / 0.00% (+31.25%) | 31.25% / 0.00% (+31.25%) | **40.62% / 0.00% (+40.62%)** | 字符相似度：**67.57%** vs. 3.17%（*13C 后达 43.75%*） |
| **未见合成英文词（`n=12`）** | 50.00% / 0.00% (+50.00%) | 50.00% / 0.00% (+50.00%) | **66.67% / 0.00% (+66.67%)** | 字符相似度：**86.67%** vs. 0.00%（$D_{\text{KL}}^{\text{sym}}=7.79$） |

### B. 开放式图像解释能力实测评估

鉴于标准选择题或短答基准难以充分反映开放式“看图解释”能力，我们在 5 张留出真实图像（`RealWorldQA`、`COCO`、`AI2D`、`CV-Bench 3D`）上对 `final-model-stage13-bf16` 执行了“详述图像并解释推理依据”的真图 vs. 纯灰空白对照测试，观察到以下客观现象：

1. **真实场景感知已成立：** 在一张包含单板滑雪者的 COCO 实拍图上，真图输出 `"A skier is skiing down a snowy slope."`，而空白对照输出 `"A person is standing on a sidewalk holding a skateboard."`；在 `RealWorldQA` 室内照片上，真图输出了电视、配色与房间光照布局，空白对照则输出无关的立方体先验；在 `CV-Bench 3D` 场景的 `<think>` 推理中，模型主动引用了提示词未提及的背景物体（`sofa`、`chair`）。
2. **解释类提示仍偏向极简输出：** 当在 `RealWorldQA` 与 `AI2D` 选择题末尾明确要求“结合图像简要解释推理过程”时，模型仍直接输出单字母 `"B"`；在 `CV-Bench 3D` 上，`</think>` 之后也仅输出 `"Answer: A"`，且内部思考链较早截断。
3. **细粒度命名与描述深度仍偏弱：** 自由描述通常限于 1–3 个短句，且存在相近视觉概念混淆（如将单板滑雪者表述为 `"skier"`）。

综合而言，**Stage-13 已具备真实、可复核的像素级物体存在性、属性、印刷文字与粗粒度三维空间感知能力，但在面向用户的流畅、详尽多句图像解释方面仍处于早期阶段。**

---

## 五、 由实验数据合理推断出的机理与局限

结合多阶段消融、Stage-13A/13B/13C 演化与定性探针，我们可以归纳出五项具有实证支撑的推断：

### A. 视觉感知与多选推理存在显著的层深度分工
对比 **Stage-13A**（仅更新 FP32 GELU 投影器与前 12 层 `self_attn`）与 **Stage-13 终版**（Phase 13B 解冻全部 36 层与顶层 8 块 `C-RADIOv4-H`），可以观察到清晰的分层规律：Stage-13A 已足以将 `POPE` 推至 **89.44%**（+39.44% 净增益）、`颜色×形状` 推至 **100.0%**、`RealWorldQA` 视觉增益推至 **+11.25%**，但在多选对比推理任务 `AI2D`（0.00% 增益，$D_{\text{KL}}^{\text{sym}} = 0.0007$）与 `BLINK`（-2.78% 增益）上仍未建立有效视觉耦合。直到 Phase 13B 解冻 12–35 层后，`AI2D` 才转为正向视觉增益（**+3.75%**，$D_{\text{KL}}^{\text{sym}}$ 跃升约 1,350 倍至 **0.9487 nats**），`BLINK` 同步升至 **+5.56%**（选项翻转率达 30.56%）。这表明浅层注意力可胜任视觉词元绑定与存在性检索，而深层 Transformer 块（12–35 层）的协同适配对复杂多选视觉比较至关重要。

### B. 在线 `VC-GRPO` 的分布锐化与教师强制 NLL 门控失配
Phase 13C 在 1,200 道可验证题目上完成了 **4,800 条在线采样**（`exact_rate = 60.38%`，平均视觉因果 $\Delta\log p = +0.928$ nats，156 次有效参数更新）。在表 VI 的五项快速反事实探针上，13C 较 13B 在**全部五项指标上均持平或提升**——`OCRBench` 探针由 40.62% 升至 **43.75%**（场景文本 VQA 由 0% 升至 14.29%）、`POPE-Random` 由 90.00% 升至 **92.50%**、`RealWorldQA` 探针由 47.50% 升至 **52.50%**、`颜色×形状` 净增益达 **+100.0%**，且留出校准集的 `media_advantage` 提升 **+0.008**。但因原有自动门控沿用了 SFT 阶段的教师强制交叉熵上界（`loss_ratio <= 1.04`），13C 因 5.13% 的参考困惑度微升（`loss_ratio = 1.0513`）被自动回滚至 13B。这一现象与既有 RLVR 研究一致 [32]：在线强化学习在锐化贪婪决策准确率的同时，通常会小幅抬高离线 SFT 参考答案的交叉熵，因此不应仅以教师强制 NLL 作为 RLVR 晋升的否决条件。

**表 VI**  
*Stage-13 内部快速反事实探针逐阶段演进（真实图像准确率 / 真图 − 空白净增益）*

| 反事实探针赛道 | `pre_13`（Stage-12） | `post_13a`（9h `B=8`） | `post_13b`（全 36L+B50） | `post_13c`（`VC-GRPO`） |
| :--- | :---: | :---: | :---: | :---: |
| **颜色×形状（`n=8`）** | 0.0%（`+0.0%`） | 100.0%（`+87.5%`） | 100.0%（`+87.5%`） | **100.0%（`+100.0%`）** |
| **OCRBench 探针（`n=32`，原问）** | 31.25%（`+31.25%`） | 31.25%（`+31.25%`） | 40.62%（`+40.62%`） | **43.75%（`+43.75%`）** |
| **POPE-Random（`n=40`）** | 50.0%（`+17.5%`） | **95.0%（`+45.0%`）** | 90.0%（`+40.0%`） | 92.5%（`+42.5%`） |
| **POPE-Adv（`n=40`）** | 70.0%（`+22.5%`） | **80.0%（`+30.0%`）** | 77.5%（`+27.5%`） | 77.5%（`+27.5%`，F1 80.0%） |
| **RealWorldQA 探针（`n=40`）** | 52.5%（`+7.5%`） | **60.0%（`+7.5%`）** | 47.5%（`-2.5%`） | 52.5%（`+2.5%`） |

### C. 开放式解释偏简与密集文档 OCR 受限的直接成因
当前两项最突出的能力短板均可追溯至具体的数据与分辨率配置：（1）为在 24 GB 显存下最大化吞吐，Stage-13A/13B 中超过 85% 的视觉样本采用了极简标签（`Yes`/`No`、选项字母、短数字），而 `DOCCI` 稠密长描述仅占视觉语料的 2.7%（1,494 / 54,741），致使模型形成了强短答条件反射；（2）Phase 13A 将图像分辨率限制在 512 px（$\le 256$ 视觉 token）以实现 7,065 样本/小时的高吞吐，尽管 Phase 13B 恢复了 896 px 分辨率并将规则印刷词识别推至 40.62%–43.75%（未见合成词 66.67% 精确 / 86.67% 字符相似度），但 5 组 `OCRBench`（5.00%）中的密集文档问答、关键信息抽取（KIE）与手写数学公式（HMER）仍需要更高的视觉 token 上限与更充分的专项数据覆盖。

### D. 有限 Token 预算下的多任务权衡与总体规模缺口
Stage-13 将 26.37M 新增 Token 重点投向视觉与三维感知，在 8 基准宏平均（+6.40%–7.15%）、`POPE`（+23.33%）、`MMBench-CN`（+18.33%）、`3DSRBench`（+7.50%）与 `CV-Bench 3D`（+4.17%）上带来显著增益，但也伴随纯文本与高难综合推理赛道的小幅回撤（`MMMU val` -4.63%、`GPQA-Diamond` -2.02%、`Terminal-Bench 4.1` -9.52%）。截至 Stage-13 结束，现行 `RadioMultiLayerGELUProjector` 谱系累计仅训练了 **约 35.8M tokens**（Stage-7 起全谱系累计 114.2M tokens），仍仅为 **LLaVA-1.5**（约 1.0B）[1] 的约 $1/28$、**InternVL 2.5**（120B）[4] 的约 $1/3{,}350$，以及 **Qwen2.5-VL**（约 4.0T）[3] 的不足 $1/110{,}000$。从 Stage-12（9.4M）到 Stage-13（35.8M）的单调改善表明，当前阶段仍处于数据与算力高度欠饱和区间。

### E. 洁净环境部署与上游兼容性发现
在远程 Colab A100 洁净环境执行 `snapshot_download` 重载验证时，发现两项具有工程价值的部署细节：（1）上游 `modeling_spark.py` 将 `_tied_weights_keys` 声明为 Python `list` 而非 `dict`，在 `transformers >= 5.0`（如 Colab 默认 5.22.1）下会触发 `AttributeError`，固定 `transformers==4.57.3` 可稳定解决；（2）发布清单 `MODEL_RELEASE.json` 须剔除 `.cache` 与 `__pycache__` 目录并对可选的 `SyncCraft` 依赖施加延迟保护，以确保外部环境在 Python 3.11–3.13 下无障碍加载（已在代码提交 `51fdb77` 与当前 Hugging Face 正式仓库中修复）。

---

## 六、 下一阶段可执行方向

基于上述实证结果，后续工作可划分为面向现有工作站的近期精调（**Stage-14**）与面向多卡集群的中长期扩展两层路径：

### A. 近期单机精调方向（Stage-14，利用现有双卡工作站 1–2 天）
1. **修正 `VC-GRPO` 晋升门控并引入人类视觉反馈：** 将单一教师强制 CE 阈值替换为“反事实视觉增益 + 重复度健康度 + 宽松 KL 约束”的综合门控，并利用已建成的 `scripts/collect_user_vl_feedback.py` 收集真实交互纠错对，接入交错式 DPO 更新 [33]。
2. **提升稠密描述与“先解释后作答”视觉推理占比：** 将 `DOCCI` 全量（15K）[34]、`PixMo-Cap` [5] 及结构化“观察证据 → 推理分析 → 最终结论”长响应样本占比由不足 3% 提升至 35%–40%，以系统改善自由场景描述与多句图像解释能力。
3. **896 px 高分辨率文档 OCR 恢复与推理防遗忘回放：** 在 896 px 分辨率下重点补充 `DocVQA`、`ChartQA` [35]、`TextVQA`、KIE 与手写公式数据，并提高 `MMMU`、科学推理与终端任务的回放比例，在保持 `POPE` 与三维空间表现的同时恢复 `GPQA-Diamond` 与 `Terminal-Bench 4.1`。

### B. 多卡集群扩展规划（面向通用生产级 VLM）
若需将已验证的 6.05B 架构进一步推进至具备流畅视觉解释、密集文档阅读与强多模态推理的成熟水平，建议在 **8×–32× H100/B200** 集群上按表 VII 分四阶段扩展，并同步实施两项轻量架构改动：（a）仅对视觉前缀区间解除 `sliding_window=512` 掩码，以支持 1,024–2,048 动态多切片高分辨率文档输入；（b）在中训练阶段全量解冻 32 层 `C-RADIOv4-H`。

**表 VII**  
*建议的多卡集群扩展路线图（目标算力：8×–32× H100/B200）*

| 阶段 | 可训模块范围 | 多模态 Token 预算 | 核心数据配比侧重 | 估算 H100-h | 退出验证门控指标 |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **一、 高熵稠密视觉对齐** | FP32 `GELU` 投影器 + 前 12 层 `self_attn` | **15B–25B** | 50% 稠密描述（`PixMo-Cap` [5]、`DOCCI` [34]）、35% 多语种/文档 OCR、15% 空间与属性锚定 | 350–600 h | 未见 OCR $\ge 85\%$；颜色×形状 $\ge 95\%$；具备流畅多句看图描述 |
| **二、 全分辨率中训练** | 全量 6.05B + **全 32 层 `C-RADIOv4-H`**（支持至 2,048 视觉 token） | **150B–250B** | 35% 中英图文与科学图表、25% 文档/表格/KIE/HMER、20% 三维/视频空间、20% STEM 与百科（填充 `En1B`） | 2,800–4,800 h | `AI2D` $\Delta_{\text{vis}} \ge +15\%$；`RealWorldQA` $\ge 65\%$；5 组 `OCRBench` $\ge 650/1000$ |
| **三、 可解释多模态 SFT** | 全模型（`BF16`/`FP8 COAT` + `Muon`/`AdamW`） | **8B–15B** | 先解释后作答 VQA、中英双语思维链、`PRISM-Omni`、零重叠锚定、编译验证 `Block-DSL v4.1` | 250–450 h | `POPE` 三分集 $\ge 88\%$；`CV-Bench 3D` $\ge 75\%$；`3DSRBench` $\ge 65\%$；`v28` 编译 $\ge 90\%$ |
| **四、 可验证多模态 RLVR + DPO** | 全策略（`VC-GRPO` [32] + 人类反馈 DPO [33]） | **2B–5B 采样** | 可验证数理/视觉推理（`ViRL39K` [37] 等）、终端代理与人类偏好纠错对 | 600–1,200 h | `GPQA-Diamond` $\ge 55\%$；`HLE-Diamond` $\ge 15\%$；经核实的像素因果解释 |

---

## 七、 总结

截至 Stage-13 结束，**Spark-X2.6-4B-Loop48-En1B-VL** 已从 Stage-12 仅在印刷体 OCR 上打通视觉通路的状态，演进为在物体存在性（`POPE` 86.11%–89.44%，较错配真图净增益 +32.78%–36.66%）、科学图表（`AI2D` 67.50%，正向反事实增益 +3.75%）、三维与多视角空间推理（`BLINK` 49.07%、`CV-Bench 3D` 60.00%、`3DSRBench` 51.67%）、中文理解（`MMBench-CN` 51.67%）、属性绑定（`颜色×形状` 100.0%）及未见词 OCR（66.67% 精确 / 86.67% 字符相似度）上均经反事实检验具备真实视觉因果响应的多模态模型，且 FP8+q4g128 量化保持无损表现（8 基准宏平均 44.83% vs. BF16 44.15%）。与此同时，开放式长句视觉解释尚显简略、密集文档/手写公式 OCR 仍处于低位以及有限预算下的纯文本小幅权衡回撤，清晰界定了当前 35.8M 投影器 Token 规模的客观边界，为 Stage-14 近期优化与后续多卡集群扩展提供了明确、可复核的实证依据。

---

## 参考文献（IEEE 标准格式）

[1] H. Liu, C. Li, Y. Li, and Y. J. Lee, "Improved baselines with visual instruction tuning," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2024, pp. 26296–26306. (arXiv:2310.03744)

[2] P. Wang *et al.*, "Qwen2-VL: Enhancing vision-language model's perception of the world at any resolution," *arXiv preprint arXiv:2409.12191*, 2024.

[3] S. Bai *et al.*, "Qwen2.5-VL technical report," *arXiv preprint arXiv:2502.13923*, 2025.

[4] Z. Chen *et al.*, "Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling," *arXiv preprint arXiv:2412.05271*, 2024.

[5] M. Deitke *et al.*, "Molmo and PixMo: Open weights and open data for state-of-the-art vision-language models," *arXiv preprint arXiv:2409.17146*, 2024.

[6] L. Beyer *et al.*, "PaliGemma: A versatile 3B VLM for transfer," *arXiv preprint arXiv:2407.07726*, 2024.

[7] J. Hoffmann *et al.*, "Training compute-optimal large language models," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 35, 2022, pp. 30016–30030. (arXiv:2203.15556)

[8] M. Ranzinger, G. Heinrich, J. Kautz, and P. Molchanov, "AM-RADIO: Agglomerative vision foundation model — reduce all domains into one," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2024, pp. 12490–12500. (arXiv:2312.06709)

[9] G. Heinrich *et al.*, "RADIOv2.5: Improved baselines for agglomerative vision foundation models," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2025. (arXiv:2412.07679)

[10] M. Tschannen *et al.*, "SigLIP 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features," *arXiv preprint arXiv:2502.14786*, 2025.

[11] M. Dehghani, S. Gouws, O. Vinyals, J. Uszkoreit, and Ł. Kaiser, "Universal transformers," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2019. (arXiv:1807.03819)

[12] R.-J. Zhu *et al.*, "Scaling latent reasoning via looped language models," *arXiv preprint arXiv:2510.25741*, 2025.

[13] Z. Xie *et al.*, "mHC: Manifold-constrained hyper-connections," *arXiv preprint arXiv:2512.24880*, 2025.

[14] X. Cheng *et al.*, "Conditional memory via scalable lookup: A new axis of sparsity for large language models," *arXiv preprint arXiv:2601.07372*, 2026.

[15] H. Zhu *et al.*, "APOLLO: SGD-like memory, AdamW-level performance," in *Proc. Mach. Learn. Syst. (MLSys)*, 2025. (arXiv:2412.05270)

[16] H. Xi *et al.*, "COAT: Compressing optimizer states and activation for memory-efficient FP8 training," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2025. (arXiv:2410.19313)

[17] J. Liu *et al.*, "Muon is scalable for LLM training," *arXiv preprint arXiv:2502.16982*, 2025.

[18] Y. Li, Y. Du, K. Zhou, J. Wang, W. X. Zhao, and J.-R. Wen, "Evaluating object hallucination in large vision-language models," in *Proc. Conf. Empir. Methods Nat. Lang. Process. (EMNLP)*, 2023, pp. 292–305. (arXiv:2305.10355)

[19] Y. Liu *et al.*, "On the hidden mystery of OCR in large multimodal models," *arXiv preprint arXiv:2305.07895*, 2023.

[20] L. Chen *et al.*, "Are we on the right way for evaluating large vision-language models?" in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 37, 2024. (arXiv:2403.20330)

[21] S. Tong *et al.*, "Cambrian-1: A fully open, vision-centric exploration of multimodal LLMs," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 37, 2024. (arXiv:2406.16860)

[22] A. Kembhavi, M. Salvato, E. Kolve, M. Seo, H. Hajishirzi, and A. Farhadi, "A diagram is worth a dozen images," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2016, pp. 235–251.

[23] X. Fu *et al.*, "BLINK: Multimodal large language models can see but not perceive," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2024. (arXiv:2404.12390)

[24] X. Yue *et al.*, "MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert AGI," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2024, pp. 9556–9567. (arXiv:2311.16502)

[25] X. Yue *et al.*, "MMMU-Pro: A more robust multi-discipline multimodal understanding benchmark," *arXiv preprint arXiv:2409.02813*, 2024.

[26] Y. Liu *et al.*, "MMBench: Is your multi-modal model an all-around player?" in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2024. (arXiv:2307.06281)

[27] W. Ma, H. Chen, G. Zhang, C. M. de Melo, A. Yuille, and J. Chen, "3DSRBench: A comprehensive 3D spatial reasoning benchmark," in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2025. (arXiv:2412.07825)

[28] D. Rein *et al.*, "GPQA: A graduate-level Google-proof Q&A benchmark," *arXiv preprint arXiv:2311.12022*, 2023.

[29] L. Phan *et al.*, "Humanity's last exam," *arXiv preprint arXiv:2501.14249*, 2025.

[30] J. Wei *et al.*, "Measuring short-form factuality in large language models," *arXiv preprint arXiv:2411.04368*, 2024.

[31] xAI, "Grok-1.5 Vision Preview and RealWorldQA Benchmark," Technical Blog, Apr. 2024. [Online]. Available: https://x.ai/blog/grok-1.5v

[32] Z. Shao *et al.*, "DeepSeekMath: Pushing the limits of mathematical reasoning in open language models," *arXiv preprint arXiv:2402.03300*, 2024.

[33] R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, and C. Finn, "Direct preference optimization: Your language model is secretly a reward model," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 36, 2023. (arXiv:2305.18290)

[34] Y. Onoe *et al.*, "DOCCI: Descriptions of connected and contrasting images," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2024. (arXiv:2404.19753)

[35] A. Masry, D. X. Long, J. Q. Tan, S. Joty, and E. Hoque, "ChartQA: A benchmark for question answering about charts with visual and logical reasoning," in *Findings Assoc. Comput. Linguist. (ACL)*, 2022, pp. 2263–2279. (arXiv:2203.10244)

[36] P. Lu *et al.*, "Learn to explain: Multimodal reasoning via thought chains for science question answering," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 35, 2022. (arXiv:2209.09513)

[37] H. Wang, C. Qu, Z. Huang, W. Chu, F. Lin, and W. Chen, "VL-Rethinker: Incentivizing self-reflection of vision-language models with reinforcement learning," *arXiv preprint arXiv:2504.08837*, 2025.
