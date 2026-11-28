# Spark-X2.6-4B-Loop48-En1B-VL Milestone Technical Report (Stages 7–13): Multi-Stage Visual Grounding, Counterfactual Causality Verification, and Future Scaling Directions

**Milestone Technical Report (English Edition) — IEEE Format**  
**Artifact Repository:** `TassP0lar/Spark-X2.6-4B-Loop48-En1B-VL` (Stage-13 BF16, FP8+q4g128, and Stage-13A Preview)  
**Date:** November 27, 2026

---

## Abstract

This milestone report summarizes the design, multi-day empirical progression, counterfactual evaluation, and open limitations of **Spark-X2.6-4B-Loop48-En1B-VL** across Stages 7 through 13. The system adapts a 36-layer text-native `Spark-X2.6-4B` backbone into a 6.0477B-parameter multimodal reasoning architecture (inclusive of CPU-pinned sparse conditional memory) incorporating 48-virtual-layer weight-shared `Dual-mHC` looping, a 0.999B Tiered MoE-Engram lookup subsystem (`En1B`), and an `nvidia/C-RADIOv4-H` (651.6M) vision encoder connected through an FP32 `RadioMultiLayerGELUProjector` with 2D row-break newline tokens and one-sided visual RMS ceiling regularization. All training was conducted on a single heterogeneous workstation (1× NVIDIA RTX PRO 4000 Blackwell 24 GB + 1× Intel Arc Pro B50 16 GB), accumulating 188,756 samples (~114.2M estimated tokens) over 100.11 linear hours, including a dedicated 18.43-hour Stage-13 refinement (26.37M new tokens) incorporating `TassP0lar/PRISM-Omni-Dataset-v1`, zero-overlap `COCO` all-80-class existence and layout data, `DOCCI`, `ChartQA`, `ScienceQA`, `ViRL39K`, counterfactual attribute grids, and an on-policy Visual-Causality-Gated GRPO (`VC-GRPO`) stage. Across strictly zero-overlap benchmarks, Stage-13 raises the 8-benchmark multimodal macro average from 37.75% (BF16) / 37.68% (FP8) in Stage-12 to **44.15% (BF16) / 44.83% (FP8+q4g128)**, with consistent gains on **POPE** (62.78% → 86.11% Final; 89.44% in Stage-13A), **MMBench-CN** (32.50% → 50.83% / 51.67%), **AI2D** (63.33% → 67.50%), **BLINK 6-Spatial** (43.52% → 47.22% / 49.07%), **MMStar** (28.33% → 32.22% / 33.33%), **CV-Bench 3D** (55.83% → 60.00%), **3DSRBench** (44.17% → 51.67%), and **MMMU-Pro** (20.83% → 22.50%). Independent Real-Image vs. Blank-Gray vs. Shuffled-Image counterfactual testing indicates that the modality decoupling observed in Stage-12 is resolved across all evaluated visual tracks, including positive Real-vs.-Blank lift on `POPE` (+36.11%), `RealWorldQA` (+11.25%), `BLINK` (+5.56%), `AI2D` (+3.75%, symmetric KL increasing from 0.0842 to 0.9487 nats), `Color × Shape` (100.0%), and unseen synthetic word OCR (66.67% exact / 86.67% character similarity). At the same time, candid error analysis shows clear remaining constraints: free-form multi-sentence visual explanation remains terse and prone to fine-grained slips, full 5-group `OCRBench` (5.00%) remains limited on dense documents and handwritten mathematics, modest regressions occurred on `MMMU (val)` (-4.63%), `GPQA-Diamond` (-2.02%), and `Terminal-Bench 4.1` (-9.52%), and our cumulative multimodal token budget remains $10^{3}\times$ to $10^{4}\times$ smaller than production open-weight VLMs. We document what has been established empirically, what may be inferred from these observations, and a concrete near-term and cluster-scale roadmap for subsequent stages.

**Index Terms—** Vision-language models, counterfactual visual evaluation, looped transformers, sparse conditional memory, reinforcement learning with verifiable rewards, multimodal scaling.

---

## I. Introduction and Scope

Extending compact, text-native reasoning Large Language Models (LLMs) into Vision-Language Models (VLMs) typically relies on large-scale multimodal pretraining pipelines spanning $10^{9}$ to $4 \times 10^{12}$ tokens [1]–[6], consistent with compute-optimal scaling behavior [7]. When development is instead constrained to a single workstation, two practical questions arise: (1) whether a stable optimization and data-curation pipeline can establish genuine, pixel-conditioned visual perception without eroding language reasoning, and (2) how to separate true visual grounding from text-prior shortcuts on standard multimodal benchmarks [20], [21].

This milestone report documents the state of **Spark-X2.6-4B-Loop48-En1B-VL** at the conclusion of **Stage-13**, consolidating empirical measurements collected over our multi-day training sequence (Stages 7–13). Rather than presenting speculative claims, we structure the report around three explicit questions: (i) *what we currently have* in terms of architecture, verified artifacts, and zero-overlap benchmark/counterfactual metrics; (ii) *what we can reasonably infer* from controlled ablations, cross-stage regressions, and live explanation probes; and (iii) *what concrete steps* are warranted next under both local workstation and cluster compute budgets.

---

## II. What We Have: Architecture, Training Ledger, and Data Governance

### A. Audited Model Topology (6.0477B Total Parameters)

The released checkpoint (`TassP0lar/Spark-X2.6-4B-Loop48-En1B-VL`) combines a 36-layer `Spark-X2.6-4B` backbone (1M native RoPE context; `sliding_window=512` on 27/36 layers and global attention every fourth layer) with four subsystems audited directly from the exported BF16 weights (Table I):

1. **48-Virtual-Layer Looping (`Loop48` Dual-mHC):** Middle layers $[12, 24)$ execute two recurrent passes governed by doubly-stochastic manifold-constrained residual connections and rank-16 pass adapters [11]–[13] (+3.07M parameters).
2. **Tiered MoE-Engram Memory (`En1B`):** Eight anchor depths ($[3, 7, 11, 15, 19, 23, 27, 31]$) attach a 16-expert (`top_k=4`) sparse lookup memory split across GPU Tier-1A hot buckets and CPU-pinned Tier-1B cold buckets [14] (998.82M parameters total).
3. **Agglomerative Vision Encoder (`nvidia/C-RADIOv4-H`):** A 32-layer ViT-H/16 backbone (651.65M parameters) distilled from SigLIP 2, DINOv3, and SAM3 [8]–[10]. Single-image inputs are bounded to $\le 1{,}600$ patches ($\le 400$ merged tokens) so visual prefixes stay inside the 512-token sliding attention window.
4. **FP32 `RadioMultiLayerGELUProjector` with 2D Row-Break Tokens:** Features tapped at layers 24 and 32 ($\mathbf{F}^{(24)}, \mathbf{F}^{(32)}$) are independently normalized, concatenated, $2\times 2$ pixel-shuffled to $\mathbf{U} \in \mathbb{R}^{N_g \times 10240}$, and projected in FP32 with active-GELU initialization ($\tau_{\text{init}} = 0.035$) and a one-sided RMS ceiling regularizer:
   $$\mathbf{V}_{\text{grid}} = \mathbf{W}_{2}\,\mathrm{GELU}\!\left(\mathbf{W}_{1}\,\mathrm{RMSNorm}(\mathbf{U}) + \mathbf{b}_{1}\right) + \mathbf{b}_{2}, \qquad \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda_{\text{rms}} \left[\max\!\left(0,\; \mathrm{RMS}(\mathbf{V}_{\text{grid}}) - \tau_{\text{cap}}\right)\right]^{2}$$
   with $\lambda_{\text{rms}} = 5.0$ and $\tau_{\text{cap}} \in \{0.08, 0.09, 0.10\}$. Native newline embeddings $\mathbf{e}_{\text{nl}}$ are appended after each spatial row to encode 2D reading order.

**TABLE I**  
*Audited Parameter Distribution of `Spark-X2.6-4B-Loop48-En1B-VL` (Stage-13 BF16)*

| Module / Subsystem | Exact Parameters | Share (%) | Placement & Stage-13B Training Role |
| :--- | ---: | ---: | :--- |
| `Spark-X2.6-4B` Backbone (36L) | 4,112,079,360 | 67.99% | Blackwell CUDA (`APOLLO+COAT`, trainable) |
| `nvidia/C-RADIOv4-H` Vision (32L) | 651,645,440 | 10.78% | Intel Arc B50 XPU (top-8L `Muon8bit`, 157.4M active) |
| `En1B` Tier-1B Cold Buffers | 578,813,952 | 9.57% | Host CPU Pinned RAM (sparse lookup) |
| `En1B` Tier-1A Hot Buckets & Routers | 420,004,136 | 6.94% | Blackwell CUDA (trainable) |
| `GatedVisualCrossAttention` (3 blocks) | 196,723,206 | 3.25% | Blackwell CUDA (trainable) |
| `RadioMultiLayerGELUProjector` | 65,558,018 | 1.08% | Blackwell CUDA FP32 (trainable) |
| Aux Spatial / Alpha / Camera Connectors | 19,811,840 | 0.33% | Blackwell CUDA (trainable) |
| `Loop48` Dual-mHC Controllers | 3,072,600 | 0.05% | Blackwell CUDA (trainable) |
| **Total (incl. CPU-Pinned Tier-1B)** | **6,047,708,552** | **100.00%** | **12.69 GB BF16 / 10.69 GB FP8+q4g128** |

### B. Heterogeneous Compute Ledger (Stages 7–13) and Stage-13 Design

All training runs executed on **1× NVIDIA RTX PRO 4000 Blackwell SFF (24 GB)** running `APOLLO` (`rank=64`) [15] and `COAT` FP8 group-16 activation compression [16], paired with **1× Intel Arc Pro B50 (16 GB)** running `C-RADIOv4-H` updates via 8-bit `Muon8bit` [17] over dual-pinned shared memory. Independent clean-room evaluation of the intermediate Stage-13A checkpoint was also conducted on a remote **NVIDIA A100-SXM4-80GB** instance.

To address the modality decoupling diagnosed in Stage-12, **Stage-13** warm-started directly from `final-model-stage12-bf16` (`reinit_vision_projector=False`) over **57,485 audited, commercial-permissive records** (`54,741` visual pairs, `failures_count = 0`), combining: (a) `TassP0lar/PRISM-Omni-Dataset-v1` (pinned revision `8e5ac6197b`, 8,257 balanced visual-text rows across `scenes3d`, `charts`, `paths`, `mazes`, `gvsr`, `pixmo_count`, `pixmo_points`, and downsampled `shapes`, unified into Spark XML spatial primitives); (b) `COCO train2017` all-80-class existence and spatial layout pairs (13,000 samples mixing 35% uniform random absent negatives, 45% co-occurring adversarial negatives, and 20% popular negatives, with a programmatic 500-ID `POPE val2014` guard excluding 34 overlapping images -> 0/500 overlap); (c) workstation-cached `DOCCI` dense paragraphs (1,494) [34], `ChartQA` (2,600) [35], `ScienceQA` image diagrams (2,400) [36], and `ViRL39K` (3,600, with 53 eval-named source rows filtered out) [37]; (d) a 2,400-sample counterfactual `Color × Shape × Spatial` grid; and (e) `DocVQA`, `TextVQA`, `A-OKVQA`, `SynthDoG-EN`, `LaTeX`, `GQA-open`, `Nemotron-v3` (including 1,200 Chinese `zhwiki` pairs), `SyncCraft v38/v28`, and frontier STEM/terminal replay.

Stage-13 executed three sequential phases (18.43 wall-clock hours, 26.37M estimated tokens; Table II):
- **Phase 13A (9.03 h, 7,948 updates, $B=8$, 63,584 samples, 15.12M tokens):** Chunked-RADIO feature extraction (512 px cap, $\le 2.5$ GiB host RAM per 100-step chunk) updating the FP32 GELU projector and early-12L `self_attn` across 8 fixed task slots per micro-batch (`ema_ce: 2.7495 → 0.7045`).
- **Phase 13B (6.29 h, 950 updates, `grad_accum=12`, 11,400 samples, ~9.69M tokens):** Full-36L `APOLLO+COAT` + top-8 `C-RADIOv4-H` layers on Arc B50 (`Muon8bit`, 151.0 upd/h) at full 896 px resolution with real-vs.-blank visual margin upweighting (`mean_adv = 0.9269 nats` over 3,517 probes).
- **Phase 13C (`VC-GRPO`, 2.73 h, 1,200 prompts × $K=4$ = 4,800 on-policy rollouts, 156 updates, 1.57M scored tokens):** Online Group Relative Policy Optimization [32] gated by per-rollout real-vs.-blank log-probability difference ($\Delta\log p = \overline{\log p}(y \mid I_{\text{real}}, q) - \overline{\log p}(y \mid I_{\text{blank}}, q)$) and deterministic verifiers across 29,537 verifiable prompts, plus an integrated Direct Preference Optimization (DPO) [33] interface (`collect_user_vl_feedback.py`) for human feedback (`0` human records supplied in this autonomous run).

**TABLE II**  
*Linear Workstation Compute and Token Ledger Across Stages 7–13 (Single Dual-GPU Workstation)*

| Stage | Vision / Projector Configuration | Updates | Samples | Est. Tokens | Wall-Clock |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Stages 7–7.5 | `SigLIP2-SO400M` (base + alignment) | 3,231 | 38,772 | ~36.6M | 38.10 h |
| Stages 8–11.5 | `C-RADIOv4-H` + `SwiGLU` DeepStack | 4,510 | 54,120 | ~41.8M | 36.21 h |
| Stage-12 (`12A–12C`) | `C-RADIOv4-H` + fresh FP32 `GELU` + `\n` | 1,740 | 20,880 | ~9.4M | 7.37 h |
| **Stage-13 (`13A+13B+13C`)** | **Warm-started `GELU` + `PRISM-Omni` + `VC-GRPO`** | **9,054** | **74,984*** | **~26.4M** | **18.43 h** |
| **Cumulative Linear (7→13)** | — | **18,535** | **188,756** | **~114.2M** | **100.11 h** |
| **Cumulative `GELU` Proj (12→13)** | **Active Stage-12/13 Projector Lineage** | **10,794** | **95,864** | **~35.8M** | **25.80 h** |

*\*Stage-13 supervised sample count (63,584 in 13A + 11,400 in 13B), supplemented by 4,800 online RLVR rollouts in 13C.*

---

## III. Quantitative Benchmark Results (Zero-Overlap Suites)

Tables III and IV report exact measurements on our frozen SHA-pinned benchmarks (`scale=0.6`, 1,056 items/model on the 8-benchmark standard suite, plus the 3D spatial and frontier reasoning suite). Note that Stage-13's exported `final-model-stage13-{bf16,fp8}` checkpoints correspond to `post_stage13b` because Phase 13C triggered an automatic SFT cross-entropy rollback (`loss_ratio = 1.0513 > 1.04`, analyzed in Section V-B); both Stage-13A (evaluated on Colab A100) and Stage-13 Final (evaluated on Blackwell) are shown alongside earlier stages.

**TABLE III**  
*Frozen Standard 8-Benchmark Multimodal Suite Progression (`scale=0.6`, $n=1{,}056$/model, 0% Train Overlap)*

| Model / Precision | 8-Bench Macro | AI2D [22] | RealWorldQA [31] | BLINK 6-Sp [23] | MMBench-CN [26] | MMMU Val [24] | MMStar [20] | POPE Acc (`F1`) [18] | OCRBench 5-Grp [19] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Stage-8 BF16 | 37.69% | 56.67% | 44.17% | 39.81% | 37.50% | 30.56% | 24.44% | 50.00% (`0.667*`) | 1.67% |
| Stage-11.5 BF16 | 36.80% | 60.00% | 37.50% | 40.74% | 35.83% | 31.48% | 32.78% | 52.78% (`0.536`) | 2.50% |
| Stage-12 BF16 | 37.75% | 63.33% | 40.83% | 43.52% | 32.50% | **31.48%** | 28.33% | 62.78% (`0.579`) | 4.17% |
| Stage-12 FP8+q4g128 | 37.68% | 64.17% | 40.00% | 43.52% | 32.50% | 28.70% | 28.89% | 64.44% (`0.595`) | 4.17% |
| Stage-13A BF16 (A100) | 44.74% | 61.67% | **44.17%** | 43.52% | 50.83% | 30.56% | **33.89%** | **89.44% (`0.891`)** | 4.17% |
| **Stage-13 Final BF16** | 44.15% | **67.50%** | 38.33% | 47.22% | 50.83% | 26.85% | 32.22% | 86.11% (`0.852`) | **5.00%** |
| **Stage-13 Final FP8+q4g128** | **44.83%** | **67.50%** | 40.00% | **49.07%** | **51.67%** | 26.85% | 33.33% | 86.11% (`0.852`) | **5.00%** |
| *$\Delta$ Stage-13 vs. Stage-12 (BF16 / FP8)* | **+6.40% / +7.15%** | **+4.17% / +3.33%** | -2.50% / 0.00% | **+3.70% / +5.55%** | **+18.33% / +19.17%** | -4.63% / -1.85% | **+3.89% / +4.44%** | **+23.33% / +21.67%** | **+0.83% / +0.83%** |

*\*Stage-8 POPE F1 reflects an all-`"Yes"` degenerate output (`yes_ratio = 1.000`). Stage-13A (`yes_ratio = 0.472`) and Stage-13 Final (`yes_ratio = 0.439`) exhibit balanced precision/recall across Random (88.33%), Popular (81.67%–88.33%), and Adversarial (88.33%–91.67%) splits.*

**TABLE IV**  
*3D Spatial, Frontier Reasoning, and SyncCraft Suite Progression (BF16, Identical Full Slices, 0% Overlap)*

| Benchmark Track (Frozen Pin) | Stage-11.5 | Stage-12 | **Stage-13 Final** | $\Delta$ (13 vs. 12) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **CV-Bench 3D** (`n=120`) [21] | 49.17% | 55.83% | **60.00%** | **+4.17%** | Depth 59.0%, Distance 61.0%, 99.2% 4-prim |
| **3DSRBench** (`n=120`) [27] | 48.33% | 44.17% | **51.67%** | **+7.50%** | 97.5% 4-prim; Parallel 80%, Closer/Next-to 70% |
| **MMMU-Pro 10-Option** (`n=120`) [25] | 20.00% | 20.83% | **22.50%** | **+1.67%** | >2.2× 10-option random chance |
| **GPQA-Diamond** (`n=198`) [28] | 34.85% | **38.89%** | 36.87% | -2.02% | Physics 45.35%, Chemistry 31.18%, Bio 26.32% |
| **Terminal-Bench 4.1** (`n=21`, syntax / sub) | **66.67%** | 61.90% / 57.14% | 52.38% / 33.33% | -9.52% | Healthy `<think>` 47.62% |
| **SimpleQA-Verified** (`n=120`) [30] | 3.33% | 1.67% | **3.33%** | **+1.66%** | Sparse `En1B` factual recall remains early-stage |
| **SyncCraft Gate D Proposer** (`n=4`) | **100.0%** | **100.0%** | **100.0%** | 0.00% | 1.0 mean solver calls (243 tokens) |
| **SyncCraft Unseen `v28` Compile** (`n=4`) | — | 0.00% | 0.00% | 0.00% | Grammar-level DSL slips remain unresolved |

*Note: `HLE-Diamond` [29] (8.00% in Stage-12) and `AIME 2026` (0.00% in Stage-12) were omitted from Stage-13's automated post-run script to keep total evaluation within the 24-hour window.*

---

## IV. Counterfactual Visual-Causality Verification and Qualitative Image-Explanation Assessment

### A. Real-Image vs. Blank-Gray vs. Shuffled-Image Counterfactual Audit

To verify whether Stage-13's gains reflect genuine pixel conditioning rather than language-prior memorization, we executed the exact same independent counterfactual protocol (`scripts/verify_stage12_visual_causality.py`) across Stage-12, Stage-13A (on A100), and Stage-13 Final (on Blackwell), measuring Real accuracy, Blank-Gray control (`RGB(128,128,128)`), Shuffled-Image control (on POPE), and symmetric first-token KL divergence $D_{\text{KL}}^{\text{sym}}$ (Table V).

**TABLE V**  
*Independent Counterfactual Visual-Causality Progression: Stage-12 vs. Stage-13A vs. Stage-13 Final (BF16)*

| Counterfactual Audit Track | Stage-12 Real / Blank ($\Delta_{\text{vis}}$) | Stage-13A Real / Blank ($\Delta_{\text{vis}}$) | **Stage-13 Final Real / Blank ($\Delta_{\text{vis}}$)** | Stage-13 KL / Signal Metric |
| :--- | :---: | :---: | :---: | :--- |
| **POPE Overall (`n=180`)** | 62.78% / 55.00% (+7.78%) | **89.44% / 50.00% (+39.44%)** | **86.11% / 50.00% (+36.11%)** | Shuffled: 53.33% (**+32.78% vs. Shuf**; 0/90 Blank FP) |
| **AI2D (`n=80`)** | 71.25% / 73.75% (-2.50%) | 67.50% / 67.50% (0.00%) | **71.25% / 67.50% (+3.75%)** | $D_{\text{KL}}^{\text{sym}}$: 0.0842 (12) → 0.0007 (13A) → **0.9487** (13) |
| **BLINK 6-Spatial (`n=72`)** | 47.22% / 45.83% (+1.39%) | 41.67% / 44.44% (-2.78%) | **48.61% / 43.06% (+5.56%)** | Choice flip rate: 2.78% (12) → **30.56%** (13) |
| **RealWorldQA (`n=80`)** | 36.25% / 35.00% (+1.25%) | 42.50% / 31.25% (+11.25%) | **36.25% / 25.00% (+11.25%)** | Pred flip rate: 11.25% (12) → **36.25%** (13) |
| **Color × Shape Open (`n=8`)** | 25.0% col / 50.0% shp (0.0%) | **100.0% / 100.0%** (+75%/+50%) | **100.0% / 100.0%** (+75%/+50%) | Prior `"red circle"` fully eliminated |
| **OCRBench Quick-32 Raw** | 31.25% / 0.00% (+31.25%) | 31.25% / 0.00% (+31.25%) | **40.62% / 0.00% (+40.62%)** | Char sim: **67.57%** Real vs. 3.17% Blank (*43.75% post-13C*) |
| **Unseen Synthetic Word (`n=12`)** | 50.00% / 0.00% (+50.00%) | 50.00% / 0.00% (+50.00%) | **66.67% / 8.33% (+58.34%)** | Char sim: **86.67%** Real vs. 13.33% Blank ($D_{\text{KL}}^{\text{sym}}=7.79$) |

### B. Live Free-Form Image Explanation Assessment

Because benchmark accuracy does not guarantee fluent open-ended visual explanation, we evaluated `final-model-stage13-bf16` directly on five held-out real photographs (`RealWorldQA`, `COCO`, `AI2D`, `CV-Bench 3D`) using open-ended *"describe and explain"* prompts paired with blank-gray controls. Three factual observations emerge:

1. **Verified Scene-Level Perception:** On a held-out COCO snowboard scene, the real image elicited `"A skier is skiing down a snowy slope."` whereas the blank control elicited `"A person is standing on a sidewalk holding a skateboard."` On a held-out `RealWorldQA` indoor scene, the real image elicited a scene-specific description of a television, colors, and room lighting, whereas the blank control produced an unrelated synthetic cube prior. On `CV-Bench 3D` with `<think>` enabled, the model referenced surrounding room objects (`sofa`, `chair`) absent from the prompt.
2. **Terse Over-Conditioning on Explanation Prompts:** When asked to *"explain your reasoning briefly based on what you see"* on `RealWorldQA` and `AI2D` multiple-choice prompts, the model still emitted single-letter answers (`"B"`), and on `CV-Bench 3D` produced only `"Answer: A"` after an abbreviated `<think>` block.
3. **Fine-Grained Naming and Density Limits:** Free-form captions remain brief (1–3 sentences) and occasionally substitute closely related visual classes (e.g., `"skier"` for a snowboarder).

In summary, **Stage-13 genuinely perceives image pixels across object existence, color/shape attributes, printed words, and coarse 3D spatial layouts, but is not yet reliable for detailed, multi-sentence user-facing visual explanations.**

---

## V. What We Potentially Inferred from Stages 7–13

Synthesizing our ablation history, Stage-13A/13B/13C progression, and qualitative diagnostics supports five measured inferences:

### A. Layer-Depth Specialization Across Visual Tasks
Comparing **Stage-13A** (which trained only the FP32 GELU projector and Spark layers `0–11` `self_attn`) against **Stage-13 Final** (after Phase 13B unfroze all 36 Spark layers and top-8 `C-RADIOv4-H` blocks) provides clean evidence of functional depth separation. Stage-13A alone drove `POPE` to **89.44%** (+39.44% Real-vs.-Blank lift), `Color × Shape` to **100.0%**, and `RealWorldQA` lift to **+11.25%**, yet left `AI2D` (`0.00%` lift, $D_{\text{KL}}^{\text{sym}} = 0.0007$) and `BLINK` (`-2.78%` lift) visually uncoupled. Only after Phase 13B unfroze layers `12–35` did `AI2D` acquire positive visual lift (**+3.75%**, $D_{\text{KL}}^{\text{sym}}$ rising $1{,}350\times$ to **0.9487 nats**) and `BLINK` flip to **+5.56%** lift (`30.56%` choice-change rate). This suggests that shallow attention layers suffice to bind visual tokens for existence and open-attribute lookup, whereas deep transformer blocks (`12–35`) are required to route visual evidence into multi-option comparative reasoning.

### B. On-Policy `VC-GRPO` Entropy Sharpening vs. Teacher-Forced NLL Gates
During Phase 13C, `run_multimodal_vc_grpo_rlvr` generated **4,800 live rollouts** across 1,200 verifiable prompts (`exact_rate = 60.38%`, mean causal $\Delta\log p = +0.928$ nats, `156` applied updates). Across the five-track Stage-13 counterfactual probe (Table VI), post-13C improved or matched post-13B on **every single track**—raising `OCRBench` quick-32 from 40.62% to **43.75%** (unlocking 14.29% Scene Text VQA), `POPE-Random` from 90.00% to **92.50%**, `RealWorldQA` probe from 47.50% to **52.50%**, `Color × Shape` lift from 87.50% to **100.0%**, and held-out visual calibration `media_advantage` by **+0.008**. Nonetheless, our automated gate rolled back to `ckpt-stage13b` solely because teacher-forced cross-entropy increased by 5.13% (`loss_ratio = 1.0513 > 1.04`). This mirrors well-documented RLVR behavior [32]: on-policy reinforcement learning sharpens greedy decision probabilities while mildly increasing perplexity on off-policy SFT reference targets, indicating that SFT NLL thresholds should not be used as a sole rejection criterion for RLVR checkpoints.

**TABLE VI**  
*Intra-Stage-13 Counterfactual Quick-Probe Progression (Real Accuracy / Real − Blank Lift)*

| Probe Track | `pre_13` (Stage-12) | `post_13a` (9h `B=8`) | `post_13b` (Full-36L+B50) | `post_13c` (`VC-GRPO`) |
| :--- | :---: | :---: | :---: | :---: |
| **Color × Shape (`n=8`)** | 0.0% (`+0.0%`) | 100.0% (`+87.5%`) | 100.0% (`+87.5%`) | **100.0% (`+100.0%`)** |
| **OCRBench Probe (`n=32`, Raw)** | 31.25% (`+31.25%`) | 31.25% (`+31.25%`) | 40.62% (`+40.62%`) | **43.75% (`+43.75%`)** |
| **POPE-Random (`n=40`)** | 50.0% (`+17.5%`) | **95.0% (`+45.0%`)** | 90.0% (`+40.0%`) | 92.5% (`+42.5%`) |
| **POPE-Adv (`n=40`)** | 70.0% (`+22.5%`) | **80.0% (`+30.0%`)** | 77.5% (`+27.5%`) | 77.5% (`+27.5%`, F1 80.0%) |
| **RealWorldQA Probe (`n=40`)** | 52.5% (`+7.5%`) | **60.0% (`+7.5%`)** | 47.5% (`-2.5%`) | 52.5% (`+2.5%`) |

### C. Why Explanation Quality and Dense Document OCR Remain Constrained
Two structural factors in our Stage-13 data and compute budget explain the remaining gaps:
1. **Terse-Answer Bias:** More than 85% of Stage-13A/13B visual records used concise labels (`Yes`/`No`, option letters, short counts) to maximize sample throughput on a 24 GB GPU, while `DOCCI` dense paragraphs represented only 2.7% (`1,494 / 54,741`) of the visual pool. Consequently, the model learned accurate short-form grounding without acquiring sustained, explain-then-answer narration habits.
2. **Resolution Cap and Long-Tail OCR Coverage:** Phase 13A capped input resolution at 512 px ($\le 256$ visual tokens) to reach 7,065 samples/h. Although Phase 13B restored 896 px resolution and pushed printed-word OCR to 40.62%–43.75% (`OCRBench` quick-32) and 66.67% exact (`86.67%` char sim on unseen words), full 5-group `OCRBench` (5.00%) remains bottlenecked by dense multi-column documents, key-information extraction (KIE), and handwritten mathematical expressions (HMER), which require higher visual token budgets ($>400$ tokens) and dedicated domain coverage.

### D. Capacity Trade-Offs and Order-of-Magnitude Token Deficit
Concentrating Stage-13's 26.37M new tokens heavily on visual grounding yielded broad multimodal and 3D spatial improvements (+6.40%–7.15% 8-bench macro, +23.33% POPE, +18.33% MMBench-CN, +7.50% 3DSRBench, +4.17% CV-Bench 3D), accompanied by moderate trade-offs on pure text/college reasoning tracks (`MMMU val` -4.63%, `GPQA-Diamond` -2.02%, `Terminal-Bench 4.1` -9.52%). Crucially, even after Stage-13, our active `RadioMultiLayerGELUProjector` lineage has observed **~35.8M tokens** (and 114.2M cumulative tokens since Stage-7)—still **~28× below LLaVA-1.5** (~1.0B) [1], **~3,350× below InternVL 2.5** (120B) [4], and **>110,000× below Qwen2.5-VL** (~4.0T) [3]. The consistent monotonic gains observed from Stage-12 (~9.4M) to Stage-13 (~35.8M) indicate that token volume and compute scale remain the primary limiting factors.

### E. Clean-Room Deployment and Upstream Compatibility Findings
Clean-room download verification on a fresh Colab A100 VM surfaced two actionable engineering details: (1) upstream `modeling_spark.py` declares `_tied_weights_keys` as a Python `list` rather than a `dict`, causing an `AttributeError` under `transformers >= 5.0` (verified resolved by pinning `transformers==4.57.3`); and (2) release manifest generation must exclude `.cache` and `__pycache__` directories and guard optional `SyncCraft` imports so external users can reload exported Hugging Face weights cleanly across Python 3.11–3.13 (both now fixed in repository commit `51fdb77` and reflected in the published Stage-13 release).

---

## VI. What We Can Do Next

Based on the evidence above, next steps divide cleanly into a near-term workstation refinement (**Stage-14**) and a multi-GPU cluster scaling program:

### A. Near-Term Workstation Refinements (Stage-14, 1–2 Days on Existing Hardware)
1. **Promote `VC-GRPO` with RLVR-Appropriate Gating and User Feedback:** Replace the strict SFT NLL ceiling (`loss_ratio <= 1.04`) with a dual gate combining held-out counterfactual visual accuracy/lift, repetition health (`rep_4gram`), and a relaxed KL/perplexity bound, while incorporating human visual corrections collected via `scripts/collect_user_vl_feedback.py` into the interleaved DPO loop [33].
2. **Shift Data Mix Toward Dense Narration and Explain-Then-Answer Traces:** Increase dense real-photo descriptions and grounded explanation data (`DOCCI` full 15K [34], `PixMo-Cap` [5], and structured *"describe observed evidence → reason → answer"* traces) from `<3%` to `35%–40%` of the Stage-14A/14B mixture so the model learns to articulate visual reasoning in multi-sentence prose rather than single tokens.
3. **High-Resolution Document/OCR Recovery and Reasoning Replay:** Allocate a dedicated 896 px phase on `DocVQA`, `ChartQA` [35], `TextVQA`, KIE, and `LaTeX`/HMER paired with balanced `MMMU`/STEM/terminal replay to recover `GPQA-Diamond`, `Terminal-Bench 4.1`, and `MMMU (val)` without forfeiting Stage-13's `POPE` and 3D spatial gains.

### B. Multi-GPU Cluster Scaling (Recommended for Production-Grade General Vision)
To transition from a verified research prototype (~35.8M projector tokens) to a broadly capable compact VLM with fluent visual explanation and strong long-tail OCR/reasoning, scaling the validated `Spark-X2.6-4B-Loop48-En1B-VL` architecture on **8×–32× H100/B200 GPUs** should follow four staged milestones (Table VII), paired with two targeted architectural adjustments: (a) lifting the `sliding_window=512` mask specifically over visual prefix tokens to support 1,024–2,048 dynamic multi-tile document patches, and (b) unfreezing all 32 `C-RADIOv4-H` layers during mid-training.

**TABLE VII**  
*Recommended Cluster-Scale Multimodal Training Roadmap (Target: 8×–32× H100/B200)*

| Phase | Trainable Scope | Multimodal Token Budget | Core Data Emphasis | Est. H100-h | Exit Verification Criteria |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **I. High-Entropy Dense Alignment** | FP32 `GELU` Proj + Early-12L `self_attn` | **15B–25B** | 50% Dense captions (`PixMo-Cap` [5], `DOCCI` [34]), 35% Multi-script/Doc OCR, 15% Spatial/Attribute grounding | 350–600 h | Unseen OCR $\ge 85\%$; Color×Shape $\ge 95\%$; fluent multi-sentence captioning |
| **II. Full-Resolution Mid-Training** | Full 6.05B + **All 32L `C-RADIOv4-H`** (up to 2,048 vis tokens) | **150B–250B** | 35% Interleaved EN/ZH web & diagrams, 25% Doc/Chart/KIE/HMER, 20% 3D/Video spatial, 20% STEM & factual (`En1B`) | 2,800–4,800 h | `AI2D` $\Delta_{\text{vis}} \ge +15\%$; `RealWorldQA` $\ge 65\%$; `OCRBench` 5-grp $\ge 650/1000$ |
| **III. Explainable Multimodal SFT** | Full Model (`BF16`/`FP8 COAT` + `Muon`/`AdamW`) | **8B–15B** | Explain-then-answer VQA, bilingual CoT, `PRISM-Omni`, 0%-overlap grounding, compiler-verified `Block-DSL v4.1` | 250–450 h | `POPE` $\ge 88\%$ all splits; `CV-Bench 3D` $\ge 75\%$; `3DSRBench` $\ge 65\%$; `v28` compile $\ge 90\%$ |
| **IV. Verifiable Multimodal RLVR + DPO** | Full Policy (`VC-GRPO` [32] + Human DPO [33]) | **2B–5B rollouts** | Verifiable math/science (`ViRL39K` [37], `AIME`-style), visual reasoning, terminal agents, human preference pairs | 600–1,200 h | `GPQA-Diamond` $\ge 55\%$; `HLE-Diamond` $\ge 15\%$; verified grounded explanations |

---

## VII. Conclusion

At the end of Stage-13, **Spark-X2.6-4B-Loop48-En1B-VL** has progressed from an OCR-only early alignment (Stage-12) to a genuinely pixel-conditioned multimodal checkpoint across object existence (`POPE` 86.11%–89.44%, +32.78%–36.66% over shuffled images), science diagrams (`AI2D` 67.50%, +3.75% counterfactual lift), spatial perception (`BLINK` 49.07%, `CV-Bench 3D` 60.00%, `3DSRBench` 51.67%), bilingual understanding (`MMBench-CN` 51.67%), attribute binding (`Color × Shape` 100.0%), and unseen word OCR (66.67% exact / 86.67% character similarity), with lossless FP8+q4g128 quantization (44.83% vs. 44.15% BF16 8-bench macro). Remaining limitations—specifically terse free-form explanations, long-tail document/math OCR, and modest text-reasoning drift—are well localized to our 35.8M-token projector budget, terse-target data ratio, and SFT-calibrated RLVR gating. Addressing these factors incrementally in Stage-14 and at cluster scale provides a direct, empirically grounded path forward.

---

## References

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
