---
license: apache-2.0
language:
- en
- zh
pipeline_tag: image-text-to-text
datasets:
- TassP0lar/PRISM-Omni-Dataset-v1
tags:
- multimodal
- vision-language
- spark
- c-radiov4
- loop-transformer
- engram
- 3d-spatial
- synccraft
- blockbench
- rlvr
---

# Spark-X2.6-4B-Loop48-En1B-VL (Stage-13 Milestone Release)

**Spark-X2.6-4B-Loop48-En1B-VL** is an open-weight research vision-language model adapting the 36-layer text-native `Spark-X2.6-4B` backbone into a **6.0477B-parameter** multimodal reasoning system (inclusive of CPU-pinned sparse conditional memory). All stages were developed and evaluated on a single dual-GPU workstation (1× NVIDIA RTX PRO 4000 Blackwell SFF 24 GB + 1× Intel Arc Pro B50 16 GB), with independent clean-room verification of the intermediate Stage-13A checkpoint on an NVIDIA A100-SXM4-80GB instance.

At the **Stage-13 milestone**, the active `Loop48-En1B-VL` lineage has accumulated **188,756 supervised samples (~114.2M estimated tokens)** across **100.11 linear wall-clock hours** (Stages 7–13), including **95,864 samples (~35.8M tokens, 25.80 h)** on the current FP32 `RadioMultiLayerGELUProjector` architecture introduced in Stage-12. Compared against Stage-12 on identical zero-overlap benchmarks (`scale=0.6`, `n=1,056`), Stage-13 raises the **8-benchmark multimodal macro average** from `37.75% (BF16) / 37.68% (FP8)` to **`44.15% (BF16) / 44.83% (FP8+q4g128)`** and resolves the modality decoupling previously observed under independent Real-Image vs. Blank-Gray vs. Shuffled-Image counterfactual testing.

- **Full IEEE Milestone Technical Reports:**
  - English: [`docs/Spark-X2.6-VL_Technical_Report_EN_IEEE.md`](docs/Spark-X2.6-VL_Technical_Report_EN_IEEE.md)
  - Chinese (中文): [`docs/Spark-X2.6-VL_Technical_Report_ZH_IEEE.md`](docs/Spark-X2.6-VL_Technical_Report_ZH_IEEE.md)
- **Machine-Generated Verification Artifacts:** [`benchmarks/`](benchmarks/)

---

## 1. Audited Architecture (6.0477B Total Parameters)

The released checkpoint combines four subsystems attached to `Spark-X2.6-4B` (36 layers, native 1M RoPE context, `sliding_window=512` on 27/36 layers and global attention every fourth layer):

1. **48-Virtual-Layer Looping (`Loop48` Dual-mHC):** Middle layers `[12, 24)` execute two recurrent passes governed by doubly-stochastic manifold-constrained residual connections and rank-16 pass adapters.
2. **Tiered MoE-Engram Conditional Memory (`En1B`):** Eight anchor depths (`[3, 7, 11, 15, 19, 23, 27, 31]`) attach a 16-expert (`top_k=4`) sparse lookup subsystem split across GPU Tier-1A hot buckets and CPU-pinned Tier-1B cold buckets (`998.82M` parameters total).
3. **Agglomerative Vision Encoder (`nvidia/C-RADIOv4-H`):** A 32-layer ViT-H/16 backbone (`651.65M` parameters) distilled from SigLIP 2, DINOv3, and SAM3. Single-image inputs are bounded to `<= 1,600` patches (`<= 400` merged tokens after `2x2` pixel-shuffle) so 100% of visual prefix tokens remain visible inside the 512-token sliding attention window across all 48 virtual layers.
4. **FP32 `RadioMultiLayerGELUProjector` + 2D Row-Break Tokens:** Features from layers 24 and 32 (`RMSNorm(L24) || RMSNorm(L32) -> 2x2 pixel-shuffle -> RMSNorm(10240) -> Linear(10240->4096) -> GELU -> Linear(4096->hidden)`) are projected in FP32 with active-GELU initialization (`tau_init = 0.035`), native `\n` row-break embeddings appended after each spatial row, and one-sided visual RMS ceiling regularization (`lambda_rms = 5.0`, `tau_cap in {0.08, 0.09, 0.10}`) during training.
5. **3D Spatial Primitives & SyncCraft Block-DSL:** Supports structured 4-primitive spatial grounding (`<point z="near|mid|far">`, `<box>`, `<dash>`, `<circle>`) and SyncCraft biomechanical animation proposal (`Gate D`).

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

---

## 2. Repository Layout

- `bf16/` — Portable **BF16** full-precision Stage-13 release (`~12.7 GB` `model.safetensors`, `config.json`, `MODEL_RELEASE.json`, `reload-assets/`, and `runtime/`). Exports `post_stage13b` (see Section 3 for the Phase 13C rollback note).
- `fp8/` — Portable **FP8 E4M3 channelwise + INT4 (`q4g128`) sub-LSB residual** Stage-13 release (`~10.7 GB` `model.safetensors`, lossless vs. BF16 on the 8-benchmark suite, includes `reload-assets/` and `runtime/`).
- `checkpoints/stage13a-bf16/` — Intermediate **Stage-13A** BF16 checkpoint (projector + early-12L `self_attn`, independently verified clean-room on Colab A100-SXM4-80GB).
- `benchmarks/` — Raw machine-generated JSON evaluation, training, and counterfactual visual-causality logs from Stages 10.5 through 13.
- `docs/` — Formal English and Chinese IEEE Milestone Technical Reports documenting Stages 7–13.

---

## 3. Complete Stage-by-Stage Training Procedures (Stages 1–13)

### 3.1 Pre-Loop48 Foundation Lineage (Stages 1–6.1)

Prior to introducing `Loop48` and `En1B` in Stage-7, six foundational workstation stages established the base `Spark-X2.6-4B-VL` pipeline, SyncCraft 3D geometry/animation tooling, and anti-collapse evaluation guards:

- **Stages 1–4 (`spark-synccraft-24gb`, `spark-world-v32-24gb`):** Initial `Spark-X2.6-4B` multimodal adaptation, SyncCraft Block-DSL `v3.2` supervision, and 24 GB `APOLLO+COAT` memory-compressed training validation.
- **Stage 5 (`spark-world-v32-stage5-rsi`, ADR-002):** Introduced Vision Recursive Self-Improvement (`Vision RSI-1/2`), instruction-scoped 4-gram anti-collapse gating, KV-cached `<think>` budget forcing, and geometry-neutral 3D schema prompting (`GPQA-Diamond` 24-item split `33.33%`, `SAT` unseen `62.5%`, `World Ladder` `72.61%`).
- **Stages 6 & 6.1 (`spark-world-v40-stage6*`, ADR-003):** Added clean-room parametric `sculpted_master_tier` Blockbench archetypes (`v3.2`/`v4.0`), OKLCH cel-banded RGBA texture baking, and disjoint `Terminal-Bench 4.0` holdout evaluation (`TB` 8-item holdout syntax pass rose `0.0% -> 37.5% -> 75.0%`). Stage-6.1 established the earliest frozen 8-benchmark multimodal baseline (`29.39% BF16 / 29.20% FP8`, `POPE F1 0.122`).

### 3.2 Loop48-En1B-VL Iterative Campaign Ledger (Stages 7–13)

All stages below trained on the same heterogeneous workstation (**1× NVIDIA RTX PRO 4000 Blackwell 24 GB** running `APOLLO` rank-64 + `COAT` FP8 group-16 + **1× Intel Arc Pro B50 16 GB** running vision encoder updates via 8-bit `Muon8bit` over DualPinned shared memory), with strict `0%` overlap against all evaluation benchmarks:

| Stage | Vision / Projector Architecture | Core Procedure & Key Change | Updates | Samples | Wall-Clock | Primary Outcome & Diagnosis |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- |
| **Stage 7** | `SigLIP2-SO400M` (27L) | Introduced `Loop48` Dual-mHC + `En1B` defrost (Phase A) and dual-GPU 27L `SigLIP2` + `Spark` joint SFT (`2,331` upd) + 2 Vision-RSI rounds | 2,331 | 27,972 | 28.65 h | 8-bench macro `37.36%` (`AI2D 58.33%`, `BLINK 45.37%`), but `POPE` collapsed to all-`"Yes"` (`yes=1.000`, `50.00%`); RSI candidates held back by anti-collapse gate. |
| **Stage 7.5** | `SigLIP2-SO400M` (surgery) | Frozen-backbone visual alignment (`600` upd) + joint v25 sculpted aesthetic & frontier polish (`300` upd) | 900 | 10,800 | 9.45 h | `media_advantage = 0.0469`; 8-bench macro `35.59% BF16 / 36.84% FP8`; `POPE` remained all-`"Yes"` (`yes=1.000`). |
| **Stage 8** | `C-RADIOv4-H` + `SwiGLU` DeepStack | Replaced `SigLIP2` with 32L `nvidia/C-RADIOv4-H` (L24+L32); 8A frozen-Spark (`800` upd) + 8B full-Spark & top-6L RADIO `Muon8bit` (`400` upd) | 1,200 | 14,400 | 9.22 h | 8-bench macro `37.69% BF16 / 37.98% FP8`; `CV-Bench 3D 47.00%`, `3DSRBench 39.00%`, `TB 4.1 76.19%`; `POPE` still all-`"Yes"` (`yes=1.000`). |
| **Stage 9** | `C-RADIOv4-H` + `SwiGLU` | Frontier reasoning + `v28` sculpted biomech + 50/50 Yes/No & direct OCR slice; 9A (`550` upd) + 9B Dual-Track RSI-3 (`160` upd, accepted) | 710 | 8,520 | 7.89 h | `CV-Bench 3D` jumped to `58.00%`, `GPQA 38.00%`, `TB 4.1 76.19%`; broke all-`"Yes"` prior (`POPE yes=0.278`, `Acc 43.33%`); 8-bench macro `32.52%`. |
| **Stage 10** | `C-RADIOv4-H` + `SwiGLU` + Exit RMSNorm | Added learnable exit RMSNorm (`radio_output_rms_match=True`) + 6-mode OCR/Chart/Table + `v28` repair; 10A (`750`) + 10B (`450`) + 10C RSI-4 (`140`) | 1,340 | 16,080 | 10.55 h | Severe visual-text mismatch dip (`macro 23.63%`, `AI2D 20.00%`, `CV-Bench 3D 12.00%`); diagnosed residual projector DC bias and visual prefixes exceeding `sliding_window=512`. |
| **Stage 10.5** | `C-RADIOv4-H` + clean re-init `SwiGLU` (`<=400` tok) | Zero-DC-bias projector re-init + window-aware `<=400` visual tokens + real `DocVQA`/`SynthDoG` OCR + 11-mutator `v28` repair; 10.5A (`900`) + 10.5B (`360`) + 10.5C RSI-4.5 (`120`) | 1,380 | 16,560 | 8.30 h | Recovered alignment (`macro 32.39%`, `AI2D 60.00%`, `RealWorldQA 44.17%`, `MMMU-Pro 25.00%`, `HLE 7.50%`, `TB 4.1 66.67%`, `OCRBench 2.50%`). |
| **Stage 11** | `C-RADIOv4-H` + `SwiGLU` (`L_decorr=0.15`) | Added feature decorrelation loss (`0.15`), split early-12L LR, visual-margin hard mining, and `v38.8` 64-archetype corpus; 11A (`960`) + 11B RSI-5 (`140`) | 1,100 | 13,200 | 9.52 h | Exposed attention/grad-clip bugs (`macro 28.84%`, `POPE F1 0.043`), while `TB 4.1` held `76.19%`. |
| **Stage 11.5** | `C-RADIOv4-H` + `SwiGLU` (FP32-AdamW fixes) | Fixed Stage-11 attention/eval/grad-clip bugs, FP32-master AdamW connector alignment, open-vocab OCR + RSI-5.5 (`120` upd) | 120* | 1,440* | 1.28 h* | 8-bench macro jumped to `36.80% BF16 / 36.91% FP8`, first balanced `POPE` (`52.78%`, `F1 0.536`, `yes=0.517`), `CV-Bench 3D 49.17%`, `3DSRBench 48.33%`. |
| **Stage 12** | **FP32 `GELU` Projector + 2D `\n` + RMS ceiling** | Replaced SwiGLU with fresh FP32 `RadioMultiLayerGELUProjector` + `\n` row breaks + one-sided RMS cap (`0.08`); 12A-1 (`500`, `B=16`) + 12A-2 (`500`, `B=8` chunked RADIO) + 12B (`640`) + 12C RSI-6 (`100`) | 1,740 | 20,880 | 7.37 h | 8-bench macro `37.75%`, `POPE 62.78%` (`F1 0.579`), `AI2D 63.33%`, `CV-Bench 3D 55.83%`, `GPQA 38.89%`, `HLE 8.00%`, unseen word OCR `50.0%`. Counterfactual audit exposed remaining decoupling on `POPE` (`+0.56%` vs. shuffled), `AI2D` (`-2.50%`), and `Color×Shape` (`"red circle"` prior). |
| **Stage 13A** | Warm-started FP32 `GELU` + Early-12L `self_attn` | Warm-started from Stage-12 over `57,485` audited records (`TassP0lar/PRISM-Omni-Dataset-v1` rev `8e5ac6197b`, all-80 `COCO` with 0/500 POPE overlap, `DOCCI`, `ChartQA`, `ScienceQA`, `ViRL39K`, counterfactual grids); chunked RADIO (`512 px`), `B=8` (`ema_ce: 2.75 -> 0.70`) | 7,948 | 63,584 | 9.03 h | Clean-room Colab A100 eval: 8-bench macro **`44.74%`**, **`POPE 89.44% (F1 0.891, +36.66% vs. shuffled)`**, `MMBench-CN 50.83%`, `MMStar 33.89%`, `Color×Shape 100.0%`. Deep-layer tasks (`AI2D`, `BLINK`) remained visually uncoupled until 13B. |
| **Stage 13B** | **Full-36L `Spark` + Top-8L `C-RADIOv4-H` (`896 px`)** | Full-36L `APOLLO+COAT` + top-8L `C-RADIOv4-H` (`Muon8bit`, `151.0 upd/h`) at full `896 px` with real-vs.-blank visual margin upweighting (`mean_adv = 0.9269 nats`) | 950 | 11,400 | 6.29 h | **Selected final release (`bf16/` & `fp8/`):** 8-bench macro **`44.15% BF16 / 44.83% FP8`**, `AI2D 67.50%` (`+3.75%` lift, KL `0.9487`), `BLINK 47.22% / 49.07%` (`+5.56%` lift), `CV-Bench 3D 60.00%`, `3DSRBench 51.67%`, unseen word OCR `66.67%` (`86.67%` sim). |
| **Stage 13C** | On-policy `VC-GRPO` RLVR (rolled back by NLL gate) | `1,200` verifiable prompts × `K=4` = `4,800` rollouts (`1.57M` scored tokens, `exact_rate 60.38%`, causal `mean Delta log p = +0.928 nats`, `0` human DPO items) | 156 | 4,800 rollouts | 2.73 h | Improved or matched `post_13b` on all 5 counterfactual probes (`OCRBench` quick-32 `40.62% -> 43.75%`, `POPE-Rand 92.5%`, `RWQA 52.5%`, `Color×Shape +100%`, `media_advantage +0.008`), but rolled back to `13B` due to strict SFT NLL gate (`loss_ratio = 1.0513 > 1.04`). |

*\*Stage-11.5 resumed from the completed Stage-11A checkpoint (`prealign` + `11A` fixes) before executing 120 RSI-5.5 updates. Linear Stage 7–13 total: **18,535 updates**, **188,756 supervised samples**, **~114.2M estimated tokens**, **100.11 wall-clock hours** (active Stage 12–13 GELU projector lineage: **10,794 updates**, **95,864 samples**, **~35.8M tokens**, **25.80 h**).*

---

## 4. Verified Cross-Stage Quantitative Scorecards (0% Train Overlap)

### 4.1 Standard 8-Benchmark Multimodal Suite Progression (`scale=0.6`, `n=1,056` / model)

All scores below are taken verbatim from the frozen SHA-pinned benchmark JSON reports in `benchmarks/` and workstation archives:

| Stage / Checkpoint | 8-Bench Macro | AI2D | RealWorldQA | BLINK 6-Sp | MMBench-CN | MMMU Val | MMStar | POPE Acc (`F1` / `yes`) | OCRBench 5-Grp |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Stage-6.1 BF16 (pre-Loop48) | 29.39% | 52.50% | 36.67% | 40.74% | 29.17% | 35.19% | 27.78% | 52.22% (`0.122` / `0.044`) | 0.83% |
| Stage-7 BF16 (`SigLIP2`) | 37.36% | 58.33% | 44.17% | 45.37% | 27.50% | 29.63% | 25.56% | 50.00% (`0.667*` / `1.000`) | 1.67% |
| Stage-7.5 BF16 (`SigLIP2`) | 35.59% | 49.17% | 40.83% | 37.04% | 30.83% | 32.41% | 26.11% | 50.00% (`0.667*` / `1.000`) | 1.67% |
| Stage-8 BF16 (`C-RADIOv4`) | 37.69% | 56.67% | 44.17% | 39.81% | 37.50% | 30.56% | 24.44% | 50.00% (`0.667*` / `1.000`) | 1.67% |
| Stage-9 BF16 (RSI-3) | 32.52% | 51.67% | **45.00%** | 39.81% | 35.83% | 32.41% | 26.67% | 43.33% (`0.271` / `0.278`) | 1.67% |
| Stage-10 BF16 (RMS dip) | 23.63% | 20.00% | 25.00% | 47.22% | 29.17% | **36.11%** | 29.44% | 48.33% (`0.021` / `0.028`) | 0.00% |
| Stage-10.5 BF16 (clean init) | 32.39% | 60.00% | 44.17% | 38.89% | 33.33% | 31.48% | 29.44% | 48.89% (`0.193` / `0.133`) | 2.50% |
| Stage-11 BF16 | 28.84% | 58.33% | 40.00% | 39.81% | 29.17% | 29.63% | 27.78% | 50.56% (`0.043` / `0.017`) | 1.67% |
| Stage-11.5 BF16 (RSI-5.5) | 36.80% | 60.00% | 37.50% | 40.74% | 35.83% | 31.48% | 32.78% | 52.78% (`0.536` / `0.517`) | 2.50% |
| Stage-12 BF16 (`GELU` + `\n`) | 37.75% | 63.33% | 40.83% | 43.52% | 32.50% | 31.48% | 28.33% | 62.78% (`0.579` / `0.383`) | 4.17% |
| Stage-12 FP8+q4g128 | 37.68% | 64.17% | 40.00% | 43.52% | 32.50% | 28.70% | 28.89% | 64.44% (`0.595` / `0.378`) | 4.17% |
| Stage-13A BF16 (Colab A100) | 44.74% | 61.67% | 44.17% | 43.52% | 50.83% | 30.56% | **33.89%** | **89.44% (`0.891` / `0.472`)** | 4.17% |
| **Stage-13 Final BF16** | 44.15% | **67.50%** | 38.33% | 47.22% | 50.83% | 26.85% | 32.22% | 86.11% (`0.852` / `0.439`) | **5.00%** |
| **Stage-13 Final FP8+q4g128** | **44.83%** | **67.50%** | 40.00% | **49.07%** | **51.67%** | 26.85% | 33.33% | 86.11% (`0.852` / `0.439`) | **5.00%** |
| *Delta Stage-13 vs. Stage-12 (BF16 / FP8)* | **+6.40% / +7.15%** | **+4.17% / +3.33%** | -2.50% / 0.00% | **+3.70% / +5.55%** | **+18.33% / +19.17%** | -4.63% / -1.85% | **+3.89% / +4.44%** | **+23.33% / +21.67%** | **+0.83% / +0.83%** |

*\*Stages 7–8 POPE F1 (`0.667`) reflects a degenerate all-`"Yes"` prior (`yes_ratio = 1.000`). Stage-13A and Stage-13 Final achieve balanced precision/recall across Random (`88.33%`), Popular (`81.67%–88.33%`), and Adversarial (`88.33%–91.67%`) splits. On Stage-13 `OCRBench` 5-group (`n=120`), sub-group accuracy is `Text Recognition 16.67%`, `Doc-oriented VQA 8.33%`, and `0.00%` on `KIE`, `HMER`, and `Scene Text VQA`.*

### 4.2 3D Spatial, Frontier Reasoning & SyncCraft Progression (BF16, Frozen Pins)

| Benchmark Track | Stage-8 | Stage-9 | Stage-10 | Stage-10.5 | Stage-11.5 | Stage-12 | **Stage-13 Final** | Stage-13 Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **CV-Bench 3D** | 47.00% | 58.00% | 12.00% | 40.00% | 49.17% | 55.83% | **60.00%** | `n=120`; Depth `59.0%`, Distance `61.0%`, 4-prim `99.17%` |
| **3DSRBench** | 39.00% | 40.00% | 27.00% | 37.00% | 48.33% | 44.17% | **51.67%** | `n=120`; 4-prim `97.50%`; Parallel `80%`, Closer/Next-to `70%` |
| **MMMU-Pro (10-Opt)** | 20.00% | 21.00% | 3.00% | **25.00%** | 20.00% | 20.83% | **22.50%** | `n=120`; >2.2× 10-option random chance |
| **GPQA-Diamond** | 36.00% | 38.00% | 34.00% | 36.00% | 34.85% | **38.89%** | 36.87% | `n=198`; Physics `45.35%`, Chemistry `31.18%`, Bio `26.32%` |
| **Terminal-Bench 4.1** | **76.19%** | **76.19%** | 52.38% | 66.67% | 66.67% | 61.90% | 52.38% | `n=21`; substantive `33.33%`, healthy `<think>` `47.62%` |
| **SimpleQA-Verified** | **4.00%** | 3.00% | 3.00% | **4.00%** | 3.33% | 1.67% | 3.33% | `n=120`; abstained `2` |
| **HLE-Diamond** | 5.00% | 6.25% | 3.75% | 7.50% | 7.00% | **8.00%** | — | Omitted from Stage-13 automated post-run to fit 24 h budget |
| **AIME 2026** | 0.00% | 0.00% | 3.33% | 3.33% | 3.33% | 0.00% | — | Omitted from Stage-13 automated post-run |
| **SyncCraft Gate D** | **100.0%** | **100.0%** | — | **100.0%** | **100.0%** | **100.0%** | **100.0%** | `n=4`; mean solver calls `1.0` (`243` tokens) |
| **SyncCraft Unseen `v28` Compile** | — | — | — | — | — | 0.00% | 0.00% | `n=4`; grammar-level Block-DSL slips remain unresolved |

### 4.3 Independent Counterfactual Visual-Causality Verification (BF16)

Evaluated via `scripts/verify_stage12_visual_causality.py` comparing **Real Image** vs. **Blank-Gray Control (`RGB(128,128,128)`)** vs. **Shuffled Image** (POPE) and first-token symmetric KL divergence:

| Audit Track | Stage-12 Real / Blank (`Delta_vis`) | Stage-13A A100 Real / Blank (`Delta_vis`) | **Stage-13 Final Real / Blank (`Delta_vis`)** | Stage-13 Signal / Control Details |
| :--- | :---: | :---: | :---: | :--- |
| **POPE Overall (`n=180`)** | 62.78% / 55.00% (`+7.78%`) | **89.44% / 50.00% (`+39.44%`)** | **86.11% / 50.00% (`+36.11%`)** | Shuffled: `53.33%` (**`+32.78%` vs. Shuffled**; Stage-12 was `+0.56%`); `0/90` Blank FP; `KL = 2.332` nats |
| **POPE Adversarial (`n=60`)** | — | **91.67% / 50.00% (`+41.67%`)** | **88.33% / 50.00% (`+38.33%`)** | `F1 = 0.8814` Real vs. `0.2222` Shuffled |
| **POPE Popular (`n=60`)** | — | **88.33% / 50.00% (`+38.33%`)** | **81.67% / 50.00% (`+31.67%`)** | `F1 = 0.7843` Real vs. `0.3182` Shuffled |
| **AI2D (`n=80`)** | 71.25% / 73.75% (`-2.50%`) | 67.50% / 67.50% (`0.00%`) | **71.25% / 67.50% (`+3.75%`)** | `KL_sym`: `0.0842` (12) -> `0.0007` (13A) -> **`0.9487`** (13); choice flip `22.50%` |
| **BLINK 6-Spatial (`n=72`)** | 47.22% / 45.83% (`+1.39%`) | 41.67% / 44.44% (`-2.78%`) | **48.61% / 43.06% (`+5.56%`)** | Choice flip rate: `2.78%` (12) -> **`30.56%`** (13) |
| **RealWorldQA (`n=80`)** | 36.25% / 35.00% (`+1.25%`) | 42.50% / 31.25% (`+11.25%`) | **36.25% / 25.00% (`+11.25%`)** | Pred flip rate: `11.25%` (12) -> **`36.25%`** (13) |
| **Color × Shape Open (`n=8`)** | 25.0% / 50.0% (`0.0%` lift) | **100.0% / 100.0%** (`+75% / +50%`) | **100.0% / 100.0%** (`+75% / +50%`) | Blank prior `25.0% / 50.0%` (`"yellow circle"`); `"red circle"` collapse eliminated |
| **Color × Shape Yes/No (`n=16`)** | 50.00% / 50.00% (`0.00%`) | **100.00% / 50.00% (`+50.00%`)** | **81.25% / 50.00% (`+31.25%`)** | `0/8` Blank FP |
| **OCRBench Quick-32 Raw** | 31.25% / 0.00% (`+31.25%`) | 31.25% / 0.00% (`+31.25%`) | **40.62% / 0.00% (`+40.62%`)** | Char sim: **`67.57%`** Real vs. `3.17%` Blank (*`43.75%` post-13C*) |
| **Unseen Synthetic Word (`n=12`)** | 50.00% / 0.00% (`+50.00%`) | 50.00% / 0.00% (`+50.00%`) | **66.67% / 8.33% (`+58.34%`)** | Char sim: **`86.67%`** Real vs. `13.33%` Blank (`KL_sym = 7.79` nats) |

### 4.4 Intra-Stage-13 Counterfactual Quick-Probe Progression (`pre_13 -> 13C`)

| Probe Track (`Real Acc / Real - Blank Lift`) | `pre_13` (Stage-12) | `post_13a` (9h `B=8`) | `post_13b` (Exported Final) | `post_13c` (`VC-GRPO`, rolled back) |
| :--- | :---: | :---: | :---: | :---: |
| **Color × Shape (`n=8`)** | 0.0% (`+0.0%`) | 100.0% (`+87.5%`) | 100.0% (`+87.5%`) | **100.0% (`+100.0%`)** |
| **OCRBench Probe (`n=32`, Raw)** | 31.25% (`+31.25%`) | 31.25% (`+31.25%`) | 40.62% (`+40.62%`) | **43.75% (`+43.75%`)** |
| **POPE-Random (`n=40`)** | 50.0% (`+17.5%`) | **95.0% (`+45.0%`)** | 90.0% (`+40.0%`) | 92.5% (`+42.5%`) |
| **POPE-Adversarial (`n=40`)** | 70.0% (`+22.5%`) | **80.0% (`+30.0%`)** | 77.5% (`+27.5%`) | 77.5% (`+27.5%`, `F1 80.0%`) |
| **RealWorldQA Probe (`n=40`)** | 52.5% (`+7.5%`) | **60.0% (`+7.5%`)** | 47.5% (`-2.5%`) | 52.5% (`+2.5%`) |
| **Held-Out Visual Calibration (`n=48`)** | — | — | `real_loss 0.6535`, `adv 0.0424` | `real_loss 0.6870`, **`adv 0.0504`** |

---

## 5. Final Milestone Outcome: Capabilities, Limitations & Next Steps

### What Is Established
- **Verified Pixel-Conditioned Grounding:** Stage-13 converts the early OCR-only alignment of Stage-12 into genuine visual grounding across object existence (`POPE 86.11%–89.44%`, `+32.78%–36.66%` above shuffled images), color/shape attributes (`100.0%`), unseen printed words (`66.67%` exact / `86.67%` char sim), science diagrams (`AI2D 67.50%`, `+3.75%` lift), bilingual VQA (`MMBench-CN 51.67%`), and 3D spatial understanding (`BLINK 49.07%`, `CV-Bench 3D 60.00%`, `3DSRBench 51.67%`).
- **Functional Depth Separation Observed Empirically:** Comparing `stage13a-bf16` (projector + layers `0–11`) against `final-model-stage13` (all 36 layers + top-8 `C-RADIOv4-H` layers) indicates that shallow attention layers suffice for object existence and attribute lookup, while unfreezing deep layers (`12–35`) is required to route visual evidence into comparative reasoning (`AI2D` KL rose `0.0007 -> 0.9487 nats`, `BLINK` lift flipped `-2.78% -> +5.56%`).
- **Lossless Portable Quantization & Clean-Room Reload:** `FP8+q4g128` (`10.69 GB`) slightly outperforms BF16 (`12.69 GB`) on the 8-benchmark macro (`44.83%` vs. `44.15%`). Both `bf16/` and `fp8/` include self-contained inference runtimes with clean-room manifest checks verified on Colab A100.

### Honest Assessment of Known Limitations
- **Free-Form Multi-Sentence Visual Explanation:** Live Real-vs.-Blank testing on five held-out photographs (`RealWorldQA`, `COCO`, `AI2D`, `CV-Bench 3D`) confirms real scene perception (e.g., snow slope / room TV / sofa-chair context vs. unrelated blank priors), but free-form descriptions remain brief (1–3 sentences), occasionally confuse nearby visual classes (e.g., `"skier"` for a snowboarder), and tend to emit terse single-letter outputs under *"explain your reasoning"* prompts because >85% of Stage-13 training targets used concise labels (`DOCCI` dense captions were `2.7%` of the visual pool).
- **Dense Document, KIE & Handwritten Math OCR:** Full 5-group `OCRBench` remains `5.00%` due to `0.00%` accuracy on dense `KIE`, handwritten math (`HMER`), and scene-text VQA, constrained by the `<= 400` visual token budget (`sliding_window=512`) and sparse domain coverage.
- **Text/Reasoning Drift & Block-DSL Compilation:** Focusing Stage-13 heavily on visual grounding introduced modest regressions relative to Stage-12 on `MMMU (val)` (`-4.63%`), `GPQA-Diamond` (`-2.02%`), and `Terminal-Bench 4.1` (`-9.52%`), while unseen raw `SyncCraft v28` compilation remains `0.00%` (`Gate D` proposer stays `100.0%`).
- **Order-of-Magnitude Token Scale:** The active GELU projector lineage (`~35.8M` tokens; `~114.2M` cumulative since Stage-7) remains `~28×` below LLaVA-1.5 (`~1.0B`) and `10^3×–10^5×` below production open-weight VLMs, making token volume and compute scale the primary remaining bottleneck.

### Planned Next Steps
1. **Stage-14 Workstation Refinement:** Replace the strict SFT NLL rollback gate (`loss_ratio <= 1.04`) with an RLVR-appropriate counterfactual/repetition gate to promote `VC-GRPO` + human DPO feedback; increase dense captioning and explain-then-answer data (`DOCCI` full 15K, `PixMo-Cap`) from `<3%` to `35%–40%`; add high-resolution (`896 px`) `DocVQA`/`ChartQA`/`KIE`/`HMER` + STEM/terminal replay to recover `MMMU`, `GPQA`, and `Terminal-Bench`.
2. **Multi-GPU Cluster Scaling (8×–32× H100/B200):** Lift the `sliding_window=512` mask over visual prefix tokens to support `1,024–2,048` multi-tile document patches, unfreeze all 32 `C-RADIOv4-H` layers, and scale across four staged milestones (`~175B–295B` total multimodal tokens; full roadmap in Section VI-B of the IEEE Technical Reports).

---

## 6. Quick Usage (Portable Runtime Included)

Each release directory (`fp8/`, `bf16/`, or `checkpoints/stage13a-bf16/`) bundles its matching `runtime/colab/final/` inference code and `reload-assets/`.

> **Compatibility Note:** Pin `transformers>=4.51,<5` (verified with `pip install "transformers==4.57.3"`). Upstream `modeling_spark.py` declares `_tied_weights_keys` as a Python `list` rather than a `dict`, which raises an `AttributeError` under `transformers >= 5.0`.

```python
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
from PIL import Image

# Swap "fp8/*" -> "bf16/*" for full BF16 weights (~12.7 GB)
local_dir = Path(
    snapshot_download(
        "TassP0lar/Spark-X2.6-4B-Loop48-En1B-VL",
        allow_patterns=["fp8/*"],
    )
) / "fp8"
sys.path.insert(0, str(local_dir / "runtime"))

from colab.final.inference import generate, load, visual_media

model = load(local_dir, device="cuda:0")
im = Image.open("example.jpg").convert("RGB")
print(
    generate(
        model,
        "Describe what is visible in this image and explain the 3D spatial layout.",
        media=visual_media([im]),
        enable_thinking=True,
    )
)
```

---

## 7. Hardware, Data Governance & Citation

- **Training Hardware:** Single workstation with **1× NVIDIA RTX PRO 4000 Blackwell SFF (24 GB)** (`APOLLO` rank-64 + `COAT` FP8 group-16) and **1× Intel Arc Pro B50 (16 GB)** (`nvidia/C-RADIOv4-H` 8-bit `Muon8bit` over zero-copy DualPinned shared memory). Independent clean-room evaluation of `checkpoints/stage13a-bf16/` was executed on **1× NVIDIA A100-SXM4-80GB**.
- **Data Governance:** All training datasets are strictly commercial-permissive (`Apache-2.0`, `MIT`, `CC-BY-4.0`, `CC0-1.0`), including `TassP0lar/PRISM-Omni-Dataset-v1` (pinned revision `8e5ac6197b`), `COCO train2017` (with a programmatic 500-ID `POPE val2014` guard excluding 34 overlapping images -> `0/500` overlap), `DOCCI`, `ChartQA`, `ScienceQA`, `ViRL39K` (53 eval-named rows filtered), `DocVQA`, `TextVQA`, `A-OKVQA`, `SynthDoG-EN`, `GQA`, `Nemotron-v3`, and `SyncCraft` clean-room procedural corpora.

```bibtex
@misc{spark_x26_vl_stage13_2026,
  title        = {Spark-X2.6-4B-Loop48-En1B-VL Milestone Technical Report (Stages 7--13): Multi-Stage Visual Grounding, Counterfactual Causality Verification, and Future Scaling Directions},
  author       = {Bai, Tianrui},
  year         = {2026},
  month        = nov,
  howpublished = {\url{https://huggingface.co/TassP0lar/Spark-X2.6-4B-Loop48-En1B-VL}}
}
```
