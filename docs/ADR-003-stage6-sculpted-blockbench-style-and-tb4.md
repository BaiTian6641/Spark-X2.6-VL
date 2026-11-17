# ADR-003: Stage-6 Native Sculpted Blockbench Style (`v3.2` / `v4.0`), Human-Artist Studio Synthesis, and Terminal-Bench 4.0

- **Status**: Accepted
- **Date**: 2026-11-16
- **Context**:
  - In Stage 5 + Vision RSI-2, `Spark-X2.6-4B-VL` achieved strong 15-axis `strict` geometry synthesis (27-box articulated rigs, `GPQA-Diamond = 33.33%`, `Live Python BFS = 100%`, `World Ladder Holdout = 72.61%`), and zero-shot `Terminal-Bench 4.0` improved from `0.0%` (Stage 4) to `12.5%` (`100%` `<think>` closure).
  - However, `audit_profile: strict` (`1.0u` integer grids, zero parent-child seating overlap) penalizes organic master-artist Blockbench idioms: `25–30%` seated limb/neck joints (`anchor ~ 0.85..0.95`), `0.5u` chamfers (`size_step: 0.5`), non-grid gesture tilts (`3–36°`), 8-step hue-shifted cel-banded pixel ramps (`cool violet/indigo shadows -> saturated midtones -> warm cream/gold highlights`), and zero-thickness RGBA `alpha=0` cutout `plane`s for wing membranes, jagged spike/fur fringes, tooth skirts, and foliage (`skills/blockbench-sculpted-style/SKILL.md`).
  - Simultaneously, SyncCraft `1f41187` introduced modular Block-DSL `v4.0` (`@layout`, `@component`, `@assembly`), and `Terminal-Bench` updated to `v4.0.0` (`harborframework/terminal-bench`, `Apache-2.0`, commit `3b8d57a69a7d28b73238c7ce9ecc62c8c794bff9`).

## Decisions

1. **Native Sculpted Blockbench Style + Block-DSL `v3.2` / `v4.0` Synthesis (`colab/final/sculpted_style_corpus.py`)**:
   - Implement a deterministic OKLCH -> 8-step hue-shifted cel-banded RGBA pixel-art atlas baker (`bake_sculpted_rgba_atlas`) with procedural `alpha=0/255` pixel cutouts on `plane` elements (`wing_membrane`, `spike_fringe`, `tooth_skirt`, `foliage_or_cloth`, `energy_arc`), 1px top bevels, 1px dark contact edges, stepped scale/scute/grain clusters, catchlight eyes, and focal emissive glow bleed.
   - Implement a perspective Z-buffer dark-slate studio renderer (`render_sculpted_turnaround_views`) producing `south`, `right`, `top`, `iso`, and 4-view `silhouette_sheet` check PNGs.
   - Author 8 clean-room parametric `sculpted_master_tier` archetype families (`CC0-1.0`, zero IP/franchise references; `research_nc_fan_reference` studies remain 100% quarantined):
     1. `crimson_cream_western_dragon` (`v3.2`, 87 elements, 11 cutout planes)
     2. `glacial_crystal_drake` (`v3.2`, 43 elements, 5 cutout planes)
     3. `ochre_club_wyvern` (`v3.2`, 41 elements, 5 cutout planes)
     4. `pastel_chibi_dragonet` (`v3.2`, 27 elements, 3 cutout planes)
     5. `storm_fanged_wolf_dragon_v4` (`v4.0`, `@layout/@component/@assembly`, 27 elements, 3 cutout planes)
     6. `orchard_harvest_automaton` (`v3.2`, 29 elements, 2 cutout planes, 45° octagonal hubs)
     7. `polar_expedition_crawler` (`v3.2`, 25 elements, 1 cutout plane, snow-cap slabs & 6x6 bogies)
     8. `alchemist_studio_diorama` (`v3.2`, 20 elements, 3 cutout planes, warm focal lantern vs cool violet ambient)
   - Pair these with studio-painted turnaround views & silhouette critiques of `permissive_cc` human-artist Blockbench rigs (`TassP0lar/synccraft-v3-human-artist-dataset`, `CC0-1.0` / `CC-BY-4.0`, `ai_assisted == False`) and SyncCraft v4.0 modular assembly supervision tasks (`synccraft_v4_modular_assembly`).

2. **Disjoint Terminal-Bench 4.0 Integration (`colab/final/reasoning_agentic_corpus.py`)**:
   - Hash task IDs deterministically via `sha256("tb3_holdout_v1:" + task_id) % 3`: `bucket != 0` (`2/3`) enters `terminal_bench_agent` training with fenced bash/python script supervision, while `bucket == 0` (`1/3`, 22 `v4.0.0` tasks) remains 100% unseen for zero-shot holdout evaluation.

3. **Sculpted Vision RSI Closed Loop (`colab/final/vision_rsi.py`)**:
   - Extend `score_rsi_candidate`, `build_self_critique_turn`, and `_select_stratified_rsi_pool` with `SCULPTED_STYLE_TASKS`, rendering 4-view studio turnarounds (`render_sculpted_turnaround_views`) + `evaluate_sculpted_blockbench_style` diagnostics on failed candidates with zero ground-truth leakage, gated by the 5-axis ADR-002 promotion guard.
