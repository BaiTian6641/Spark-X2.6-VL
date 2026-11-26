#!/usr/bin/env python3
"""Interactive User-Feedback Preference Collector for `Spark-X2.6-VL` Multimodal RLVR (`VC-GRPO` + DPO).

Allows you to test `Spark-X2.6-VL` on any local image + prompt, inspect its live answer, rate it, and
optionally enter a corrected (`chosen`) answer so every interaction appends a clean JSONL training pair
to `work/user_feedback_vl.jsonl` consumed directly by `scripts/run_stage13_24h_multimodal_rlvr.py`.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def append_feedback_record(
    output_jsonl: Path,
    *,
    prompt: str,
    chosen: str,
    rejected: str = "",
    image_path: str | None = None,
    rating: float = 5.0,
    note: str = "",
) -> dict[str, object]:
    rec = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "image_path": str(Path(image_path).resolve()) if image_path else None,
        "prompt": prompt.strip(),
        "chosen": chosen.strip(),
        "rejected": rejected.strip(),
        "rating": float(rating),
        "note": note.strip(),
    }
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--release-dir",
        type=Path,
        default=Path("work/spark-x26-4b-loop48-en1b-vl-stage12-frontier-vl/final-model-stage12-bf16"),
    )
    ap.add_argument("--output-jsonl", type=Path, default=Path("work/user_feedback_vl.jsonl"))
    ap.add_argument("--image", type=Path, default=None, help="Optional single-shot image path.")
    ap.add_argument("--prompt", type=str, default=None, help="Optional single-shot prompt.")
    ap.add_argument("--chosen", type=str, default=None, help="Non-interactive chosen answer (logs directly).")
    ap.add_argument("--rejected", type=str, default="", help="Optional non-interactive rejected answer.")
    ap.add_argument("--rating", type=float, default=5.0)
    ap.add_argument("--note", type=str, default="")
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    # Non-interactive direct logging mode (no GPU needed!)
    if args.prompt and args.chosen:
        rec = append_feedback_record(
            args.output_jsonl,
            prompt=args.prompt,
            chosen=args.chosen,
            rejected=args.rejected,
            image_path=str(args.image) if args.image else None,
            rating=args.rating,
            note=args.note,
        )
        print(f"Saved user feedback record to {args.output_jsonl}: {json.dumps(rec, ensure_ascii=False)}")
        return

    from colab.final.inference import generate as spark_generate, load as load_spark, visual_media
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading Spark-X2.6-VL from {args.release_dir} on {device} ...")
    model = load_spark(args.release_dir, device=device)
    print(f"Ready! Feedback will be appended to {args.output_jsonl}. Type 'quit' to exit.\n")

    while True:
        img_str = str(args.image) if args.image else input("Image path (blank for text-only, 'quit' to exit): ").strip()
        if img_str.lower() in ("q", "quit", "exit"):
            break
        med = None
        img_resolved = None
        if img_str:
            p_img = Path(img_str).expanduser()
            if not p_img.is_file():
                print(f"  [!] File not found: {p_img}")
                continue
            im = Image.open(p_img).convert("RGB")
            med = visual_media([im])
            img_resolved = str(p_img.resolve())
        prm = args.prompt or input("Prompt: ").strip()
        if not prm or prm.lower() in ("q", "quit", "exit"):
            break

        pred = spark_generate(
            model,
            prm,
            media=med,
            enable_thinking=bool(args.enable_thinking),
            max_new_tokens=int(args.max_new_tokens),
            do_sample=False,
        ).strip()
        print(f"\n--- Model Output ---\n{pred}\n--------------------")

        verdict = input("Is this answer good? [y=approve / n=provide correction / s=skip]: ").strip().lower()
        if verdict == "y":
            rating_s = input("Rating (1-5, default 5): ").strip()
            rating_v = float(rating_s) if rating_s else 5.0
            append_feedback_record(
                args.output_jsonl,
                prompt=prm,
                chosen=pred,
                rejected="",
                image_path=img_resolved,
                rating=rating_v,
                note="user_approved",
            )
            print("  [Saved approved positive example]\n")
        elif verdict == "n":
            corr = input("Enter the correct / preferred answer: ").strip()
            note = input("Optional note (why it was wrong): ").strip()
            if corr:
                append_feedback_record(
                    args.output_jsonl,
                    prompt=prm,
                    chosen=corr,
                    rejected=pred,
                    image_path=img_resolved,
                    rating=1.0,
                    note=note,
                )
                print("  [Saved chosen vs rejected DPO feedback pair]\n")
        if args.prompt:
            break


if __name__ == "__main__":
    main()
