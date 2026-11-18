"""Unit tests for `scripts/eval_standard_multimodal_benchmarks.py` parsers and scorers."""
from __future__ import annotations

from PIL import Image

from scripts.eval_standard_multimodal_benchmarks import (
    BENCHMARK_PINS,
    decode_parquet_image,
    extract_mcq_choice,
    extract_yes_no,
    parse_safe_list,
    score_mmmu_open_answer,
    score_ocrbench_answer,
    score_realworldqa_answer,
)


def test_standard_mm_mcq_and_ai2d_label_safety():
    # Standard explicit markers and <think> blocks
    assert extract_mcq_choice("<think>\nCheck A vs C.\n</think>\nAnswer: C", ["A", "B", "C", "D"]) == "C"
    assert extract_mcq_choice("答案：B", ["A", "B", "C", "D"]) == "B"
    assert extract_mcq_choice("(D) $77,490", ["A", "B", "C", "D"]) == "D"

    # AI2D diagram labels ['c', 'D', 'b', 'a']: Answer: B must return 'B', never collide with option text 'b'/'D'!
    assert extract_mcq_choice("Answer: B", ["A", "B", "C", "D"], ["c", "D", "b", "a"], min_option_text_len=3) == "B"

    # Unique multi-word option text match
    opts = ["Seal population will become extinct", "Fish population would decrease.", "Whale population would increase."]
    assert extract_mcq_choice("Fish population would decrease", ["A", "B", "C"], opts, min_option_text_len=3) == "B"


def test_ocrbench_realworldqa_mmmu_and_pope_scorers():
    assert len(BENCHMARK_PINS) == 8
    assert parse_safe_list("['$6', '$7', '$8']") == ["$6", "$7", "$8"]

    # OCRBench regular + HMER whitespace-insensitive LaTeX
    assert score_ocrbench_answer("The sign reads CENTRE.", ["CENTRE"], "Regular Text Recognition") is True
    assert score_ocrbench_answer(r"x ^ { 2 } + y _ { 1 }", [r"x^{2}+y_{1}"], "Handwritten Mathematical Expression Recognition") is True
    assert score_ocrbench_answer("WRONG", ["CENTRE"], "Regular Text Recognition") is False

    # RealWorldQA MCQ + word-number + binary
    q_mcq = "Which is smallest?\nA. Left\nB. Middle\nC. Right\nPlease answer directly."
    assert score_realworldqa_answer("Answer: C", q_mcq, "C") is True
    assert score_realworldqa_answer("Three", "How many cars?", "3") is True
    assert score_realworldqa_answer("Yes, the red car is closer.", "Is it closer?", "Yes") is True

    # MMMU open numeric & string
    assert score_mmmu_open_answer("<think>Compute NPV</think>\nAnswer: $12,000", "12000") is True
    assert score_mmmu_open_answer("Answer: 3.14", "['3.1415']") is True

    # POPE binary
    assert extract_yes_no("Answer: Yes") == "yes"
    assert extract_yes_no("No, there is no snowboard.") == "no"

    # Parquet image decoder
    im = Image.new("P", (8, 8))
    dec = decode_parquet_image(im)
    assert dec is not None and dec.mode == "RGB"
