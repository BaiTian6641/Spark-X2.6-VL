"""Generate the phase-split Colab notebooks from the canonical full notebook."""

import argparse
import copy
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = Path("colab/Spark_Formal_Full_Finetune.ipynb")
CHECKPOINT_REPO = "TassP0lar/SparkX2.6-MM-checkpoints-g4-v9"
PHASES = {
    "train": {
        "filename": "Spark_Full_Finetune.ipynb",
        "result": "train_complete",
        "instructions": """## Phase: training (`RUN_PHASE = "train"`)

Run all cells to execute only the training phase. This notebook uses the same checkout, work root, campaign, and checkpoint lineage as the canonical full run, then stops after the engine records `train_complete`. A session checkpoint still exits with code `3`; rerun all cells to resume. `OPENCODE_API_KEY` is not required for this phase.
""",
    },
    "rsi": {
        "filename": "Spark_RSI_Post_Training.ipynb",
        "result": "rsi_complete",
        "instructions": """## Phase: RSI post-training (`RUN_PHASE = "rsi"`)

Run all cells after training has committed a compatible checkpoint. This notebook resumes the same checkout, work root, campaign, and checkpoint lineage, executes only the instructor-guided RSI phase, and stops after the engine records `rsi_complete`. Enable `OPENCODE_API_KEY` in Colab Secrets before running. A session checkpoint still exits with code `3`; rerun all cells to resume.
""",
    },
    "audit": {
        "filename": "Spark_Result_Checking.ipynb",
        "result": "audit_complete",
        "instructions": """## Phase: result checking (`RUN_PHASE = "audit"`)

Run all cells after RSI post-training has committed a compatible checkpoint. This notebook resumes the same checkout, work root, campaign, and checkpoint lineage, executes only the protected audit and publication phase, and records `audit_complete`. `OPENCODE_API_KEY` is not required. Successful audit publication is confirmed by the existing `MODEL_UPLOAD_RECEIPT.json`; a session checkpoint still exits with code `3`.
""",
    },
}


def _source(cell):
    return "".join(cell["source"])


def _set_source(cell, source):
    cell["source"] = source.splitlines(keepends=True)


def _replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected exactly one canonical fragment: {old!r}")
    return source.replace(old, new, 1)


def build_notebook(phase, canonical):
    """Return one independently mutated deep copy of the canonical notebook."""
    if phase not in PHASES:
        raise ValueError(f"Unknown phase: {phase}")
    spec = PHASES[phase]
    notebook = copy.deepcopy(canonical)
    cells = notebook["cells"]
    if len(cells) != 8:
        raise ValueError("Canonical notebook layout changed; expected eight cells")

    markdown = _source(cells[0])
    cells[0]["source"] = (markdown.rstrip("\n") + "\n\n" + spec["instructions"]).splitlines(keepends=True)

    setup = _source(cells[1])
    setup = _replace_once(
        setup,
        "SESSION_START = time.time()\n",
        f'SESSION_START = time.time()\nRUN_PHASE = "{phase}"\n',
    )
    setup = re.sub(r'HF_CHECKPOINT_REPO = "[^"]+"', f'HF_CHECKPOINT_REPO = "{CHECKPOINT_REPO}"', setup, count=1)
    _set_source(cells[1], setup)

    config = _source(cells[5])
    config = _replace_once(
        config,
        'INSTRUCTOR_KEY = get_secret("OPENCODE_API_KEY")',
        'INSTRUCTOR_KEY = get_secret("OPENCODE_API_KEY", required=RUN_PHASE in ("full", "rsi"))',
    )
    config = _replace_once(
        config,
        '    "work_root": str(WORK),\n',
        '    "work_root": str(WORK),\n    "run_phase": RUN_PHASE,\n',
    )
    _set_source(cells[5], config)

    receipt = _source(cells[7])
    receipt = _replace_once(
        receipt,
        '''status = (
    "COMPLETE_AND_EXPORTED" if EXIT_CODE == 0 and upload else
    "SESSION_CHECKPOINTED_RERUN_TO_RESUME" if EXIT_CODE == DIRECT_SESSION_EXIT else
    "QUALIFICATION_COMPLETE_RERUN_FOR_RECOVERY" if EXIT_CODE == DIRECT_QUALIFICATION_EXIT else
    "EXITED_WITHOUT_COMPLETION" if EXIT_CODE == 0 else "FAILED"
)
''',
        '''phase_result = state.get("phase_result") if EXIT_CODE == 0 else None
expected_phase_result = {
    "train": "train_complete",
    "rsi": "rsi_complete",
    "audit": "audit_complete",
}[RUN_PHASE]
phase_complete = EXIT_CODE == 0 and phase_result == expected_phase_result
status = (
    "COMPLETE_AND_EXPORTED" if phase_complete and RUN_PHASE == "audit" and upload else
    "PHASE_COMPLETE" if phase_complete and RUN_PHASE != "audit" else
    "SESSION_CHECKPOINTED_RERUN_TO_RESUME" if EXIT_CODE == DIRECT_SESSION_EXIT else
    "QUALIFICATION_COMPLETE_RERUN_FOR_RECOVERY" if EXIT_CODE == DIRECT_QUALIFICATION_EXIT else
    "EXITED_WITHOUT_COMPLETION" if EXIT_CODE == 0 else "FAILED"
)
''',
    )
    receipt = _replace_once(
        receipt,
        '    "status": status,\n',
        '    "status": status,\n    "run_phase": RUN_PHASE,\n    "phase_result": phase_result,\n',
    )
    _set_source(cells[7], receipt)

    notebook["metadata"].setdefault("colab", {})["name"] = spec["filename"]
    notebook["metadata"].setdefault("spark_formal", {})["run_phase"] = phase
    return notebook


def render_notebook(notebook):
    return json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"


def write_notebooks(root=ROOT):
    """Build every phase in a stable order and return the output paths."""
    root = Path(root)
    canonical = json.loads((root / CANONICAL).read_text(encoding="utf-8"))
    outputs = []
    for phase, spec in PHASES.items():
        destination = root / "colab" / spec["filename"]
        destination.write_text(
            render_notebook(build_notebook(phase, canonical)), encoding="utf-8"
        )
        outputs.append(destination)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for path in write_notebooks():
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
