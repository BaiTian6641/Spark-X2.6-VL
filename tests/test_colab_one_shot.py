import ast
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from colab.stage1.prepare_public_image_frame import choose_splits
from colab.stage1.port_trainer import materialize

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "colab" / "Spark_Full_Finetune_OneShot.ipynb"


class OneShotNotebookTests(unittest.TestCase):
    def setUp(self):
        self.notebook = json.loads(NOTEBOOK.read_text())
        self.code = [cell["source"] for cell in self.notebook["cells"] if cell["cell_type"] == "code"]

    def test_notebook_is_valid_and_has_no_cpu_gate(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        joined = "\n".join(self.code)
        for index, source in enumerate(self.code):
            ast.parse(source, filename=f"notebook-cell-{index}")
        for forbidden in ("run_campaign('cpu')", "RUN_QUALIFICATION", "OPENCODE_API_KEY",
                          "Spark_Formal_Full_Finetune.ipynb"):
            self.assertNotIn(forbidden, joined)
        self.assertIn("prepare_public_image_frame", joined)
        self.assertIn("train_multimodal_cuda.py", joined)
        self.assertIn("--formal", joined)

    def test_notebook_declares_fresh_work_root_and_required_source(self):
        joined = "\n".join(self.code)
        self.assertIn('/content/spark-mm', joined)
        self.assertIn('REPO_REF = "codex/formal-colab"', joined)
        self.assertIn("require_reviewed_gpu=False", joined)


class PublicBuilderTests(unittest.TestCase):
    def test_group_split_is_deterministic_and_group_isolated(self):
        records = [
            {"source_group": f"group-{index}", "task": "factual_caption", "language": "en"}
            for index in range(40)
        ]
        first = choose_splits(records)
        second = choose_splits(records)
        self.assertEqual(first, second)
        self.assertEqual(set(first), {record["source_group"] for record in records})
        # Every group is assigned exactly one split; no record-level split is used.
        self.assertTrue(set(first.values()) <= {"train", "dev"})


class GenericPortTests(unittest.TestCase):
    def test_generic_guard_does_not_require_reviewed_gpu_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            shutil.copytree(ROOT / "scripts", project / "scripts")
            shutil.copytree(ROOT / "multimodal", project / "multimodal")
            receipt = materialize(project, minimum_gpu_gib=46, require_reviewed_gpu=False)
            source = (project / "scripts" / "train_multimodal_cuda.py").read_text()
            ast.parse(source)
            self.assertIn("at least 46 GiB", source)
            self.assertNotIn("RTX PRO 6000", source)
            self.assertNotEqual(receipt["original_rocm_sha256"], receipt["cuda_entrypoint_sha256"])
