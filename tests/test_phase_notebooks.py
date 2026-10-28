import ast
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts import build_phase_notebooks as builder


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PATH = ROOT / "colab/Spark_Formal_Full_Finetune.ipynb"
EXPECTED = {
    "train": ("Spark_Full_Finetune.ipynb", "train_complete"),
    "rsi": ("Spark_RSI_Post_Training.ipynb", "rsi_complete"),
    "audit": ("Spark_Result_Checking.ipynb", "audit_complete"),
}
NEW_CHECKPOINT_REPO = "TassP0lar/SparkX2.6-MM-checkpoints-g4-v9"


def source(notebook, index):
    return "".join(notebook["cells"][index]["source"])


def literal_assignment(code, name):
    for node in ast.parse(code).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            if isinstance(node.value, ast.Constant):
                return node.value.value
    raise AssertionError(f"Missing literal assignment for {name}")


class PhaseNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.canonical_bytes = CANONICAL_PATH.read_bytes()
        cls.canonical = json.loads(cls.canonical_bytes)
        cls.notebooks = {
            phase: json.loads((ROOT / "colab" / filename).read_text(encoding="utf-8"))
            for phase, (filename, _) in EXPECTED.items()
        }

    def test_every_notebook_is_valid_and_has_its_phase_and_checkpoint(self):
        self.assertEqual(set(self.notebooks), set(EXPECTED))
        for phase, notebook in self.notebooks.items():
            filename, _ = EXPECTED[phase]
            with self.subTest(phase=phase):
                self.assertEqual(notebook["nbformat"], 4)
                self.assertEqual(notebook["metadata"]["colab"]["name"], filename)
                self.assertEqual(notebook["metadata"]["spark_formal"]["run_phase"], phase)
                self.assertEqual(len(notebook["cells"]), 8)
                for index, cell in enumerate(notebook["cells"]):
                    if cell["cell_type"] == "code":
                        ast.parse(source(notebook, index), filename=f"{filename}-cell-{index}")
                setup = source(notebook, 1)
                self.assertEqual(literal_assignment(setup, "RUN_PHASE"), phase)
                self.assertEqual(
                    literal_assignment(setup, "HF_CHECKPOINT_REPO"), NEW_CHECKPOINT_REPO
                )
                self.assertIn(f'## Phase:', source(notebook, 0))
                self.assertIn(f'`RUN_PHASE = "{phase}"`', source(notebook, 0))

    def test_core_environment_checkout_and_launcher_cells_are_unchanged(self):
        for phase, notebook in self.notebooks.items():
            with self.subTest(phase=phase):
                for index in (2, 3, 4, 6):
                    self.assertEqual(source(notebook, index), source(self.canonical, index))

                setup = source(notebook, 1).replace(f'RUN_PHASE = "{phase}"\n', "", 1)
                setup = setup.replace(NEW_CHECKPOINT_REPO, literal_assignment(source(self.canonical, 1), "HF_CHECKPOINT_REPO"), 1)
                self.assertEqual(setup, source(self.canonical, 1))

                config = source(notebook, 5).replace(
                    'INSTRUCTOR_KEY = get_secret("OPENCODE_API_KEY", required=RUN_PHASE in ("full", "rsi"))',
                    'INSTRUCTOR_KEY = get_secret("OPENCODE_API_KEY")',
                    1,
                ).replace('    "run_phase": RUN_PHASE,\n', "", 1)
                self.assertEqual(config, source(self.canonical, 5))

                self.assertTrue(
                    source(notebook, 0).startswith(source(self.canonical, 0).rstrip("\n"))
                )
                self.assertEqual(
                    source(notebook, 7).split("state = read_if_exists", 1)[0],
                    source(self.canonical, 7).split("state = read_if_exists", 1)[0],
                )
                self.assertEqual(
                    source(notebook, 7).split("receipt_path =", 1)[1],
                    source(self.canonical, 7).split("receipt_path =", 1)[1],
                )
        self.assertEqual(CANONICAL_PATH.read_bytes(), self.canonical_bytes)

    def test_config_and_instructor_key_are_phase_wired(self):
        for phase, notebook in self.notebooks.items():
            with self.subTest(phase=phase):
                tree = ast.parse(source(notebook, 5))
                update = next(
                    node for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "cfg"
                    and node.func.attr == "update"
                )
                entries = {
                    key.value: value
                    for key, value in zip(update.args[0].keys, update.args[0].values)
                    if isinstance(key, ast.Constant)
                }
                self.assertIsInstance(entries.get("run_phase"), ast.Name)
                self.assertEqual(entries["run_phase"].id, "RUN_PHASE")
                self.assertEqual(entries["checkpoint_repo"].id, "HF_CHECKPOINT_REPO")

                instructor = next(
                    node for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "get_secret"
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "OPENCODE_API_KEY"
                )
                required = next(
                    keyword.value for keyword in instructor.keywords if keyword.arg == "required"
                )
                expected = ast.parse('RUN_PHASE in ("full", "rsi")', mode="eval").body
                self.assertEqual(ast.dump(required), ast.dump(expected))

    def _execute_receipt(self, phase, phase_result, upload=False, exit_code=0):
        notebook = self.notebooks[phase]
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            (work / "CAMPAIGN.json").write_text(json.dumps({"phase_result": phase_result}))
            model_upload = {"repository": "model", "commit": "abc"} if upload else None
            if upload:
                (work / "MODEL_UPLOAD_RECEIPT.json").write_text(json.dumps(model_upload))
            namespace = {
                "json": json,
                "Path": Path,
                "WORK": work,
                "HEAD": "source-commit",
                "SOURCE_SHA256": "source-sha",
                "RUN_PHASE": phase,
                "EXIT_CODE": exit_code,
                "DIRECT_SESSION_EXIT": 3,
                "DIRECT_QUALIFICATION_EXIT": 10,
                "RETRY_CHECKPOINT_DIRECTORY": None,
                "print": lambda *args, **kwargs: None,
            }
            try:
                exec(compile(source(notebook, 7), f"<{phase}-receipt>", "exec"), namespace)
            except RuntimeError as exc:
                assert 'Pipeline failed' in str(exc)
            receipt = json.loads((work / "ONESHOT_RECEIPT.json").read_text())
            return receipt, namespace["status"]

    def test_final_receipt_uses_successful_phase_result(self):
        for phase, (_, expected_result) in EXPECTED.items():
            with self.subTest(phase=phase):
                receipt, status = self._execute_receipt(
                    phase, expected_result, upload=phase == "audit"
                )
                self.assertEqual(receipt["run_phase"], phase)
                self.assertEqual(receipt["phase_result"], expected_result)
                expected_status = "COMPLETE_AND_EXPORTED" if phase == "audit" else "PHASE_COMPLETE"
                self.assertEqual(status, expected_status)
                self.assertEqual(receipt["status"], expected_status)
                if phase == "audit":
                    self.assertEqual(receipt["model_upload"]["commit"], "abc")
                else:
                    self.assertIsNone(receipt["model_upload"])

                receipt, status = self._execute_receipt(phase, expected_result, exit_code=1)
                self.assertEqual(status, "FAILED")
                self.assertIsNone(receipt["phase_result"])

    def test_audit_completion_still_requires_model_upload_receipt(self):
        receipt, status = self._execute_receipt("audit", "audit_complete", upload=False)
        self.assertEqual(receipt["phase_result"], "audit_complete")
        self.assertEqual(status, "EXITED_WITHOUT_COMPLETION")
        self.assertIsNone(receipt["model_upload"])

    def test_builder_is_deterministic_and_reproduces_shipped_bytes(self):
        canonical_before = copy.deepcopy(self.canonical)
        for phase, (filename, _) in EXPECTED.items():
            with self.subTest(phase=phase):
                built = builder.build_notebook(phase, self.canonical)
                self.assertEqual(self.canonical, canonical_before)
                expected = builder.render_notebook(built).encode("utf-8")
                self.assertEqual((ROOT / "colab" / filename).read_bytes(), expected)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "colab").mkdir()
            shutil.copy(CANONICAL_PATH, root / builder.CANONICAL)
            first = {path.name: path.read_bytes() for path in builder.write_notebooks(root)}
            second = {path.name: path.read_bytes() for path in builder.write_notebooks(root)}
            self.assertEqual(first, second)
            for phase, (filename, _) in EXPECTED.items():
                self.assertEqual(first[filename], (ROOT / "colab" / filename).read_bytes())


if __name__ == "__main__":
    unittest.main()
