import ast
import json
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

from colab.final.common import read, validate_config
from colab.final.data import CommittedCursor, cursor_rows
from colab.final.run import directory_bytes, storage_envelope
from colab.final import launcher

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'colab' / 'Spark_Formal_Full_Finetune.ipynb'
CONFIG = read(ROOT / 'colab/final/campaign.json')


class NotebookTests(unittest.TestCase):
    def setUp(self):
        self.notebook = json.loads(NOTEBOOK.read_text())
        self.code = [''.join(cell['source']) for cell in self.notebook['cells'] if cell['cell_type'] == 'code']
    def test_notebook_is_valid_and_uses_full_git_checkout(self):
        self.assertEqual(self.notebook['nbformat'], 4)
        for index, source in enumerate(self.code):
            ast.parse(source, filename=f'notebook-cell-{index}')
        joined = '\n'.join(self.code)
        self.assertIn('git_source_sha256', joined)
        self.assertIn('"git", "clone"', joined)
        self.assertIn('"colab.final.run", "direct"', joined)
        self.assertIn('allow_codes=(DIRECT_QUALIFICATION_EXIT, DIRECT_SESSION_EXIT)', joined)
        self.assertIn('"drive_project_folder": None', joined)
        self.assertIn('INSTRUCTOR_MODEL', joined)
        for forbidden in ('colab.oneshot', 'spark-formal-runtime.zip', 'drive.mount(', 'run_campaign(\'cpu\')'):
            self.assertNotIn(forbidden, joined)

    def test_git_digest_covers_every_tracked_file(self):
        tree = ast.parse(self.code[2])
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'git_source_sha256')
        namespace = {'hashlib': __import__('hashlib'), 'subprocess': subprocess}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<notebook-digest>', 'exec'), namespace)
        digest = namespace['git_source_sha256']
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            subprocess.run(['git', 'init', '-q'], cwd=repo, check=True)
            subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=repo, check=True)
            subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=repo, check=True)
            (repo / 'a.py').write_text('x = 1\n')
            (repo / 'b.lock').write_text('locked\n')
            subprocess.run(['git', 'add', 'a.py', 'b.lock'], cwd=repo, check=True)
            subprocess.run(['git', 'commit', '-qm', 'initial'], cwd=repo, check=True)
            first = digest(repo)
            (repo / 'ignored.tmp').write_text('not tracked')
            self.assertEqual(first, digest(repo))
            (repo / 'a.py').write_text('x = 2\n')
            self.assertNotEqual(first, digest(repo))

    def test_direct_config_is_valid_drive_free_and_bounded(self):
        cfg = dict(CONFIG)
        cfg.update(work_root='/content/spark-formal-test', drive_project_folder=None,
                   instructor_enabled=True, instructor_model='mimo-v2.6-pro',
                   instructor_route_eligibility_receipt=None,
                   enforce_compute_budget=False, direct_wall_hours=22.0, max_wall_hours=110.0)
        validate_config(cfg)
        self.assertEqual(cfg['gradient_accumulation'] // cfg['microbatch_size'], 4)
        self.assertEqual(cfg['gradient_accumulation'], 8)
        from colab.final.teacher import eligibility
        self.assertEqual(eligibility(cfg)['mode'], 'operator-selected')
        self.assertEqual(cfg['max_wall_hours'], 110)
    def test_wall_budget_stops_instead_of_shortening_curriculum(self):
        from colab.final.common import ComputeBudget
        from colab.final.engine import Campaign
        campaign = object.__new__(Campaign)
        campaign.c = {'max_wall_hours': 0.001, 'enforce_compute_budget': False}
        state = {'used_cu': 0.0, 'active_seconds': 5.0, 'observed_at': None,
                 'observed_balance': None, 'rate': None}
        campaign.budget = ComputeBudget(campaign.c, state)
        with self.assertRaisesRegex(RuntimeError, 'no stage or epoch will be shortened'):
            campaign.check_wall_budget(0)



class StorageAndResumeTests(unittest.TestCase):
    def test_storage_envelope_and_hardlink_accounting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'one').write_bytes(b'abc')
            os.link(root / 'one', root / 'two')
            self.assertEqual(directory_bytes(root), 3)
            with patch('shutil.disk_usage', return_value=types.SimpleNamespace(free=200_000_000_000)):
                storage_envelope(root, CONFIG, CONFIG['initial_checkpoint_reservation_bytes'])
            with patch('shutil.disk_usage', return_value=types.SimpleNamespace(free=100_000_000_000)):
                with self.assertRaises(RuntimeError):
                    storage_envelope(root, CONFIG, CONFIG['initial_checkpoint_reservation_bytes'])

    def test_out_of_order_workers_commit_only_contiguous_prefix(self):
        cursor = CommittedCursor(list('abcdefghij'))
        with patch('torch.utils.data.get_worker_info', side_effect=[types.SimpleNamespace(id=0, num_workers=2), types.SimpleNamespace(id=1, num_workers=2)]):
            even = list(cursor_rows(cursor.order, cursor))
            odd = list(cursor_rows(cursor.order, cursor))
        self.assertEqual([x['_index'] for x in even], [0, 2, 4, 6, 8])
        self.assertEqual([x['_index'] for x in odd], [1, 3, 5, 7, 9])
        actual = [even[0], even[1], odd[0], odd[1], even[2], even[3], odd[2], odd[3]]
        for item in actual:
            cursor.record_computed(item['_index'])
        cursor.commit(len(actual))
        self.assertEqual(cursor.committed, 8)
        restored = CommittedCursor(cursor.order, cursor.state())
        self.assertEqual(next(cursor_rows(restored.order, restored))['_index'], 8)
        gap = CommittedCursor(list('abcd'))
        gap.record_computed(0)
        gap.record_computed(2)
        with self.assertRaises(ValueError):
            gap.commit(2)

    def test_launcher_returns_expected_resume_code_and_rejects_unexpected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            child = root / 'exit.py'
            child.write_text('raise SystemExit(10)\n')
            args = [os.environ.get('PYTHON', 'python'), str(child)]
            code = launcher.run_process(args, cwd=root, env=os.environ.copy(), secrets={},
                                        log_path=root / 'allowed.log', output=lambda *a, **k: None,
                                        allow_codes=(10,))
            self.assertEqual(code, 10)
            with self.assertRaises(RuntimeError):
                launcher.run_process(args, cwd=root, env=os.environ.copy(), secrets={},
                                     log_path=root / 'rejected.log', output=lambda *a, **k: None)


if __name__ == '__main__':
    unittest.main()
