"""Direct-mode lineage recovery orchestration with a fake but exact remote boundary."""
from pathlib import Path

import pytest

from colab.final.common import identity, read, write
from colab.final import run


BOUND = {'source_archive_sha256': 'a' * 64, 'policy_sha256': 'b' * 64}
INITIAL = {'revision': 'i' * 40, 'commit_sha256': 'i' * 64}
COMPLETE = {'revision': 'c' * 40, 'commit_sha256': 'c' * 64}


def documents():
    profile = {
        'passed': True,
        'binding': BOUND,
        'complete_checkpoint_bytes': 1,
        'projected_hours': 1,
        'qualification_initial_receipt': INITIAL,
        'qualification_complete_receipt': COMPLETE,
    }
    reset = {
        'stage': 'initial', 'updates': 0, 'completed_stages': [], 'rsi_rounds': [],
        'cursor': None, 'last_checkpoint': INITIAL,
    }
    proof = {
        'binding': BOUND, 'passed': True, 'fresh_process': True,
        'complete_optimizer_recovery': True, 'native_updates': 2,
        'initial_reset_complete': True, 'initial_reset_receipt': INITIAL,
        'initial_reset_state': reset, 'initial_reset_state_sha256': identity(reset),
    }
    return {'qualification-profile': profile, 'full-recovery-proof': proof}


def artifact(kind, documents, created=1):
    return {
        'backend': 'huggingface_artifact', 'kind': kind, 'revision': 'r' * 40,
        'commit_sha256': 'd' * 64, 'created_unix': created,
        'commit': {'binding': BOUND}, 'document': documents[kind],
    }


def execute(monkeypatch, tmp_path, items, docs, qualify_result=None):
    config = {
        'work_root': str(tmp_path), 'campaign_id': 'spark-test',
        'observed_compute_balance': None, 'observed_cu_per_hour': None,
        'initial_checkpoint_reservation_bytes': 1, 'cache_bytes_per_component': 1,
        'stages': [], 'instructor_rounds': 0, 'max_wall_hours': 110,
    }

    class Store:
        def discover(self):
            return items

        def restore(self, receipt, destination):
            destination = Path(destination)
            destination.mkdir(parents=True, exist_ok=True)
            filename = 'QUALIFICATION.json' if receipt['kind'] == 'qualification-profile' else 'FULL_RECOVERY_PROOF.json'
            write(destination / filename, receipt['document'])
            return receipt['commit']

    class Campaign:
        instance = None

        def __init__(self, config, paths, source_sha, token, key):
            self.c = config
            self.root = Path(config['work_root'])
            self.bound = BOUND
            self.store = Store()
            self.restored = []
            Campaign.instance = self

        def restore(self, receipt):
            self.restored.append(receipt)
            return object(), object(), {'fresh_vm': False}

        def run_formal(self, profile):
            self.formal_profile = profile
            return {}

    monkeypatch.setattr('colab.final.teacher.eligibility', lambda config: None)
    monkeypatch.setattr(run, 'direct_doctor', lambda root: None)
    monkeypatch.setattr(run, 'storage_envelope', lambda *args: None)
    monkeypatch.setattr('colab.final.assets.prepare_assets', lambda *args, **kwargs: {})
    monkeypatch.setattr('colab.final.assets.prepare_data', lambda *args, **kwargs: {
        'instructions': 1, 'original_instructions': 1, 'split_membership_sha256': 's' * 64,
    })
    monkeypatch.setattr('colab.final.engine.Campaign', Campaign)
    if qualify_result is None:
        monkeypatch.setattr('colab.final.qualify.qualify', lambda campaign: (_ for _ in ()).throw(AssertionError('unexpected requalification')))
    else:
        monkeypatch.setattr('colab.final.qualify.qualify', lambda campaign: qualify_result)
    monkeypatch.setattr('colab.final.qualify.recovery_check', lambda campaign, profile: (_ for _ in ()).throw(AssertionError('unexpected recovery rerun')))
    Campaign.instance.result = run.direct(config, 'hf-secret', 'teacher-secret', 'a' * 64)
    return Campaign.instance


def test_fresh_vm_restores_hf_profile_proof_and_initial_lineage(monkeypatch, tmp_path):
    docs = documents()
    latest = {'backend': 'huggingface', 'kind': 'checkpoint', **COMPLETE, 'revision': 'l' * 40}
    campaign = execute(monkeypatch, tmp_path, [latest, artifact('qualification-profile', docs), artifact('full-recovery-proof', docs)], docs)
    assert [item['revision'] for item in campaign.restored] == [INITIAL['revision']]
    assert campaign.formal_profile['binding'] == BOUND
    assert read(tmp_path / 'QUALIFICATION.json')['passed']
    assert read(tmp_path / 'FULL_RECOVERY_PROOF.json')['initial_reset_complete']


def test_remote_checkpoint_without_profile_never_restarts_qualification(monkeypatch, tmp_path):
    docs = documents()
    latest = {'backend': 'huggingface', 'kind': 'checkpoint', **COMPLETE}
    with pytest.raises(RuntimeError, match='without a bound qualification profile'):
        execute(monkeypatch, tmp_path, [latest], docs)


def test_failed_qualification_can_retry_before_formal_training(monkeypatch, tmp_path):
    docs = documents()
    write(tmp_path / 'CAMPAIGN.json', {'stage': 'qualification-studio', 'completed_stages': []})
    latest = {'backend': 'huggingface', 'kind': 'checkpoint', **INITIAL}
    campaign = execute(monkeypatch, tmp_path, [latest], docs, qualify_result=docs['qualification-profile'])
    assert campaign.result == 10 and campaign.restored == []
