"""Public-artifact sanitization with no second full-weight copy.

Only explicitly opted-in public artifacts use this module. Evaluation rows and
account/private metadata are removed, while model/optimizer/RNG/cursor and
RSI-order state needed for continuation remain available.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from .common import read, write

_PRIVATE_CONFIG_KEYS = {
    'observed_compute_balance', 'observed_cu_per_hour',
    'instructor_route_eligibility_receipt', 'drive_project_folder',
}
_EVALUATION_AGGREGATE_KEYS = {
    'split', 'membership_sha256', 'by_stratum', 'scope',
    'human_audio_review', 'broad_multimodal_capability_passed',
}
_JSON_SANITIZE = {
    'STATE.json', 'config.json', 'INSTRUCTOR_LEDGER.json',
    'provenance/BASELINE.json', 'provenance/INSTRUCTOR_FORMAT_PROBE.json',
}


def public_value(value):
    """Remove protected rows/rationales/account fields, retaining resume state."""
    if isinstance(value, dict):
        if 'by_stratum' in value:
            return {key: public_value(value[key]) for key in _EVALUATION_AGGREGATE_KEYS if key in value}
        result = {}
        for key, item in value.items():
            if key in _PRIVATE_CONFIG_KEYS:
                result[key] = None
            elif key == 'rationale':
                continue
            else:
                result[key] = public_value(item)
        return result
    if isinstance(value, list):
        return [public_value(item) for item in value]
    return value


def public_artifact(source, destination, kind):
    """Build a sanitized artifact using hardlinks for non-JSON payloads."""
    source, destination = Path(source), Path(destination)
    if kind not in ('checkpoint', 'model'):
        raise ValueError('Unknown public artifact kind')
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for path in sorted(source.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(source).as_posix()
        if relative in ('INTENT.json', 'COMPUTE_BUDGET.json', 'MODEL_RELEASE.json'):
            continue
        if kind == 'model' and (relative == 'RUN.md' or relative.startswith(('evaluation/', 'design/'))):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative in _JSON_SANITIZE:
            write(target, public_value(read(path)))
        else:
            os.link(path, target)
    return destination
