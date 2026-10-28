"""One cached OpenCode Go readiness call; key is an in-memory argument only.

This establishes current notebook/key/model/accounting transport. It does not
certify instructor competence or authorize bulk annotation traffic.
"""
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid

from colab.stage1.common import now, write_json

ENDPOINT = 'https://opencode.ai/inference/go/openai/v1/chat/completions'
MODEL = 'mimo-v2.6-pro'
RESERVATION_USD = .02
HISTORICAL_RESERVE_USD = .25  # Covers the earlier 12-request qualification ceiling conservatively.
CAMPAIGN_CEILING_USD = 50.0
PROMPT = ('We are implementing the English/Chinese multimodal model training pipeline. '
          'For this API integration check, return only the JSON object '
          '{"ready":true,"english":"ready","chinese":"就绪"}.')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def costs(parsed):
    usage = parsed.get('usage', {})
    provider = usage.get('opencode', {})
    values = {'cash_usd': usage.get('cost'), 'list_usd': provider.get('list_cost'),
              'allowance_usd': provider.get('allowance')}
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError('Provider cash/list/allowance accounting is incomplete')
    if not isinstance(provider.get('billing_source'), str) or not provider['billing_source']:
        raise ValueError('Provider billing source is missing')
    return {**values, 'billing_source': provider['billing_source'], 'maximum_exposure_usd': max(values.values())}


def redact(value, key):
    if isinstance(value, str):
        return value.replace(key, '[REDACTED]')
    if isinstance(value, dict):
        return {name: redact(item, key) for name, item in value.items()}
    if isinstance(value, list):
        return [redact(item, key) for item in value]
    return value


def run(key, root, publish):
    """publish(path) durably saves a sanitized intent/receipt before proceeding."""
    if not isinstance(key, str) or not key:
        raise ValueError('Colab Secret OPENCODE_API_KEY is unavailable')
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    with (root / '.readiness.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_locked(key, root, publish)


def run_locked(key, root, publish):
    path = root / 'readiness-ledger.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get('status') == 'passed':
            return {**previous, 'from_cached_receipt': True, 'new_requests': 0}
        raise RuntimeError('An existing pending/failed teacher request requires reconciliation; no automatic retry')
    if HISTORICAL_RESERVE_USD + RESERVATION_USD > CAMPAIGN_CEILING_USD:
        raise ValueError('Teacher campaign reservation exceeds the authorized cap')
    session = 'spark-formal-v2-' + uuid.uuid4().hex
    payload = json.dumps({'model': MODEL, 'stream': False, 'temperature': 0,
        'max_completion_tokens': 160, 'thinking': {'type': 'disabled'},
        'response_format': {'type': 'json_object'},
        'messages': [{'role': 'user', 'content': PROMPT}]}, ensure_ascii=False).encode()
    value = {'created_utc': now(), 'status': 'reserved', 'provider': 'OpenCode Go',
        'endpoint': ENDPOINT, 'model_requested': MODEL, 'session': session,
        'request_sha256': hashlib.sha256(payload).hexdigest(), 'retry_count': 0,
        'per_request_reservation_usd': RESERVATION_USD,
        'historical_qualification_reserve_usd': HISTORICAL_RESERVE_USD,
        'campaign_ceiling_usd': CAMPAIGN_CEILING_USD, 'ambiguous_outcome': True,
        'broad_teacher_competence_qualified': False, 'bulk_review_eligibility': 'unresolved',
        'scope': 'One current notebook/key/model/accounting readiness request only'}
    write_json(path, value)
    intent = root / ('teacher-intent-' + uuid.uuid4().hex + '.json')
    write_json(intent, value)
    intent_receipt = publish(intent)  # A failure here means no teacher request is sent.
    request = urllib.request.Request(ENDPOINT, data=payload, method='POST', headers={
        'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json',
        'User-Agent': 'spark-multimodal-research-integration/1.0', 'x-opencode-session': session})
    started = time.monotonic()
    try:
        try:
            response = urllib.request.build_opener(NoRedirect()).open(request, timeout=90)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            value['http_status'] = response.status
            raw = response.read(512 * 1024 + 1)
            if len(raw) > 512 * 1024:
                raise ValueError('Teacher readiness response exceeded its bound')
        parsed = json.loads(raw)
        value['model_returned'] = parsed.get('model')
        value['costs'] = costs(parsed)
        value['ambiguous_outcome'] = False
        choices = parsed.get('choices', [])
        answer = json.loads(choices[0]['message']['content']) if choices else None
        expected = {'ready': True, 'english': 'ready', 'chinese': '就绪'}
        value['answer_matches_expected'] = answer == expected
        value['status'] = 'passed' if (value['http_status'] == 200 and value['model_returned'] == MODEL
            and answer == expected and value['costs']['maximum_exposure_usd'] <= RESERVATION_USD) else 'failed'
        value['serving_revision_scope'] = 'Returned alias only; no immutable serving revision exposed'
    except Exception as exc:
        # Neither raw bodies/headers nor exception strings are persisted: they
        # can echo credentials or unreviewed provider content.
        value['status'] = 'failed'
        value['error_type'] = type(exc).__name__
    value['elapsed_seconds'] = time.monotonic() - started
    value['intent_receipt'] = intent_receipt
    value = redact(value, key)
    write_json(path, value)
    receipt = root / ('teacher-readiness-' + uuid.uuid4().hex + '.json')
    write_json(receipt, value)
    uploaded = publish(receipt)
    return {**value, 'receipt': uploaded, 'new_requests': 1, 'from_cached_receipt': False}
