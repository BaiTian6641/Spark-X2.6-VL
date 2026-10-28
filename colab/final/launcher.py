"""Stdlib-only subprocess output capture for Colab/IPython and the CLI."""
from __future__ import annotations

import json
import os
import queue
import re
import signal
import subprocess
import threading
from pathlib import Path
from urllib.parse import quote


def sanitize(text, secrets=()):
    """Redact values, common credential formats and signed URL query strings."""
    for value in sorted((v for v in secrets if isinstance(v, str) and v), key=len, reverse=True):
        for variant in (value, json.dumps(value)[1:-1], quote(value, safe='')):
            text = text.replace(variant, '[REDACTED]')
    text = re.sub(r'https?://[^\s<>"\']+', lambda m: m[0].split('?')[0].split('#')[0] +
                  ('?[query redacted]' if '?' in m[0] else ''), text)
    text = re.sub(r'\b(?:hf_[A-Za-z0-9]{8,}|sk-[A-Za-z0-9_-]{8,})\b', '[REDACTED]', text)
    text = re.sub(r'(?i)\bBearer\s+[^\s,"\']+', 'Bearer [REDACTED]', text)
    text = re.sub(r'(?i)(\b(?:token|api_key|access_token)=)[^\s&"\']+', r'\1[REDACTED]', text)
    return text


def run_process(args, *, cwd, env, secrets, log_path, output=print, label='Campaign', allow_codes=()):
    """Stream both file descriptors through Python so Jupyter captures them.

    Only stdin carries credentials. A bounded queue decouples reads from notebook
    output. Ctrl-C signals the runner once, then drains its checkpoint output.
    Logs are sanitized before writing and capped at 8 MiB per invocation.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(env)
    for key in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'OPENCODE_API_KEY'):
        env.pop(key, None)
    env['PYTHONUNBUFFERED'] = '1'
    safe = lambda text: sanitize(text, secrets.values())
    messages = queue.Queue(maxsize=128)
    stopped = None
    last_line = ''
    interrupted = False
    log_bytes = 0
    log_full = False
    with log_path.open('x', encoding='utf-8') as log:
        def show(text):
            nonlocal log_bytes, log_full
            clean = safe(text).rstrip('\r\n')
            if not log_full:
                size = len(clean.encode('utf-8')) + 1
                if log_bytes + size <= 8 * 1024**2:
                    log.write(clean + '\n'); log.flush(); log_bytes += size
                else:
                    log.write('[Log size limit reached; subsequent output remains in the notebook.]\n')
                    log.flush(); log_full = True
            output(clean, flush=True)
            return clean

        process = subprocess.Popen(args, cwd=cwd, env=env, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace')

        def reader():
            try:
                while True:
                    line = process.stdout.readline(65536)
                    if not line: break
                    if len(line) == 65536 and not line.endswith('\n'):
                        # Discard the WHOLE oversized line, including its tail:
                        # a secret can straddle a read boundary.
                        while line and not line.endswith('\n'):
                            line = process.stdout.readline(65536)
                        line = '[Oversized child output line omitted.]\n'
                    messages.put(line)
            finally:
                messages.put(None)

        threading.Thread(target=reader, daemon=True).start()
        payload = json.dumps(secrets) + '\n'
        try:
            try:
                process.stdin.write(payload); process.stdin.close()
            except BrokenPipeError:
                pass  # Still drain the child's startup error.
        finally:
            del payload
        while True:
            try:
                try: line = messages.get(timeout=0.2)
                except queue.Empty: continue
                if line is None: break
                last_line = show(line)
                try: event = json.loads(last_line)
                except (ValueError, TypeError): continue
                if isinstance(event, dict) and event.get('event') == 'stopped': stopped = event
            except KeyboardInterrupt:
                if not interrupted and process.poll() is None:
                    process.send_signal(signal.SIGINT)
                interrupted = True
                show('Interrupt requested; waiting for the runner to finish its resumable save.' if label=='Campaign'
                     else 'Interrupt requested; waiting for environment setup to exit. Readiness is not assumed.')
        process.wait()
        process.stdout.close()
        show(f'{label} subprocess exit code: {process.returncode}. Sanitized log: {log_path}')
    if interrupted:
        raise KeyboardInterrupt('Runner exited after interrupt; inspect its checkpoint receipt above.')
    if process.returncode and process.returncode not in set(allow_codes):
        detail = (f"{stopped.get('error_type', 'Error')}: {stopped.get('reason', '')}" if stopped else
                  last_line or 'No diagnostic output; inspect runtime resource availability.')
        raise RuntimeError(f'{label} {args[3] if len(args)>3 else "process"} stopped '
                           f'(exit {process.returncode}): {detail[:2000]}\nSanitized log: {log_path}')
    return process.returncode
