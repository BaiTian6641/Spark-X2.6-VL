"""Isolated, hash-pinned Colab environments without system pip or ensurepip.

This module uses only the standard library. Both the notebook and this module
are verified source before execution. uv's binary is independently pinned.
"""
from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

FREE_FLOOR = 44_000_000_000
OWNER = "spark-formal-colab-environment-v1"
LOCKS = {"cpu": "requirements-preflight.lock", "cuda": "requirements.lock"}
_PROBE = """import importlib.util, json, platform, sys
print(json.dumps({'version':list(sys.version_info[:3]),
 'implementation':platform.python_implementation(), 'prefix':sys.prefix,
 'base_prefix':sys.base_prefix, 'pip':importlib.util.find_spec('pip') is not None}))
"""


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def _write(path, value):
    temporary = path.with_name(path.name + ".partial")
    if path.is_symlink() or temporary.is_symlink():
        raise ValueError("Refusing a symlink for bootstrap state")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def _inside(root, relative):
    """Reject symlinked parents; an environment's bin/python symlink is valid."""
    target = root / relative
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Bootstrap path escapes the work directory")
    current = root
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Refusing a symlinked bootstrap directory or state file")
    return target


def _environment(work):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("UV_", "PIP_", "PYTHON"))
           and k not in {"VIRTUAL_ENV", "CONDA_PREFIX", "HF_TOKEN",
                         "HUGGING_FACE_HUB_TOKEN", "OPENCODE_API_KEY"}}
    temporary = _inside(work, ".installer-tmp")
    temporary.mkdir(exist_ok=True)
    env.update(UV_NO_CONFIG="1", UV_PYTHON_DOWNLOADS="never",
               UV_CACHE_DIR=str(_inside(work, ".installer-cache")),
               UV_CREDENTIALS_DIR=str(_inside(work, ".installer-no-auth")),
               UV_CONCURRENT_DOWNLOADS="2", UV_CONCURRENT_INSTALLS="2",
               TMPDIR=str(temporary), PYTHONNOUSERSITE="1")
    return env


def _probe(python, env):
    result = subprocess.run([str(python), "-I", "-c", _PROBE], check=True,
                            capture_output=True, text=True, env=env, timeout=30)
    return json.loads(result.stdout)


def _require_python(probe):
    if probe["implementation"] != "CPython" or probe["version"][:2] != [3, 13]:
        raise RuntimeError("The reviewed lock requires CPython 3.13; the existing environment was preserved")


def installer(work, source_root):
    """Fetch one verified wheel, extracting only its hash-verified uv binary."""
    pin = json.loads((source_root / "colab/final/bootstrap.lock.json").read_text())
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("This installer is pinned for Linux x86_64 Colab")
    binary = _inside(work, f"tools/uv-{pin['version']}/uv")
    binary.parent.mkdir(parents=True, exist_ok=True)
    if binary.exists():
        if binary.stat().st_size != pin["binary_bytes"] or _sha(binary) != pin["binary_sha256"]:
            raise RuntimeError("Existing installer does not match the reviewed binary hash")
        binary.chmod(0o755)
        return binary, pin
    request = urllib.request.Request(pin["url"], headers={"User-Agent": "Spark-Colab-bootstrap/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read(pin["bytes"] + 1)
    if len(payload) != pin["bytes"] or hashlib.sha256(payload).hexdigest() != pin["sha256"]:
        raise RuntimeError("Installer wheel size or SHA256 mismatch")
    with zipfile.ZipFile(io.BytesIO(payload)) as wheel:
        content = wheel.read(pin["wheel_member"])
    if len(content) != pin["binary_bytes"] or hashlib.sha256(content).hexdigest() != pin["binary_sha256"]:
        raise RuntimeError("Installer executable SHA256 mismatch")
    temporary = _inside(work, f"tools/uv-{pin['version']}/uv.partial")
    temporary.write_bytes(content)
    temporary.chmod(0o755)
    os.replace(temporary, binary)
    return binary, pin


def install_environment(work_root, source_root, kind, base_python=None):
    """Repair/create only env-cpu or env-cuda and sync its complete hashed lock.

    An earlier failed stdlib venv with bin/python but no pip is adopted after
    checking its PEP 405 isolation. No environment directory is cleared. Readiness
    is recorded only after sync, isolated interpreter checks and pip check pass.
    """
    if kind not in LOCKS:
        raise ValueError("Expected cpu or cuda environment")
    raw_work = Path(work_root).absolute()
    if raw_work != raw_work.resolve() or any(x in {"drive", "gdrive", "MyDrive"} for x in raw_work.parts):
        raise ValueError("Use a real local work directory outside mounted Drive")
    if len(raw_work.parts) < 3:
        raise ValueError("A dedicated work directory is required")
    work = raw_work
    work.mkdir(parents=True, exist_ok=True)
    source_root = Path(source_root).resolve()
    target = _inside(work, "env-" + kind)
    lock = source_root / "colab/final" / LOCKS[kind]
    if not lock.is_file() or not lock.stat().st_size:
        raise ValueError("The reviewed nonempty requirements lock is missing")
    lock_sha = _sha(lock)
    mutex = _inside(work, ".environment-setup.lock")
    with mutex.open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = _environment(work)
        base_python = str(base_python or sys.executable)
        _require_python(_probe(base_python, env))
        owner_path = _inside(work, f"env-{kind}/.spark-environment.json")
        ready = _inside(work, f"env-{kind}/.spark-environment-ready.json")
        if owner_path.exists():
            owner = json.loads(owner_path.read_text())
            if owner.get("owner") != OWNER or owner.get("kind") != kind:
                raise ValueError("This environment belongs to a different setup; it was preserved")
        elif target.exists() and any(target.iterdir()):
            # The previous notebook's stdlib venv is the only unmarked layout
            # adopted here. Unexpected data in that directory is preserved.
            allowed = {"bin", "lib", "lib64", "include", "share", "pyvenv.cfg", ".gitignore", "CACHEDIR.TAG"}
            if not (target / "pyvenv.cfg").is_file() or any(p.name not in allowed for p in target.iterdir()):
                raise ValueError("Unrecognized existing environment directory; nothing was removed")
        cfg = target / "pyvenv.cfg"
        if cfg.is_symlink():
            raise ValueError("Refusing symlinked virtual-environment configuration")
        if cfg.exists() and "include-system-site-packages = true" in cfg.read_text().lower():
            raise ValueError("The environment exposes kernel packages; use an isolated environment")
        python = target / "bin/python"
        prior = None
        if python.exists():
            prior = _probe(python, env)
            _require_python(prior)
            if Path(prior["prefix"]).resolve() != target.resolve() or prior["prefix"] == prior["base_prefix"]:
                raise ValueError("Existing Python is not isolated inside the expected environment")
        was_ready = ready.exists() and json.loads(ready.read_text()).get("lock_sha256") == lock_sha
        # Never leave a stale success receipt after an interrupted or failed sync.
        if ready.exists():
            ready.unlink()
        pending = 0 if was_ready else (2_000_000_000 if kind == "cpu" else 24_000_000_000)
        if shutil.disk_usage(work).free - pending < FREE_FLOOR:
            raise RuntimeError("Environment setup would violate the 44GB local free floor")
        uv, pin = installer(work, source_root)
        if prior is None:
            subprocess.run([str(uv), "venv", "--no-config", "--no-cache", "--no-project", "--no-python-downloads",
                            "--python", base_python, "--allow-existing", str(target)],
                           check=True, env=env)
        target.mkdir(parents=True, exist_ok=True)
        _write(owner_path, {"owner": OWNER, "kind": kind, "schema_version": 1})
        args = [str(uv), "pip", "sync", "--no-config", "--no-cache", "--no-python-downloads",
                "--python", str(python), "--require-hashes", "--only-binary", ":all:",
                "--strict", "--default-index", "https://pypi.org/simple"]
        if kind == "cuda":
            # Selectively routes PyTorch packages; unrelated packages stay on
            # PyPI rather than being shadowed by old PyTorch-index copies.
            args += ["--torch-backend", "cu130"]
        subprocess.run(args + [str(lock)], check=True, env=env)
        installed = _probe(python, env)
        _require_python(installed)
        if Path(installed["prefix"]).resolve() != target.resolve() or not installed["pip"]:
            raise RuntimeError("Environment validation failed; setup is not ready")
        subprocess.run([str(python), "-I", "-m", "pip", "check"], check=True, env=env)
        if shutil.disk_usage(work).free < FREE_FLOOR:
            raise RuntimeError("Installation finished below the required local free floor")
        receipt = {"schema_version": 1, "owner": OWNER, "kind": kind, "ready": True,
                   "lock_sha256": lock_sha, "installer_version": pin["version"],
                   "installer_sha256": pin["binary_sha256"], "python": installed,
                   "recovered_python_without_pip": bool(prior and not prior["pip"]),
                   "checked_unix": time.time()}
        _write(ready, receipt)
        print(json.dumps({"event": "environment_setup_complete", "kind": kind,
                          "ready": True, "recovered_python_without_pip": receipt["recovered_python_without_pip"],
                          "lock_sha256": lock_sha}), flush=True)
        return str(python)
