"""Small integrity utilities shared by the Colab qualification release."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import uuid
import zipfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial-' + uuid.uuid4().hex)
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def now():
    return datetime.now(timezone.utc).isoformat()


def safe_relative(name):
    if not isinstance(name, str) or not name or '\\' in name:
        raise ValueError('Invalid relative artifact path')
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or '..' in path.parts or str(path) != name:
        raise ValueError('Artifact path must be canonical and stay inside its root')
    return path


def unpack_source(archive, destination, expected_sha256):
    """Verify the complete source release before writing any of its files."""
    archive, destination = Path(archive), Path(destination)
    if sha256(archive) != expected_sha256:
        raise ValueError('Source archive hash mismatch')
    with zipfile.ZipFile(archive) as zipped:
        entries = zipped.infolist()
        names = [item.filename for item in entries]
        if len(names) != len(set(names)) or sum(item.file_size for item in entries) > 10 * 1024 * 1024:
            raise ValueError('Unexpected source archive size or duplicate member')
        for item in entries:
            safe_relative(item.filename)
            if item.is_dir() or stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError('Source archive supports ordinary files only')
        manifest = json.loads(zipped.read('RELEASE_MANIFEST.json'))
        expected = {item['path']: item for item in manifest['files']}
        if len(expected) != len(manifest['files']) or set(names) != set(expected) | {'RELEASE_MANIFEST.json'}:
            raise ValueError('Source manifest and archive differ')
        verified = {}
        for name, record in expected.items():
            payload = zipped.read(name)
            if len(payload) != record['bytes'] or hashlib.sha256(payload).hexdigest() != record['sha256']:
                raise ValueError('Invalid source payload: ' + name)
            target = destination / safe_relative(name)
            if target.is_symlink() or (target.exists() and sha256(target) != record['sha256']):
                raise ValueError('Refusing to overwrite changed release source: ' + name)
            # Resolve before creating directories so pre-existing symlink parents fail.
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError('Source destination escapes release root')
            verified[name] = payload
        destination.mkdir(parents=True, exist_ok=True)
        for name, payload in verified.items():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with target.open('xb') as stream:
                    stream.write(payload)
        write_json(destination / 'RELEASE_MANIFEST.json', manifest)
    return {'archive_sha256': expected_sha256, 'files_verified': len(verified),
            'destination': str(destination)}
