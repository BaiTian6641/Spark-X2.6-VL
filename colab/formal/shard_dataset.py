"""Pack verified immutable dataset files into deterministic uncompressed shards."""
import argparse
import json
from pathlib import Path
import tarfile

from colab.stage1.common import safe_relative, sha256, write_json
from .local_storage import local_path


def verify_tar(path, expected):
    if len({item['path'] for item in expected}) != len(expected):
        raise ValueError('Duplicate paths in the immutable shard index')
    expected = {item['path']: item for item in expected}
    with tarfile.open(path, 'r:') as archive:
        members = archive.getmembers()
        if len(members) != len(expected) or {item.name for item in members} != set(expected):
            raise ValueError('TAR members differ from the immutable shard index')
        import hashlib
        for member in members:
            safe_relative(member.name)
            item = expected[member.name]
            if not member.isfile() or member.size != item['bytes']:
                raise ValueError('TAR member type/size differs')
            digest = hashlib.sha256()
            with archive.extractfile(member) as source:
                for block in iter(lambda: source.read(1024**2), b''):
                    digest.update(block)
            if digest.hexdigest() != item['sha256']:
                raise ValueError('TAR member content differs')


def package(root, output, max_bytes=1024**3):
    root, output = local_path(root), local_path(output)
    if not 1024**2 <= max_bytes <= 2 * 1024**3:
        raise ValueError('Shards must be bounded between 1 MiB and 2 GiB')
    manifest = json.loads((root / 'dataset_manifest.json').read_text())
    if manifest['role'] not in {'training', 'promotion', 'audit'}:
        raise ValueError('Unknown dataset role')
    media = json.loads((root / 'media_manifest.json').read_text())
    if sha256(root / 'media_manifest.json') != manifest['media_manifest_sha256']:
        raise ValueError('Media manifest differs from the dataset binding')
    files = {name: {'path': name, 'sha256': digest} for name, digest in manifest['record_files'].items()}
    allowed_records = {'train.jsonl', 'dev.jsonl'} if manifest['role'] == 'training' else {'test.jsonl'}
    if set(files) != allowed_records:
        raise ValueError('Unexpected record files for this dataset role')
    files.update({name: {'path': name, 'sha256': sha256(root / name)} for name in ['dataset_manifest.json', 'media_manifest.json']})
    if len({item['path'] for item in media}) != len(media):
        raise ValueError('Duplicate media names in the source manifest')
    for item in media:
        if item['path'] in files:
            raise ValueError('Media path collides with dataset metadata')
        files[item['path']] = item
    records = []
    for name, item in sorted(files.items()):
        safe_relative(name)
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('Dataset paths must remain within the prepared source')
        if sha256(path) != item['sha256'] or ('bytes' in item and path.stat().st_size != item['bytes']):
            raise ValueError('Dataset file differs from its immutable manifest: ' + name)
        records.append({'path': name, 'sha256': item['sha256'], 'bytes': path.stat().st_size})
    contract = {'dataset_manifest_sha256': sha256(root / 'dataset_manifest.json'),
                'media_manifest_sha256': manifest['media_manifest_sha256'], 'shard_max_bytes': max_bytes,
                'role': manifest['role']}
    output.mkdir(parents=True, exist_ok=True)
    intent = output / 'PACKAGE_INTENT.json'
    if intent.exists() and json.loads(intent.read_text()) != contract:
        raise ValueError('Existing shard package belongs to a different dataset contract')
    write_json(intent, contract)
    groups, current, occupied = [], [], 10240
    for item in records:
        needed = 512 + ((item['bytes'] + 511) // 512) * 512
        if needed + 10240 > max_bytes:
            raise ValueError('One file exceeds the declared shard bound')
        if current and occupied + needed > max_bytes:
            groups.append(current); current, occupied = [], 10240
        current.append(item); occupied += needed
    if current:
        groups.append(current)
    shards = []
    for index, items in enumerate(groups):
        name = f'data-{index:05d}.tar'
        destination, receipt = output / name, output / (name + '.json')
        if destination.exists():
            if not receipt.exists():
                verify_tar(destination, items)
                write_json(receipt, {'path': name, 'sha256': sha256(destination), 'bytes': destination.stat().st_size, 'files': items})
            old = json.loads(receipt.read_text())
            if old['files'] != items or old['sha256'] != sha256(destination):
                raise ValueError('Existing immutable shard differs from its receipt')
        else:
            temporary = destination.with_suffix('.tar.partial')
            with tarfile.open(temporary, 'w', format=tarfile.USTAR_FORMAT) as archive:
                for item in items:
                    info = tarfile.TarInfo(item['path']); info.size = item['bytes']
                    info.mtime = info.uid = info.gid = 0; info.mode = 0o644
                    with (root / item['path']).open('rb') as stream:
                        archive.addfile(info, stream)
            if temporary.stat().st_size > max_bytes:
                raise ValueError('Actual TAR exceeds the shard bound')
            temporary.replace(destination)
            write_json(receipt, {'path': name, 'sha256': sha256(destination), 'bytes': destination.stat().st_size, 'files': items})
        shards.append(json.loads(receipt.read_text()))
    index = {'schema_version': 1, 'kind': 'immutable_multimodal_dataset_shards', **contract,
        'dataset_version': manifest['dataset_version'], 'shards': shards,
        'payload_bytes': sum(item['bytes'] for item in shards),
        'training_ready': manifest.get('training_ready') is True,
        'compression': 'None; media are already JPEG compressed; deterministic TAR headers',
        'protected_roles_included': manifest['role'] != 'training'}
    write_json(output / 'SHARDS.json', index)
    return index


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shard-mib', type=int, default=1024)
    args = parser.parse_args()
    result = package(args.root, args.output, args.shard_mib * 1024**2)
    print(json.dumps({'shards': len(result['shards']), 'payload_bytes': result['payload_bytes'],
                      'role': result['role'], 'training_ready': result['training_ready']}, indent=2))
