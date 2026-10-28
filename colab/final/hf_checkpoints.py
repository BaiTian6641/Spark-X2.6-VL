"""Complete private or sanitized-public HF snapshots, pinned receipts, and verified direct restores.

The branch's checkpoint/ directory is replaced in one commit. Older receipts
retain their immutable revisions; Git history is never squashed by this runner.
No git/LFS checkout or second Hub cache copy is created on the Colab disk.
"""
import hashlib, json, os, re, shutil, time
from pathlib import Path
from .common import canonical, identity, read, write, sha, disk_guard, emit
from .assets import fetch
from .publication import public_artifact
from colab.stage1.common import safe_relative

PREFIX = 'checkpoint/'
MARKER = PREFIX + 'COMMITTED.json'
OWNER = 'CHECKPOINT_REPOSITORY.json'


def receipt_path(directory):
    directory = Path(directory)
    return directory.with_name(directory.name + '.checkpoint-receipt.json')


class HFCheckpointStore:
    def __init__(self, root, config, binding, token, api=None):
        from huggingface_hub import HfApi
        if not token:
            raise ValueError('HF_TOKEN is required for checkpoint read/write access')
        self.root, self.c, self.binding = Path(root), config, binding
        self.repo, self.token = config['checkpoint_repo'], token
        self.api = api or HfApi(token=token)
        self.owner = {'schema_version': 1, 'purpose': 'resumable-training-checkpoints',
                      'campaign_id': config['campaign_id'], 'binding': binding,
                      'public_sanitized': not bool(config['checkpoint_repo_private'])}
        self.private = self.root / 'private-upload-state' / 'huggingface'
        self.private.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _read_remote(self, path, revision, limit=2*1024**2):
        import requests
        from huggingface_hub import hf_hub_url
        session = requests.Session(); session.trust_env = False
        try:
            with session.get(hf_hub_url(self.repo, path, revision=revision),
                             headers={'Authorization': 'Bearer ' + self.token},
                             stream=True, timeout=(30, 120)) as response:
                if response.status_code != 200:
                    raise RuntimeError(f'HF checkpoint metadata read failed (HTTP {response.status_code})')
                payload = bytearray()
                for chunk in response.iter_content(65536):
                    payload.extend(chunk)
                    if len(payload) > limit:
                        raise ValueError('Oversized HF checkpoint metadata')
                return bytes(payload)
        except requests.RequestException as exc:
            raise RuntimeError('HF checkpoint metadata transport failed: ' + type(exc).__name__) from None
        finally:
            session.close()

    def _head_remote(self,path,revision):
        from huggingface_hub import get_hf_file_metadata,hf_hub_url
        return get_hf_file_metadata(hf_hub_url(self.repo,PREFIX+path,revision=revision),token=self.token,timeout=30)

    def _stream_remote_sha256(self,path,revision,expected_bytes):
        import requests
        from huggingface_hub import hf_hub_url
        session=requests.Session();session.trust_env=False
        digest=hashlib.sha256();total=0
        try:
            with session.get(hf_hub_url(self.repo,PREFIX+path,revision=revision),
                             headers={'Authorization':'Bearer '+self.token,'Accept-Encoding':'identity'},
                             stream=True,timeout=(30,120)) as response:
                if response.status_code!=200:
                    raise RuntimeError(f'HF checkpoint content read failed (HTTP {response.status_code}): '+path)
                for chunk in response.iter_content(1024*1024):
                    digest.update(chunk);total+=len(chunk)
                    if total>expected_bytes:
                        raise ValueError('Remote HF checkpoint size differs: '+path)
        except requests.RequestException:
            raise RuntimeError('HF checkpoint content transport failed: '+path) from None
        finally:
            session.close()
        if total!=expected_bytes:
            raise ValueError('Remote HF checkpoint size differs: '+path)
        return digest.hexdigest()


    def ensure_repository(self):
        """Small CPU preflight write; never publishes model weights or a key."""
        from huggingface_hub import CommitOperationAdd
        if self.api.whoami().get('name') != 'TassP0lar':
            raise ValueError('Unexpected HF checkpoint account')
        private=bool(self.c['checkpoint_repo_private'])
        if not self.api.repo_exists(self.repo, repo_type='model'):
            self.api.create_repo(self.repo, repo_type='model', private=private)
        info = self.api.model_info(self.repo)
        if bool(info.private) != private:
            raise RuntimeError(f'Checkpoint repository visibility must be {"private" if private else "public"}')
        names = set(self.api.list_repo_files(self.repo, repo_type='model', revision=info.sha)) if info.sha else set()
        if OWNER in names:
            if any(n not in {'.gitattributes', 'README.md', OWNER} and not n.startswith(PREFIX) for n in names):
                raise RuntimeError('Unrelated files in checkpoint repository; no overwrite performed')
            previous = json.loads(self._read_remote(OWNER, info.sha))
            if previous != self.owner:
                # A setup repair can change the runtime hash after preflight
                # created an identity-only repository. Rebind only if the same
                # campaign has never stored a payload, including in history.
                allowed = {'.gitattributes', 'README.md', OWNER}
                if (any(previous.get(k) != self.owner[k] for k in ('schema_version','purpose','campaign_id'))
                        or names - allowed):
                    raise RuntimeError('Checkpoint repository belongs to a different campaign/source')
                history = self.api.list_repo_commits(self.repo, repo_type='model', revision=info.sha)
                if not history or len(history) > 32 or any(
                        set(self.api.list_repo_files(self.repo, repo_type='model', revision=c.commit_id)) - allowed
                        for c in history):
                    raise RuntimeError('A trained or unverified checkpoint history cannot be rebound to new source')
                result = self.api.create_commit(self.repo, repo_type='model', parent_commit=info.sha,
                    operations=[CommitOperationAdd(OWNER, canonical(self.owner))],
                    commit_message='Bind unused checkpoint destination to repaired formal runtime')
                if json.loads(self._read_remote(OWNER, result.oid)) != self.owner:
                    raise RuntimeError('HF checkpoint ownership update verification failed')
                emit('empty_checkpoint_destination_rebound', repo_id=self.repo, revision=result.oid,
                     previous_binding=previous.get('binding'), binding=self.binding, weights_uploaded=False)
                return result.oid
            return info.sha
        if names - {'.gitattributes'}:
            raise RuntimeError('Existing checkpoint destination is unrelated; choose an empty repository')
        visibility = 'private' if private else 'public sanitized'
        card = (f'---\nlicense: other\nprivate: {str(private).lower()}\n---\n'
                f'# Spark multimodal {visibility} resumable training checkpoints\n\n'
                'Full training state: weights, optimizer, scheduler, RNG and data cursor. '
                'Public snapshots are sanitized to remove protected evaluation rows and account metadata. '
                'Restore only from a verified receipt at its exact commit. Earlier snapshots remain '
                'in commit history for recovery and the controlled RSI comparison. Do not squash '
                'history during an active campaign. Updating the branch does not free history storage. '
                'See the [runtime and instructions](https://github.com/BaiTian6641/Spark-X2.6-VL) '
                'and [dataset attribution](https://huggingface.co/datasets/TassP0lar/SparkX2.6-MM).\n')
        result = self.api.create_commit(self.repo, repo_type='model', parent_commit=info.sha,
            operations=[CommitOperationAdd(OWNER, canonical(self.owner)), CommitOperationAdd('README.md', card.encode())],
            commit_message=f'Initialize {visibility} Spark checkpoint destination')
        if json.loads(self._read_remote(OWNER, result.oid)) != self.owner:
            raise RuntimeError('HF checkpoint ownership write verification failed')
        return result.oid

    def _validate_marker(self, value):
        from .storage import REQUIRED
        if (value.get('status') != 'COMMITTED' or value.get('kind') != 'checkpoint'
                or value.get('campaign_id') != self.c['campaign_id'] or value.get('binding') != self.binding):
            raise ValueError('HF checkpoint belongs to a different campaign/source or is incomplete')
        files = value['files']; names = [str(safe_relative(f['path'])) for f in files]
        if len(names) != len(set(names)) or not REQUIRED <= set(names) or 'COMMITTED.json' in names:
            raise ValueError('Incomplete or ambiguous HF checkpoint inventory')
        if sum(f['bytes'] for f in files) != value['total_bytes']:
            raise ValueError('HF checkpoint byte accounting differs')
        if any(f['bytes'] < 0 or not re.fullmatch('[0-9a-f]{64}', f['sha256']) for f in files):
            raise ValueError('Invalid checkpoint file identity')

    def _verify_remote(self, revision, marker_bytes, directory=None):
        """Verify LFS hashes from tree/HEAD and stream-hash ordinary Git blobs."""
        from huggingface_hub.hf_api import RepoFile
        value = json.loads(marker_bytes); self._validate_marker(value)
        if self._read_remote(MARKER, revision) != marker_bytes:
            raise ValueError('Remote HF checkpoint commit marker differs')
        remote = {f.path: f for f in self.api.list_repo_tree(self.repo, path_in_repo='checkpoint',
                  repo_type='model', revision=revision, recursive=True, expand=True) if isinstance(f, RepoFile)}
        if set(remote) != {PREFIX + f['path'] for f in value['files']} | {MARKER}:
            raise ValueError('Remote HF checkpoint inventory differs')
        for f in value['files']:
            item = remote[PREFIX + f['path']]
            if item.size != f['bytes']:
                raise ValueError('Remote HF checkpoint size differs: ' + f['path'])
            lfs_sha=getattr(item.lfs,'sha256',None) if item.lfs else None
            if lfs_sha:
                if lfs_sha!=f['sha256']:
                    raise ValueError('Remote HF checkpoint LFS hash differs: '+f['path'])
                continue
            if f['bytes']<=8*1024**2:
                payload=self._read_remote(PREFIX+f['path'],revision,limit=8*1024**2)
                if len(payload)!=f['bytes'] or hashlib.sha256(payload).hexdigest()!=f['sha256']:
                    raise ValueError('Remote HF checkpoint Git blob differs: '+f['path'])
                continue
            metadata=self._head_remote(f['path'],revision)
            etag=(metadata.etag or '').lower()
            if metadata.size is not None and metadata.size!=f['bytes']:
                raise ValueError('Remote HF checkpoint HEAD size differs: '+f['path'])
            if re.fullmatch('[0-9a-f]{64}',etag):
                if etag!=f['sha256']:
                    raise ValueError('Remote HF checkpoint HEAD LFS hash differs: '+f['path'])
                continue
            if not re.fullmatch('[0-9a-f]{40}',etag):
                raise RuntimeError('Large checkpoint file lacks verifiable Hub content metadata: '+f['path'])
            if self._stream_remote_sha256(f['path'],revision,f['bytes'])!=f['sha256']:
                raise ValueError('Remote HF checkpoint streamed Git blob differs: '+f['path'])
        return value

    def _receipt(self, revision, payload, seconds=0):
        value = json.loads(payload)
        return {'backend': 'huggingface', 'repo_id': self.repo, 'repo_type': 'model', 'private': bool(self.c['checkpoint_repo_private']),
                'revision': revision, 'marker_path': MARKER,
                'commit_sha256': hashlib.sha256(payload).hexdigest(), 'kind': 'checkpoint',
                'total_bytes': value['total_bytes'], 'seconds': seconds,
                'verified': True, 'binding': self.binding,
                'url': f'https://huggingface.co/{self.repo}/tree/{revision}/checkpoint'}

    def commit(self, directory):
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete
        from .storage import manifest, REQUIRED
        directory = Path(directory).resolve()
        if not directory.is_relative_to((self.root / 'checkpoints').resolve()):
            raise ValueError('Only campaign-owned checkpoints may be uploaded')
        upload_directory = directory if self.c['checkpoint_repo_private'] else public_artifact(directory, self.root/'public-artifacts'/directory.name, 'checkpoint')
        snapshot = manifest(upload_directory)
        if not REQUIRED <= {f['path'] for f in snapshot['files']} or read(directory/'BINDING.json') != self.binding:
            raise ValueError('Incomplete or foreign resumable checkpoint')
        head = self.ensure_repository()
        tx = identity([directory.name, self.binding, snapshot])
        state = self.private / tx; state.mkdir(exist_ok=True, mode=0o700)
        intent = state / 'INTENT.json'
        if intent.exists():
            saved = read(intent)
            if saved['snapshot'] != snapshot:
                raise ValueError('Closed HF checkpoint changed during upload')
        else:
            value = {'schema_version': 1, 'status': 'COMMITTED', 'kind': 'checkpoint',
                     'campaign_id': self.c['campaign_id'], 'binding': self.binding,
                     **snapshot, 'transaction': tx, 'created_unix': time.time(),
                     'runtime_boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
            saved = {'snapshot': snapshot, 'parent': head, 'marker': value}
            write(intent, saved); intent.chmod(0o600)
        payload = canonical(saved['marker']) + b'\n'
        started = time.monotonic()
        names = set(self.api.list_repo_files(self.repo, repo_type='model', revision=head))
        # An upload may have committed before its HTTP response was lost. Reconcile
        # that exact transaction instead of creating another snapshot blindly.
        previous = state / 'RESULT.json'
        if previous.exists():
            revision = read(previous)['revision']
        elif MARKER in names and self._read_remote(MARKER, head) == payload:
            revision = head
        else:
            if head != saved['parent']:
                raise RuntimeError('HF branch changed during checkpoint transaction; preserve local snapshots and reconcile')
            wanted = {PREFIX + f['path'] for f in snapshot['files']} | {MARKER}
            operations = [CommitOperationAdd(PREFIX + f['path'], upload_directory/f['path']) for f in snapshot['files']]
            operations += [CommitOperationAdd(MARKER, payload)]
            operations += [CommitOperationDelete(n) for n in sorted(names) if n.startswith(PREFIX) and n not in wanted]
            emit('hf_checkpoint_upload_start', repo_id=self.repo, bytes=snapshot['total_bytes'], files=len(snapshot['files']))
            result = self.api.create_commit(self.repo, repo_type='model', parent_commit=head,
                operations=operations, num_threads=2, commit_message='Save full resumable Spark checkpoint: ' + directory.name)
            revision = result.oid
        write(previous, {'revision': revision})
        self._verify_remote(revision, payload, directory)
        receipt = self._receipt(revision, payload, time.monotonic()-started)
        # A local deletion acknowledgement is written by ColdStore only after
        # this verification AND its small receipt has been committed to Drive.
        emit('hf_checkpoint_verified', **receipt)
        if upload_directory != directory:
            shutil.rmtree(upload_directory, ignore_errors=True)
        return receipt

    def restore(self, receipt, destination):
        if (receipt.get('backend') != 'huggingface' or receipt.get('repo_id') != self.repo
                or receipt.get('marker_path') != MARKER or receipt.get('repo_type') != 'model'
                or not re.fullmatch('[0-9a-f]{40}', receipt.get('revision', ''))):
            raise ValueError('Use an exact checkpoint receipt for the configured repository')
        revision = receipt['revision']; payload = self._read_remote(MARKER, revision)
        if hashlib.sha256(payload).hexdigest() != receipt['commit_sha256']:
            raise ValueError('HF checkpoint receipt hash differs')
        value = self._verify_remote(revision, payload)
        destination = Path(destination)
        if not destination.resolve().is_relative_to((self.root/'checkpoints').resolve()):
            raise ValueError('Restore must use the campaign local checkpoint directory')
        destination.mkdir(parents=True, exist_ok=True)
        pending = sum(f['bytes'] for f in value['files'] if not (destination/f['path']).exists())
        # The slot manager has retained at most one verified fallback snapshot.
        # Do not reserve a third checkpoint on top of the two actual slots.
        disk_guard(self.root, pending=pending)
        from huggingface_hub import hf_hub_url
        for f in value['files']:
            target = destination/f['path']
            if not target.resolve().is_relative_to(destination.resolve()) or target.is_symlink():
                raise ValueError('Unsafe HF checkpoint restore path')
            fetch(hf_hub_url(self.repo, PREFIX+f['path'], revision=revision), target, f, self.token)
        write(receipt_path(destination), {**receipt, 'verified': True, 'binding': self.binding})
        return value

    def latest(self):
        head = self.ensure_repository()
        if MARKER not in self.api.list_repo_files(self.repo, repo_type='model', revision=head):
            return None
        payload = self._read_remote(MARKER, head)
        self._verify_remote(head, payload)
        return self._receipt(head, payload)
