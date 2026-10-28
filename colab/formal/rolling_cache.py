"""Local-only bounded TAR cache and resumable windowed sample ordering.

This is storage/sampler infrastructure. It does not implement a training loop or
establish that a dataset has adequate semantic or modality coverage.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tarfile

from colab.stage1.common import safe_relative, sha256
from .local_storage import local_path
from .shard_dataset import verify_tar

GiB = 1024**3


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Expected a complete lowercase SHA256')
    return value


class DatasetContract:
    """Bind the actual SHARDS.json bytes to one immutable HF repository commit."""

    def __init__(self, index_path, *, repo_id, commit, index_sha256, prefix=''):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo_id):
            raise ValueError('Expected an explicit owner/dataset repository ID')
        if not re.fullmatch('[0-9a-f]{40}', commit):
            raise ValueError('Pin a complete HF Git commit; branches/tags are mutable')
        if prefix:
            safe_relative(prefix)
        if Path(index_path).stat().st_size > 256 * 1024**2:
            raise ValueError('Shard index exceeds the 256 MiB controller bound')
        self.index_bytes = Path(index_path).read_bytes()
        if hashlib.sha256(self.index_bytes).hexdigest() != _hash(index_sha256):
            raise ValueError('SHARDS.json SHA256 differs from the pinned index')
        index = json.loads(self.index_bytes)
        if index.get('schema_version') != 1 or index.get('kind') not in {'immutable_multimodal_dataset_shards', 'spark_webdataset_release'}:
            raise ValueError('Unsupported shard index schema')
        if index.get('role') not in {'training', 'promotion', 'audit'}:
            raise ValueError('Unknown dataset role')
        self.shards, self.members, self.member_records = {}, {}, {}
        for item in index['shards']:
            name = item['path']
            safe_relative(name)
            if Path(name).name != name or not name.endswith('.tar') or name in self.shards:
                raise ValueError('Shards must have unique canonical TAR basenames')
            _hash(item['sha256'])
            if type(item['bytes']) is not int or not 0 < item['bytes'] <= 2 * GiB:
                raise ValueError('Shard size must be positive and at most 2 GiB')
            for member in item['files']:
                safe_relative(member['path'])
                _hash(member['sha256'])
                if type(member['bytes']) is not int or member['bytes'] < 0:
                    raise ValueError('Invalid member byte count')
                if member['path'] in self.members:
                    raise ValueError('A dataset member appears in multiple shards')
                self.members[member['path']] = name
                self.member_records[member['path']] = member
            self.shards[name] = item
        if not self.shards or sum(x['bytes'] for x in self.shards.values()) != index['payload_bytes']:
            raise ValueError('Shard payload accounting differs')
        self.repo_id, self.commit, self.prefix = repo_id, commit, prefix
        self.index = index
        self.binding = {'schema_version': 1, 'repo_id': repo_id, 'commit': commit, 'prefix': prefix,
                        'index_sha256': index_sha256, 'dataset_manifest_sha256': _hash(index['dataset_manifest_sha256']),
                        'media_manifest_sha256': _hash(index['media_manifest_sha256']), 'role': index['role']}
        if index['kind'] == 'spark_webdataset_release':
            self.binding.update({'rights_ledger_sha256': _hash(index['rights_ledger_sha256']),
                                 'sample_catalog_sha256': _hash(index['sample_catalog']['sha256']),
                                 'qualification_only': index['qualification_only']})
        self.identity = _digest(self.binding)

    def remote_path(self, name):
        if name not in self.shards:
            raise ValueError('Shard is absent from the pinned index')
        return self.prefix + '/' + name if self.prefix else name

    def load_catalog(self, path, *, split='train'):
        """Validate the paired WebDataset catalog and select one explicit split.

        Legacy SHARDS archives have no example catalog; callers must first freeze
        one from the approved records. No category/curriculum is inferred here.
        """
        if self.index['kind'] != 'spark_webdataset_release' or self.binding['role'] != 'training' or split not in {'train', 'dev'}:
            raise ValueError('An explicit train/dev paired WebDataset catalog is required')
        expected, path = self.index['sample_catalog'], Path(path)
        if path.stat().st_size != expected['bytes'] or path.stat().st_size > 256 * 1024**2 or sha256(path) != expected['sha256']:
            raise ValueError('Sample catalog differs from its frozen size/SHA256')
        samples, seen_ids, seen_members = [], set(), set()
        with path.open() as source:
            for line in source:
                record = json.loads(line)
                if record['id'] in seen_ids or record['shard'] not in self.shards or record['split'] not in {'train', 'dev'}:
                    raise ValueError('Duplicate sample or unknown catalog shard')
                if self.shards[record['shard']].get('split') != record['split']:
                    raise ValueError('Catalog split differs from its immutable shard split')
                seen_ids.add(record['id'])
                members = []
                for item in [record['metadata'], *record['media']]:
                    name = item['member']
                    if self.members.get(name) != record['shard']:
                        raise ValueError('Catalog media/metadata not paired in the declared shard')
                    actual = self.member_records[name]
                    if item['length'] != actual['bytes'] or item['sha256'] != actual['sha256']:
                        raise ValueError('Catalog member content binding differs from the release')
                    if 'offset_data' in actual and actual['offset_data'] != item['offset']:
                        raise ValueError('Catalog TAR offset differs from the release')
                    seen_members.add(name)
                    members.append({'path': name, 'offset_data': item['offset'], 'bytes': item['length']})
                if record['split'] == split:
                    samples.append({'sample_id': record['id'], 'members': members,
                                    **{key: record[key] for key in ['source_group', 'split', 'task', 'language']}})
        if seen_members != set(self.members):
            raise ValueError('Catalog does not enumerate the exact release member inventory')
        return samples


class HFHTTPTransport:
    """Direct pinned HTTP stream, with no snapshot_download/hf_hub_download cache.

    The optional token stays in this process. No key, signed URL, response body or
    authorization header is written to cache receipts. Exceptions are sanitized.
    """

    def __init__(self, contract, token=None, chunk_bytes=8 * 1024**2):
        self.contract, self._token, self.chunk_bytes = contract, token, chunk_bytes
        if type(chunk_bytes) is not int or not 64 * 1024 <= chunk_bytes <= 16 * 1024**2:
            raise ValueError('Bound the HTTP read buffer to 64 KiB through 16 MiB')

    def iter_bytes(self, remote_path, offset=0):
        from huggingface_hub import hf_hub_url
        from huggingface_hub.utils import build_hf_headers
        import requests
        url = hf_hub_url(self.contract.repo_id, remote_path, repo_type='dataset', revision=self.contract.commit)
        # Public downloads must not implicitly read cached/environment HF keys.
        headers = build_hf_headers(token=False if self._token is None else self._token)
        headers['Accept-Encoding'] = 'identity'
        if offset:
            headers['Range'] = f'bytes={offset}-'
        try:
            with requests.Session() as session:
                session.trust_env = False
                with session.get(url, headers=headers, stream=True, timeout=(15, 90)) as response:
                    if offset and (response.status_code != 206 or
                                   not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
                        raise ValueError('HF server did not honor the exact requested byte range')
                    if not offset and response.status_code != 200:
                        raise ValueError('HF initial shard request did not return HTTP 200')
                    if response.headers.get('Content-Encoding', 'identity') != 'identity':
                        raise ValueError('Encoded shard response cannot preserve byte offsets')
                    for block in response.iter_content(self.chunk_bytes):
                        if block:
                            yield block
        except Exception as error:
            raise RuntimeError('Pinned HF shard transfer failed; local partial remains resumable') from None


class CacheBudgetError(RuntimeError):
    pass


class ShardLease:
    def __init__(self, path, lock, expected):
        self.path, self._lock, self._expected = path, lock, expected
        self.archive = tarfile.open(path, 'r:')
        self._streams = []

    def open(self, member, *, offset_data=None):
        """Return a seekable TAR member stream; no media extraction/cache copy.

        A frozen per-example catalog may bind the actual TAR data byte offset.
        The stream must be closed before leaving the containing shard lease.
        """
        if member not in self._expected:
            raise ValueError('Member is not in this leased shard')
        if offset_data is not None and self.archive.getmember(member).offset_data != offset_data:
            raise ValueError('Catalog member offset differs from the actual pinned TAR')
        self._streams = [stream for stream in self._streams if not stream.closed]
        stream = self.archive.extractfile(member)
        self._streams.append(stream)
        return stream

    def read(self, member, *, max_bytes=64 * 1024**2, offset_data=None):
        """Convenience read with an explicit bounded decoded-input buffer."""
        if self.archive.getmember(member).size > max_bytes:
            raise ValueError('Member exceeds the bounded read; use an explicit streaming decoder')
        with self.open(member, offset_data=offset_data) as stream:
            return stream.read()

    def close(self):
        for stream in self._streams:
            stream.close()
        self.archive.close()
        self._lock.close()


class RollingShardCache:
    """One contract per local cache. Arrivals and eviction serialize via flock.

    All regular files under root count, including partials, lock allocation,
    manifests and atomic state writes. Only verified .tar payloads with no live
    shared lease can be evicted. Interrupted partials are kept and counted.
    """

    def __init__(self, contract, transport, *, root='/content/spark_mm/rolling_cache',
                 max_bytes=40_000_000_000, reserve_free_bytes=44_000_000_000, lookahead=2,
                 disk_usage=shutil.disk_usage):
        self.root = local_path(root)
        self.contract, self.transport = contract, transport
        self.max_bytes, self.reserve_free_bytes = max_bytes, reserve_free_bytes
        self.lookahead, self.disk_usage = lookahead, disk_usage
        if (type(max_bytes) is not int or max_bytes <= 0 or type(reserve_free_bytes) is not int or
                reserve_free_bytes < 0 or type(lookahead) is not int or not 0 <= lookahead <= 8):
            raise ValueError('Invalid cache ceiling, disk reserve or bounded lookahead')
        self.root.mkdir(parents=True, exist_ok=True)
        self._verified = {}
        self._consumed, self._download_counts = set(), {}
        self._seen_downloads = {name for name in contract.shards if (self.root / name).exists()}
        self._eviction_counts = {'consumed': 0, 'unconsumed': 0}
        self._redownloads, self._transferred_bytes, self._events = 0, 0, []
        with self._global():
            binding = self.root / 'CONTRACT.json'
            if binding.exists():
                if json.loads(binding.read_text()) != contract.binding:
                    raise ValueError('Cache belongs to a changed data/index/commit contract')
            else:
                self._atomic('CONTRACT.json', _canonical(contract.binding))
            index = self.root / 'SHARDS.json'
            if index.exists() and sha256(index) != contract.binding['index_sha256']:
                raise ValueError('Cached shard index changed')
            if not index.exists():
                self._atomic('SHARDS.json', contract.index_bytes)
            self._room(0)

    @contextmanager
    def _global(self):
        with os.fdopen(os.open(self.root / '.controller.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600), 'a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def occupied_bytes(self):
        occupied, seen = 0, set()
        for path in [self.root, *self.root.rglob('*')]:
            if path.is_symlink():
                raise ValueError('Symlinks are forbidden inside the shard cache')
            if path.is_file() or path.is_dir():
                info = path.stat()
                if (info.st_dev, info.st_ino) not in seen:
                    occupied += max(info.st_size, info.st_blocks * 512)
                    seen.add((info.st_dev, info.st_ino))
        return occupied

    @contextmanager
    def lease_many(self, names):
        """Atomically fit and lease one explicit working window (at most 16 TARs)."""
        names = list(dict.fromkeys(names))
        if not names or len(names) > 16 or any(name not in self.contract.shards for name in names):
            raise ValueError('Lease an explicit bounded window of known shards')
        leases = {}
        try:
            with self._global():
                remaining = sum(self.contract.shards[name]['bytes'] -
                                ((self.root / (name + '.partial')).stat().st_size
                                 if (self.root / (name + '.partial')).exists() else 0)
                                for name in names if not (self.root / name).exists())
                if remaining < 0:
                    raise ValueError('One incomplete shard exceeds its pinned size')
                self._room(remaining, names)
                for name in names:
                    path = self._ensure(name, names)
                    lock = (self.root / (name + '.lease')).open('a+b')
                    fcntl.flock(lock, fcntl.LOCK_SH)
                    try:
                        leases[name] = ShardLease(path, lock, {item['path'] for item in self.contract.shards[name]['files']})
                    except BaseException:
                        lock.close()
                        raise
                self._room(0, names)
            yield leases
        finally:
            for lease in leases.values():
                lease.close()

    @staticmethod
    def _signature(path):
        info = path.stat()
        return (info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    def _validate(self, name):
        path, item = self.root / name, self.contract.shards[name]
        signature = self._signature(path)
        if self._verified.get(name) != signature:
            if signature[1] != item['bytes'] or sha256(path) != item['sha256']:
                raise ValueError('Cached shard content/size differs from its immutable index')
            verify_tar(path, item['files'])
            self._verified[name] = signature

    def _evict_one(self, protected):
        candidates = sorted((self.root / name for name in self.contract.shards
                             if name not in protected and (self.root / name).exists()),
                            key=lambda path: (path.name not in self._consumed, path.stat().st_atime_ns))
        for path in candidates:
            with (self.root / (path.name + '.lease')).open('a+b') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                # Only complete closed payloads can leave the cache.
                self._validate(path.name)
                path.unlink()
                self._verified.pop(path.name, None)
                disposition = 'consumed' if path.name in self._consumed else 'unconsumed'
                self._eviction_counts[disposition] += 1
                self._event('evict', path.name, disposition=disposition)
                return True
        return False

    def _room(self, arriving, protected=()):
        # Reserve filesystem block rounding for an arriving file or atomic write.
        need = ((arriving + 4095) // 4096) * 4096
        while self.occupied_bytes() + need > self.max_bytes or self.disk_usage(self.root).free - need < self.reserve_free_bytes:
            if not self._evict_one(set(protected)):
                raise CacheBudgetError('No closed unleased shard can satisfy cache ceiling and free-disk reserve')

    def _atomic(self, name, payload):
        safe_relative(name)
        if Path(name).name != name:
            raise ValueError('Cache metadata must be a basename')
        self._room(len(payload))
        temporary = self.root / (name + '.new')
        with temporary.open('wb') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        temporary.replace(self.root / name)

    def _ensure(self, name, protected=()):
        if name not in self.contract.shards:
            raise ValueError('Shard is absent from the pinned index')
        path, item = self.root / name, self.contract.shards[name]
        if path.exists():
            self._validate(name)
            return path
        partial = self.root / (name + '.partial')
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > item['bytes']:
            raise ValueError('Partial shard exceeds its immutable size; inspect before discarding')
        self._room(item['bytes'] - offset, (*protected, name))
        with partial.open('ab') as stream:
            if offset < item['bytes']:
                for block in self.transport.iter_bytes(self.contract.remote_path(name), offset):
                    if not isinstance(block, bytes) or len(block) > 16 * 1024**2:
                        raise ValueError('Transport chunks must be bounded bytes')
                    if stream.tell() + len(block) > item['bytes']:
                        raise ValueError('Transport exceeded the pinned shard size')
                    if self.disk_usage(self.root).free - len(block) < self.reserve_free_bytes:
                        raise CacheBudgetError('Other local work consumed the reserved disk floor during arrival')
                    stream.write(block)
                    self._transferred_bytes += len(block)
            stream.flush(); os.fsync(stream.fileno())
        if partial.stat().st_size != item['bytes'] or sha256(partial) != item['sha256']:
            raise ValueError('Arriving shard failed exact size/SHA256; partial is not usable')
        verify_tar(partial, item['files'])
        partial.replace(path)
        self._verified[name] = self._signature(path)
        if name in self._seen_downloads:
            self._redownloads += 1
        self._seen_downloads.add(name)
        self._download_counts[name] = self._download_counts.get(name, 0) + 1
        self._event('verified_arrival', name, resumed_bytes=offset)
        self._room(0, (*protected, name))
        return path

    def _event(self, kind, shard, **details):
        self._events.append({'kind': kind, 'shard': shard, **details})
        self._events = self._events[-256:]

    def acknowledge_consumed_windows(self, sampler):
        """Prefer windows whose samples the trainer explicitly committed.

        Recompute from the sampler's committed offset on every call, including
        after restore or epoch changes. Issued/prefetched work never qualifies.
        """
        if sampler.contract.identity != self.contract.identity:
            raise ValueError('Consumption acknowledgement belongs to another data contract')
        consumed, cursor = set(), 0
        for window in sampler.windows:
            cursor += len(window['sample_indexes'])
            if cursor <= sampler.offset:
                consumed.update(window['shards'])
        with self._global():
            self._consumed = consumed

    def accounting(self):
        """Bounded process-lifetime telemetry; persist with campaign receipts."""
        return {'scope': 'process lifetime; caller must persist with campaign/checkpoint receipts',
                'data_contract': self.contract.identity, 'transferred_bytes': self._transferred_bytes,
                'verified_downloads': dict(self._download_counts), 'redownloads': self._redownloads,
                'evictions': dict(self._eviction_counts), 'consumed_shards': sorted(self._consumed),
                'recent_events': list(self._events)}

    @contextmanager
    def lease(self, name):
        """Keep open worker input protected until its stream has been closed."""
        with self._global():
            path = self._ensure(name)
            lock = (self.root / (name + '.lease')).open('a+b')
            fcntl.flock(lock, fcntl.LOCK_SH)
            try:
                self._room(0, (name,))
                lease = ShardLease(path, lock, {item['path'] for item in self.contract.shards[name]['files']})
            except BaseException:
                lock.close()
                raise
        try:
            yield lease
        finally:
            lease.close()

    def prefetch(self, names):
        """Synchronous bounded lookahead; performs no background unbounded queue."""
        names = list(dict.fromkeys(names))
        if len(names) > self.lookahead:
            raise ValueError('Requested prefetch exceeds the configured lookahead')
        with self._global():
            for name in names:
                self._ensure(name, names)

    def discard_partial(self, name):
        """Explicit operator recovery, confined to one incomplete cached payload."""
        if name not in self.contract.shards:
            raise ValueError('Unknown shard')
        with self._global():
            (self.root / (name + '.partial')).unlink(missing_ok=True)

    def save_sampler(self, sampler, name='SAMPLER.json'):
        if not re.fullmatch(r'SAMPLER(?:-[A-Za-z0-9_-]+)?\.json', name):
            raise ValueError('Sampler state must have a dedicated SAMPLER basename')
        if sampler.contract.identity != self.contract.identity:
            raise ValueError('Sampler data contract differs from this cache')
        with self._global():
            self._atomic(name, _canonical(sampler.state()))


class WindowedSampleMixer:
    """Deterministic shard-local windows mixed by sample, not semantic category.

    The catalog contains unique sample_id and nonempty media member paths. It
    must itself be frozen from the approved record split. Samples crossing a
    window's shard boundary are rejected, rather than silently exceeding memory.
    issue() advances only the issued cursor. commit_optimizer_step() acknowledges
    an ordered prefix of microbatches after the real optimizer step succeeds.
    Checkpoints retain the committed cursor and record pending work for rewind.
    """

    def __init__(self, contract, samples, *, seed, epoch=0, window_shards=4, state=None):
        if type(window_shards) is not int or not 1 <= window_shards <= 16 or type(seed) is not int or type(epoch) is not int or epoch < 0:
            raise ValueError('Invalid deterministic mixing configuration')
        self.contract, self.samples = contract, json.loads(_canonical(list(samples)))
        self.seed, self.epoch, self.window_shards = seed, epoch, window_shards
        sample_ids, dependencies = set(), []
        for sample in self.samples:
            sample_id, members = sample['sample_id'], sample['members']
            if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids or not members:
                raise ValueError('Sample IDs must be unique, with explicit nonempty member paths')
            names = [member['path'] if isinstance(member, dict) else member for member in members]
            if len(set(names)) != len(names) or any(name not in contract.members for name in names):
                raise ValueError('Sample media differ from the immutable shard member index')
            for member in members:
                if isinstance(member, dict):
                    if type(member.get('offset_data')) is not int or member['offset_data'] < 512 or member['offset_data'] % 512:
                        raise ValueError('Explicit TAR offsets must be positive 512-byte aligned data offsets')
                    actual = contract.member_records[member['path']]
                    if member.get('bytes') != actual['bytes']:
                        raise ValueError('Catalog member byte count differs from pinned member')
            sample_ids.add(sample_id)
            dependencies.append(set(contract.members[name] for name in names))
        if not self.samples:
            raise ValueError('Empty sample catalog')
        self.catalog_sha256 = _digest(self.samples)
        # Hash sorting is reproducible across Python random/shuffle versions.
        self.shard_permutation = sorted(set().union(*dependencies), key=lambda name: _digest([seed, epoch, 'shard', name]))
        self.windows, self.permutation = [], []
        for start in range(0, len(self.shard_permutation), window_shards):
            window = self.shard_permutation[start:start + window_shards]
            indexes = [index for index, required in enumerate(dependencies) if required <= set(window)]
            indexes.sort(key=lambda index: _digest([seed, epoch, 'sample', self.samples[index]['sample_id']]))
            self.windows.append({'shards': window, 'sample_indexes': indexes})
            self.permutation.extend(indexes)
        if len(set(self.permutation)) != len(self.samples):
            raise ValueError('A sample crosses a rolling window; package its media together or revise the explicit window')
        self.offset = self.issued_offset = self.optimizer_step = 0
        self.pending_microbatches, self.last_optimizer_commit = [], None
        if state is not None:
            if not isinstance(state, dict):
                raise ValueError('Sampler checkpoint state must be an object')
            expected = self.state()
            dynamic = ['offset', 'optimizer_step', 'last_optimizer_commit', 'rewind_on_restore']
            for field in dynamic:
                expected[field] = state.get(field)
            if expected != state or type(state.get('offset')) is not int or not 0 <= state['offset'] <= len(self.samples):
                raise ValueError('Resume sampler contract/permutations/offset differ')
            if type(state['optimizer_step']) is not int or not 0 <= state['optimizer_step'] <= state['offset']:
                raise ValueError('Invalid committed optimizer step')
            last = state['last_optimizer_commit']
            if state['optimizer_step'] == 0:
                if last is not None or state['offset'] != 0:
                    raise ValueError('Untrained sampler cannot claim committed samples')
            else:
                if (not isinstance(last, dict) or set(last) != {'optimizer_step', 'start_offset', 'end_offset', 'microbatches'} or
                        last.get('optimizer_step') != state['optimizer_step'] or last.get('end_offset') != state['offset'] or
                        not last['microbatches'] or type(last['start_offset']) is not int or last['start_offset'] >= state['offset']):
                    raise ValueError('Last optimizer acknowledgement differs from committed cursor')
                self._validate_microbatches(last['microbatches'], last['start_offset'], state['offset'])
            rewind = state['rewind_on_restore']
            if not isinstance(rewind, dict) or set(rewind) != {'pending_microbatches', 'issued_offset'}:
                raise ValueError('Missing explicit pending-work restore policy')
            self._validate_microbatches(rewind['pending_microbatches'], state['offset'], rewind['issued_offset'])
            self.offset = state['offset']
            self.optimizer_step = state['optimizer_step']
            self.last_optimizer_commit = json.loads(_canonical(last))
            # Model/RNG come from the same committed checkpoint. Reissue every
            # prefetched/yielded/uncommitted microbatch from that cursor.
            self.issued_offset = self.offset

    def state(self):
        return {'schema_version': 2, 'algorithm': 'windowed-shard-sample-shuffle-v2-committed',
                'data_binding': dict(self.contract.binding), 'catalog_sha256': self.catalog_sha256,
                'seed': self.seed, 'epoch': self.epoch, 'window_shards': self.window_shards,
                'shard_permutation': list(self.shard_permutation),
                'windows': [{'shards': list(window['shards']), 'sample_indexes': list(window['sample_indexes'])}
                            for window in self.windows],
                'sample_permutation': list(self.permutation), 'offset': self.offset,
                'optimizer_step': self.optimizer_step,
                'last_optimizer_commit': json.loads(_canonical(self.last_optimizer_commit)),
                'rewind_on_restore': {'issued_offset': self.issued_offset,
                                      'pending_microbatches': json.loads(_canonical(self.pending_microbatches))}}

    def current_window(self):
        cursor = 0
        for window in self.windows:
            end = cursor + len(window['sample_indexes'])
            if cursor <= self.issued_offset < end:
                return {'shards': list(window['shards']), 'first_sample_offset': cursor, 'end_sample_offset': end}
            cursor = end
        return None

    def peek(self):
        return None if self.issued_offset == len(self.permutation) else self.samples[self.permutation[self.issued_offset]]

    def _validate_microbatches(self, batches, start, end):
        if type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(self.samples):
            raise ValueError('Invalid microbatch cursor range')
        cursor, identifiers = start, set()
        if not isinstance(batches, list):
            raise ValueError('Microbatch acknowledgements must be an ordered list')
        for batch in batches:
            if not isinstance(batch, dict) or set(batch) != {'microbatch_id', 'sample_ids'}:
                raise ValueError('Malformed microbatch acknowledgement')
            identifier = batch.get('microbatch_id')
            if not isinstance(identifier, str) or not identifier or identifier in identifiers:
                raise ValueError('Microbatch identifiers must be nonempty and unique')
            identifiers.add(identifier)
            sample_ids = batch.get('sample_ids')
            if not isinstance(sample_ids, list) or not sample_ids:
                raise ValueError('Microbatch must enumerate actual deterministic sample IDs')
            expected = [self.samples[self.permutation[index]]['sample_id']
                        for index in range(cursor, min(cursor + len(sample_ids), len(self.samples)))]
            if sample_ids != expected:
                raise ValueError('Microbatch acknowledgement differs from deterministic sample order')
            cursor += len(sample_ids)
        if cursor != end:
            raise ValueError('Microbatch acknowledgements do not cover their claimed range')

    def issue(self, microbatch_id, count=1):
        """Issue deterministic data without claiming an optimizer update."""
        if not isinstance(microbatch_id, str) or not microbatch_id or type(count) is not int or count <= 0:
            raise ValueError('Issue a positive sample count with an explicit microbatch ID')
        if any(batch['microbatch_id'] == microbatch_id for batch in self.pending_microbatches):
            raise ValueError('A pending microbatch ID cannot be reused')
        if self.issued_offset + count > len(self.permutation):
            raise ValueError('Microbatch exceeds the remaining deterministic epoch')
        result = [self.samples[self.permutation[index]] for index in range(self.issued_offset, self.issued_offset + count)]
        self.pending_microbatches.append({'microbatch_id': microbatch_id,
                                         'sample_ids': [sample['sample_id'] for sample in result]})
        self.issued_offset += count
        return json.loads(_canonical(result))

    def commit_optimizer_step(self, optimizer_step, microbatch_ids):
        """Call only AFTER the corresponding real optimizer step succeeded.

        A gradient-accumulation update commits all of its ordered microbatch IDs
        together. Pending lookahead batches remain uncommitted and rewind on
        checkpoint restore. Out-of-order, skipped and duplicate acks are rejected.
        """
        microbatch_ids = list(microbatch_ids)
        if type(optimizer_step) is not int or optimizer_step != self.optimizer_step + 1 or not microbatch_ids:
            raise ValueError('Commit the next successful optimizer step and its nonempty microbatch IDs')
        batches = self.pending_microbatches[:len(microbatch_ids)]
        if [batch['microbatch_id'] for batch in batches] != microbatch_ids:
            raise ValueError('Optimizer acknowledgement must be an ordered pending microbatch prefix')
        end = self.offset + sum(len(batch['sample_ids']) for batch in batches)
        self._validate_microbatches(batches, self.offset, end)
        self.last_optimizer_commit = {'optimizer_step': optimizer_step, 'start_offset': self.offset,
                                      'end_offset': end, 'microbatches': json.loads(_canonical(batches))}
        self.offset, self.optimizer_step = end, optimizer_step
        del self.pending_microbatches[:len(microbatch_ids)]

    def advance(self, expected_sample_id, *, optimizer_step=None):
        """Single-sample convenience; an explicit successful step is mandatory."""
        sample = self.peek()
        if sample is None or sample['sample_id'] != expected_sample_id or self.pending_microbatches:
            raise ValueError('Consumption acknowledgement differs from the next deterministic sample')
        if type(optimizer_step) is not int or optimizer_step != self.optimizer_step + 1:
            raise ValueError('Consumption acknowledgement requires an explicit successful optimizer step')
        identifier = f'single-step-{optimizer_step}'
        self.issue(identifier)
        self.commit_optimizer_step(optimizer_step, [identifier])
