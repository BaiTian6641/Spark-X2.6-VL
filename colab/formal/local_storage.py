"""Keep random-access working files off mounted Drive and remote filesystems."""
from pathlib import Path
import re


def local_path(value):
    path = Path(value).resolve()
    if path.is_relative_to('/content/drive'):
        raise ValueError('Use local working disk; mounted Drive is an archive destination')
    mountinfo = Path('/proc/self/mountinfo')
    if mountinfo.exists():
        matches = []
        for line in mountinfo.read_text().splitlines():
            before, after = line.split(' - ', 1)
            fields = before.split()
            mount = Path(re.sub(r'\\([0-7]{3})', lambda match: chr(int(match[1], 8)), fields[4]))
            if path.is_relative_to(mount):
                matches.append((len(mount.parts), after.split()[0]))
        if matches:
            filesystem = max(matches)[1]
            if filesystem in {'nfs', 'nfs4', 'cifs', 'smb3', 'fuse', 'fuseblk'} or filesystem.startswith('fuse.'):
                raise ValueError('Working files require local disk; detected ' + filesystem)
    return path
