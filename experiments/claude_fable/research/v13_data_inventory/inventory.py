"""Read metadata and small train/valid labels; never read test labels or video payloads."""

from collections import Counter
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re
import subprocess
import zipfile

from huggingface_hub import HfApi, get_token
import requests
from SoccerNet.utils import getListGames

OUT = Path(__file__).resolve().parent
ROOT = Path('/home/kuba/projects/football_data/xg_shots_full')
REPO = Path('/home/kuba/projects/tactifoot_vision')


class RemoteZip(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0
        self.bytes_read = 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        return self.position

    def tell(self):
        return self.position

    def read(self, count=-1):
        count = self.size - self.position if count < 0 else count
        count = min(count, self.size - self.position)
        if count <= 0:
            return b''
        if count > 2_000_000:
            raise ValueError('Metadata-only read exceeds 2 MB')
        start = self.position
        # The unique query prevents an intermediary cache from reusing a different range.
        with requests.get(self.url + f'?range_start={start}&range_count={count}',
                          headers={'Range': f'bytes={start}-{start + count - 1}'},
                          stream=True, timeout=45) as response:
            if response.status_code != 206:
                raise ValueError(f'Range request rejected: HTTP {response.status_code}')
            actual_range = response.headers.get('Content-Range', '')
            if not actual_range.startswith(f'bytes {start}-'):
                raise ValueError(f'Wrong range: {actual_range}')
            data = response.raw.read(count + 1)
            if len(data) != count:
                raise ValueError('Unexpected range response size')
        self.position += len(data)
        self.bytes_read += len(data)
        return data


def probe_video(path):
    result = subprocess.run(['ffprobe', '-v', 'quiet', '-show_entries',
                             'format=duration:stream=width,height', '-of', 'json', str(path)],
                            capture_output=True, text=True, timeout=20)
    return json.loads(result.stdout or '{}')


def main():
    api = HfApi()
    inventory = {'checked_at_utc': datetime.now(timezone.utc).isoformat()}
    inventory['hf_token_present'] = bool(get_token())
    inventory['hf_token_valid'] = bool(api.whoami().get('name'))
    main_games = set(getListGames('train') + getListGames('valid') + getListGames('test'))
    inventory['main500_game_count'] = len(main_games)
    replay = api.dataset_info('Homie0609/SoccerReplay-1988', files_metadata=True)
    replay_files = [s.rfilename for s in replay.siblings if s.rfilename.endswith('.json')]
    replay_path = next(p for p in replay_files if p.startswith('train/'))
    with requests.get('https://huggingface.co/datasets/Homie0609/SoccerReplay-1988/resolve/main/' + replay_path,
                      headers={'Authorization': 'Bearer ' + get_token(), 'Range': 'bytes=0-100'},
                      stream=True, timeout=30) as response:
        inventory['soccerreplay'] = {'gated': replay.gated, 'probe_file': replay_path,
                                    'http_status': response.status_code,
                                    'error_code': response.headers.get('X-Error-Code'),
                                    'listed_json_files': len(replay_files)}
    (OUT / 'soccerreplay_public_file_listing.json').write_text(json.dumps(replay_files, indent=2))
    inventory['soccertrack_halves'] = []
    for match in ['117092', '117093', '118575', '118576', '118577', '118578', '128057', '128058', '132831', '132877']:
        for half in ['1st', '2nd']:
            candidates = [ROOT / f'soccertrack_v2/videos/{match}/{match}_{half}.mp4',
                          REPO / f'data/soccertrack_v2/videos/{match}_panorama_{half}_half.mp4']
            found = [p for p in candidates if p.exists()]
            gsr = ROOT / f'soccertrack_v2/gsr/{match}/{match}_{half}.json.zst'
            bas = json.loads((ROOT / f'soccertrack_v2/bas/{match}_12_class_events.json').read_text())
            inventory['soccertrack_halves'].append({'match_id': match, 'half': half,
                'videos': [{'path': str(p), 'bytes': p.stat().st_size, **probe_video(p)} for p in found],
                'gsr_path': str(gsr) if gsr.exists() else None,
                'bas_match_label_counts': dict(Counter(a['label'] for a in bas['actions']))})
    inventory['soccernet_bas_2025'] = []
    info = api.dataset_info('SoccerNet/SN-BAS-2025', files_metadata=True)
    for split in ['train', 'valid']:
        size = next(s.size for s in info.siblings if s.rfilename == split + '.zip')
        games = getListGames(split, task='spotting-ball')
        remote = RemoteZip(f'https://huggingface.co/datasets/SoccerNet/SN-BAS-2025/resolve/main/{split}.zip', size)
        record = {'split': split, 'gated': info.gated, 'archive_bytes': size,
                  'sdk_games': games, 'sdk_overlap_main500': sorted(main_games.intersection(games))}
        with zipfile.ZipFile(remote) as archive:
            record['members'] = [{'name': s.filename, 'bytes': s.file_size} for s in archive.infolist()]
            record['labels'] = []
            for member in archive.infolist():
                if not member.filename.endswith('.json'):
                    continue
                if member.file_size > 2_000_000:
                    continue
                if member.flag_bits & 1:
                    record['labels'].append({'archive_member': member.filename,
                        'status': 'encrypted_password_required', 'compression': member.compress_type})
                    continue
                content = archive.read(member)
                local = OUT / 'bas_labels' / split / Path(member.filename).name
                if local.exists():
                    local = local.with_name(re.sub(r'[^a-zA-Z0-9_.-]', '_', member.filename))
                local.parent.mkdir(parents=True, exist_ok=True)
                local.write_bytes(content)
                data = json.loads(content)
                actions = data.get('annotations', data.get('actions', [])) if isinstance(data, dict) else data
                record['labels'].append({'archive_member': member.filename, 'local_path': str(local),
                    'top_level_keys': list(data) if isinstance(data, dict) else None,
                    'label_counts': dict(Counter(a.get('label', '?') for a in actions)),
                    'first_record_keys': list(actions[0]) if actions else []})
        record['metadata_bytes_downloaded'] = remote.bytes_read
        inventory['soccernet_bas_2025'].append(record)
        print(split, 'games', len(games), 'overlap', len(record['sdk_overlap_main500']),
              'label_files', len(record['labels']), 'metadata_bytes', remote.bytes_read, flush=True)
    inventory['test_labels_inspected'] = False
    (OUT / 'inventory.json').write_text(json.dumps(inventory, indent=2))
    print('Wrote', OUT / 'inventory.json', flush=True)


if __name__ == '__main__':
    main()
