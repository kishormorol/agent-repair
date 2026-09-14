"""Retrieve each finished archive while the independent AWS deadline is armed."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time

root = Path(__file__).resolve().parent
repo = root.parents[2]
remote = '/workspace/aws119-session/extension-20260914-v1'
host = 'ubuntu@16.61.13.26'
common = ['-i', str(repo/'outputs/aws-private/aws119-20260910'), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
          '-o', 'UserKnownHostsFile='+str(repo/'outputs/aws-private/known_hosts')]
archives = root/'archives'
retrieved = root/'retrieved'
archives.mkdir(exist_ok=True)
retrieved.mkdir(exist_ok=True)
verified = {}
deadline = dt.datetime.fromisoformat(json.loads((root/'live-checks.json').read_text())['deadline_utc'])
failures = 0


def ssh(command):
    return subprocess.check_output(['ssh', *common, host, command], text=True, stderr=subprocess.PIPE, timeout=45)


def fetch(path, target):
    subprocess.run(['scp', *common, host+':'+path, str(target)], check=True, timeout=300, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


while dt.datetime.now(dt.timezone.utc) < deadline+dt.timedelta(minutes=5):
    try:
        status = json.loads(ssh('cat '+remote+'/status.json'))
        (root/'latest-status.json').write_text(json.dumps(status, indent=2)+'\n')
        for item in status.get('archives', []):
            name = Path(item['path']).name
            if name in verified:
                continue
            target = archives/name
            if not target.exists():
                partial = target.with_suffix(target.suffix+'.part')
                fetch(item['path'], partial)
                assert hashlib.sha256(partial.read_bytes()).hexdigest() == item['sha256'], 'Downloaded archive checksum mismatch'
                partial.replace(target)
            assert hashlib.sha256(target.read_bytes()).hexdigest() == item['sha256']
            with tarfile.open(target) as archive:
                for member in archive.getmembers():
                    path = Path(member.name)
                    assert not path.is_absolute() and '..' not in path.parts
                    assert member.isdir() or member.isfile(), 'Unexpected archive member type'
                    local = retrieved/path
                    if member.isdir():
                        local.mkdir(parents=True, exist_ok=True)
                    else:
                        data = archive.extractfile(member).read()
                        local.parent.mkdir(parents=True, exist_ok=True)
                        if local.exists():
                            assert local.read_bytes() == data, 'Do not overwrite different retrieved records'
                        else:
                            local.write_bytes(data)
            verified[name] = {'sha256': item['sha256'], 'bytes':target.stat().st_size, 'verified':True}
            (root/'archive-verification.json').write_text(json.dumps(verified,indent=2)+'\n')
            print(json.dumps({'archive_downloaded_and_verified':name}),flush=True)
        failures = 0
        print(json.dumps({'observed_utc':status['observed_utc'], 'state':status['state'],
                          'active_model':status.get('active_model'), 'verified_archives':len(verified)}),flush=True)
        if status['state'] in ['complete','failed']:
            # The controller writes failed status before creating a partial archive.
            if status['state'] == 'failed' and not status.get('archives'):
                time.sleep(15)
                continue
            for name in ['controller.log','preflight.log','qwen32b.log','qwen32b-audit.log','mistral12b.log','mistral12b-audit.log','pooled-audit.log','controller-start.json']:
                try: fetch(remote+'/'+name, root/name)
                except subprocess.CalledProcessError: pass
            if status['state'] == 'complete':
                for name in ['pooled-analysis.json','pooled-analysis.trials.csv']:
                    fetch(remote+'/results/'+name, root/('remote-'+name))
            (root/'retrieval-status.json').write_text(json.dumps({'state':status['state'],'verified_archives':len(verified),
                'finished_utc':dt.datetime.now(dt.timezone.utc).isoformat()},indent=2)+'\n')
            break
    except (subprocess.SubprocessError, json.JSONDecodeError) as error:
        failures += 1
        print(json.dumps({'retrieval_attempt_failed':type(error).__name__, 'consecutive_failures':failures,
                          'detail':str(error)[-300:]}),flush=True)
        if failures >= 3:
            raise
    time.sleep(45)
