"""Run the archived GPU experiment in a new workspace with this handoff's data."""
from datetime import datetime
from pathlib import Path
import shutil
import json
import subprocess
import sys

here = Path(__file__).resolve().parent
archive = here / 'meter_cause_v6_gpu_20261116'
run = here.parent / 'runs' / ('root_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
run.mkdir(parents=True, exist_ok=False)
for name in ('train.py', 'evaluate.py', 'data.py', 'metrics.py', 'models.py'):
    shutil.copy2(archive / name, run / name)
shutil.copytree(archive / 'vendor', run / 'vendor')
protocol = json.loads((archive / 'protocol.json').read_text(encoding='utf-8'))
protocol['source'] = str(here.parent / 'data')
(run / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
for script in ('train.py', 'evaluate.py'):
    subprocess.run([sys.executable, '-u', str(run / script)], cwd=run, check=True)
print(run)
