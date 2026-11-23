"""Package this experiment without dependencies, temporary probes, or QA images."""
from pathlib import Path
import json,hashlib,zipfile

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/(ROOT.name+'_训练测试交付.zip')
excluded={'qa_render','__pycache__','cuda_download_probe.bin','numpy_probe.whl','delivery_manifest.json'}

def sha(b):return hashlib.sha256(b).hexdigest()
files=sorted(p for p in ROOT.rglob('*') if p.is_file() and not any(part in excluded for part in p.relative_to(ROOT).parts))
manifest={'experiment':ROOT.name,'file_count_without_manifest':len(files),
          'files':{p.relative_to(ROOT).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())} for p in files}}
mp=ROOT/'delivery_manifest.json';mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
files.append(mp)
with zipfile.ZipFile(OUT,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in files:z.write(p,ROOT.name+'/'+p.relative_to(ROOT).as_posix())
with zipfile.ZipFile(OUT) as z:
    assert z.testzip() is None
    for name,meta in manifest['files'].items():assert sha(z.read(ROOT.name+'/'+name))==meta['sha256']
result={'passed':True,'zip':str(OUT),'bytes':OUT.stat().st_size,'sha256':sha(OUT.read_bytes()),
        'zip_crc_passed':True,'file_hashes_verified':len(manifest['files'])}
(ROOT.parent/(ROOT.name+'_交付验收.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
