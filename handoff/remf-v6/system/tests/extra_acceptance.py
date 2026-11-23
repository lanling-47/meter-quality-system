import os,sys,csv,json,hashlib,shutil,tempfile,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'remf_acceptance'
sys.path.insert(0,str(ROOT))
os.environ['REMF_GPU_PYTHON']=sys.executable
os.environ['REMF_GPU_DEPS']=''
os.environ['REMF_SOURCE']=str(ROOT.parent/'data')
from fastapi.testclient import TestClient
from remf_app.web import create_app
from remf_app.schema import FIELDS,OBS
from remf_app.store import Store
source=Path(os.environ['REMF_SOURCE'])
with (source/'生产与出厂总表_10000.csv').open(encoding='utf-8-sig') as f:
    rows=list(csv.DictReader(f))
normal=next(r for r in rows if r['y_any_fail']=='0')
fault=next(r for r in rows if r['y_any_fail']=='1')
results=[]
def ok(id,name,details):results.append(dict(id=id,name=name,result='通过',details=details))
with tempfile.TemporaryDirectory(prefix='meter-site-') as temp:
    state=Path(temp)/'state'
    with TestClient(create_app(state,seed=False)) as c:
        t=c.get('/api/template.csv');assert t.status_code==200
        fields=next(csv.reader(t.content.decode('utf-8-sig').splitlines()));assert len(fields)==48
        b=c.get('/api/sample-workbook');assert b.status_code==200
        digest=hashlib.sha256(b.content).hexdigest();assert digest==hashlib.sha256(next(source.glob('*.xlsx')).read_bytes()).hexdigest()
        ok('B01','模板及样本下载',{'template_columns':len(fields),'workbook_sha256':digest})
        pending={k:fault[k] for k in FIELDS[:34]};pending.update(sample_id='SITE-PENDING',batch_id='SITE-B')
        assert c.post('/api/record',json={'row':pending,'origin':'synthetic','stage':'manufacturing'}).status_code==200
        obs={k:fault[k] for k in OBS}
        for operator in ['', 'x', 'x'*81]:
            assert c.post('/api/record/SITE-PENDING/inspection',json={'operator':operator,'values':obs}).status_code==422
        assert c.get('/api/record/SITE-PENDING').json()['inspection']['pending']
        ok('B02','检测记录人校验',{'rejected_lengths':[0,1,81],'pending_preserved':True})
        a=dict(normal,sample_id='000SITE-OK',batch_id='SITE-B')
        assert c.post('/api/record',json={'row':a,'origin':'enterprise'}).status_code==200
        assert c.get('/api/records?status=pending').json()['total']==1
        assert c.get('/api/records?status=passed').json()['total']==1
        assert c.get('/api/records?origin=enterprise').json()['total']==1
        assert c.get('/api/records?q=NO-SUCH-METER').json()['items']==[]
        assert c.get('/api/records?page=0').status_code==422
        ok('B03','筛选与空查询',{'pending':1,'passed':1,'enterprise':1,'unknown':0,'page_zero':422})
        assert c.post('/api/record/SITE-PENDING/inspection',json={'operator':'现场测试员','values':obs}).status_code==200
        result=c.post('/api/infer/SITE-PENDING').json();assert len(result['candidates'])==8
        rv=c.post('/api/risk/scenario',json={'sample_id':'SITE-PENDING','baseline':pending,'current':pending});assert rv.status_code==200
        saved=c.get('/api/report/SITE-PENDING').json()
    with TestClient(create_app(state,seed=False,load_model=False)) as c:
        assert c.get('/api/report/SITE-PENDING').json()==saved
        assert c.get('/api/risk/scenario/'+str(rv.json()['id'])).status_code==200
        assert c.get('/api/summary').json()['total']==2
        ok('B04','应用重开后数据持久化',{'records':2,'diagnosis_preserved':True,'scenario_preserved':True})
    backup=Path(temp)/'restored';shutil.copytree(state,backup)
    store=Store(backup/'workbench.sqlite3')
    with store.db() as db:assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    with TestClient(create_app(backup,seed=False,load_model=False)) as c:
        assert c.get('/api/report/SITE-PENDING').json()==saved
        assert c.get('/api/summary').json()['total']==2
    ok('B05','停止应用后备份恢复',{'sqlite_integrity':'ok','records':2,'record_and_diagnosis_equal':True})
    empty=Path(temp)/'empty-source';empty.mkdir()
    os.environ['REMF_SOURCE']=str(empty)
    with TestClient(create_app(Path(temp)/'empty-state',seed=True,load_model=False)) as c:
        assert c.get('/api/summary').json()['total']==0
        assert c.get('/api/sample-workbook').status_code==404
    ok('B06','企业空库初始化',{'total':0,'sample_download_without_file':404})
out={'tested_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'method':'真实模型及隔离临时库；持久化用应用生命周期重开，不等同整机断电测试','checks':results}
(OUT/'extra-acceptance.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
