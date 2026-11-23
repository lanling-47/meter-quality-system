# -*- coding: utf-8 -*-
"""Integration tests use an isolated SQLite directory and the actual frozen REMF weights."""
import csv, io, json, math, os, sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from remf_app.web import create_app
from remf_app.schema import FIELDS, OBS, LABELS, clean_row, inspect_values, CAUSES
from remf_app.ingest import validate_rows, parse_file

PROJECT=Path(__file__).resolve().parents[1]
SOURCE=PROJECT.parent/'data'
EXPERIMENT=PROJECT.parent/'research'/'meter_cause_v6_gpu_20261116'

@pytest.fixture(scope='module')
def rows():
    with (SOURCE/'生产与出厂总表_10000.csv').open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

@pytest.fixture(scope='module')
def client(tmp_path_factory,rows):
    previous={k:os.environ.get(k) for k in ['REMF_GPU_PYTHON','REMF_GPU_DEPS']}
    os.environ['REMF_GPU_PYTHON']=sys.executable;os.environ['REMF_GPU_DEPS']=''
    app=create_app(tmp_path_factory.mktemp('remf'),seed=False,load_model=True)
    with TestClient(app) as c:
        assert c.get('/api/model').json()['health']['ready']
        yield c
    for k,v in previous.items():
        if v is None:os.environ.pop(k,None)
        else:os.environ[k]=v

def test_original_excel_preview_commit_idempotent(client):
    book=next(SOURCE.glob('*.xlsx'))
    with book.open('rb') as f:
        r=client.post('/api/import/preview',files={'file':(book.name,f)},data={'origin':'synthetic'})
    assert r.status_code==200,r.text
    p=r.json();assert p['total']==10000 and p['matched_fields']==46 and p['error_count']==0,p
    assert client.get('/api/summary').json()['total']==0
    r=client.post('/api/import/commit',json={'token':p['token']})
    assert r.json()=={'added':10000,'skipped':0,'completed':0}
    assert client.post('/api/import/commit',json={'token':p['token']}).status_code==409
    s=client.get('/api/summary').json();assert (s['total'],s['failed'],s['batches'])==(10000,1000,500)

def test_normal_route_and_inference_probabilities(client,rows):
    normal=next(r for r in rows if r['y_any_fail']=='0')
    assert client.post('/api/infer/'+normal['sample_id']).status_code==409
    fault=next(r for r in rows if r['y_any_fail']=='1')
    r=client.post('/api/infer/'+fault['sample_id']);assert r.status_code==200,r.text
    result=r.json();assert result['model']=='REMF V6' and result['confirmed'] is False
    assert len(result['candidates'])==8
    assert abs(sum(c['probability'] for c in result['candidates'])-1)<1e-6
    assert all(c['component'] and c['stage'] for c in result['candidates'])
    assert result['weight_hashes'] and len(result['weight_hashes'])==3
    saved=client.get('/api/report/'+fault['sample_id']);assert saved.status_code==200
    assert saved.json()['result']==result

def test_against_frozen_test_probabilities(client):
    # Compare deployment to original test ensemble after the same temperature calibration.
    with (EXPERIMENT/'测试逐台候选原因.csv').open(encoding='utf-8-sig',newline='') as f:
        expected=[r for r in csv.DictReader(f) if r['model']=='remf_v6']
    meta=json.loads((EXPERIMENT/'selection_frozen.json').read_text(encoding='utf-8'))
    t=meta['calibration']['remf_v6']['temperature'];diff=0
    for row in expected:
        p=[math.exp(math.log(max(1e-10,float(row['p_'+c['code']])))/t) for c in CAUSES];s=sum(p);p=[v/s for v in p]
        response=client.post('/api/infer/'+row['sample_id']);assert response.status_code==200,response.text
        actual={c['code']:c['probability'] for c in response.json()['candidates']}
        diff=max(diff,max(abs(actual[c['code']]-v) for c,v in zip(CAUSES,p)))
    assert len(expected)==180
    assert diff<5e-5,diff
    (PROJECT/'remf_acceptance'/'probability_parity.json').write_text(json.dumps({'cases':len(expected),'max_absolute_error':diff,'tolerance':5e-5,'weights':'real REMF V6, seeds 17/43/101','device':'CPU'},indent=2),encoding='utf-8')

def test_manual_and_no_future_label_leak(client,rows):
    row=dict(rows[0],sample_id='0000007',batch_id='MANUAL-01',confirmed_cause_code='LCD_DRIVER',repair_target='LCD_DRIVER')
    r=client.post('/api/record',json={'row':row,'origin':'enterprise'});assert r.status_code==200,r.text
    r=client.get('/api/record/0000007').json();assert r['sample_id']=='0000007'
    assert set(r['values'])==set(FIELDS)
    assert 'confirmed_cause_code' not in r['values']
    assert client.post('/api/record',json={'row':row,'origin':'enterprise'}).json()['skipped']==1
    changed=dict(row,comm_failures=200)
    assert client.post('/api/record',json={'row':changed,'origin':'enterprise'}).status_code==422
    assert client.get('/api/record/0000007').json()['values']==r['values']

@pytest.mark.parametrize('change',[{FIELDS[0]:None},{FIELDS[0]:'NaN'},{'comm_failures':201},{'lcd_missing_segments':1.5},{'comm_trials':100},{'lcd_expected_segments':40},{'aoi_wetting_fraction_sim':1.01},{'sample_id':123}])
def test_invalid_inputs_rejected(rows,change):
    with pytest.raises(ValueError):clean_row({**rows[0],**change})

def test_threshold_boundaries_and_recomputed_flags(rows):
    row=clean_row(rows[0]);v=row['values']
    for i,k in enumerate(OBS):v[k]=1.5 if i<2 else 1 if i<10 else 10 if i==10 else 0
    assert not inspect_values(v)['failed']
    v[OBS[2]]=1.0001;assert inspect_values(v)['domains']==['计量']
    v[OBS[10]]=11;v[OBS[11]]=1;assert inspect_values(v)['domains']==['计量','通信','显示']

def test_duplicate_and_chinese_header_csv(rows):
    a=dict(rows[0]);a['sample_id']='CSV-001'
    cleaned,errors=validate_rows([a,a]);assert len(errors)==1
    sio=io.StringIO();keys=['sample_id','batch_id']+FIELDS
    writer=csv.writer(sio);writer.writerow([LABELS[k] for k in keys]);writer.writerow([a[k] for k in keys])
    raw,sheets,mapped=parse_file(sio.getvalue().encode('utf-8-sig'),'sample.csv')
    assert clean_row(raw[0])==clean_row(a)
    with pytest.raises(ValueError):parse_file(b'bad','bad.xlsx')

def test_atomic_conflict_and_cross_origin(client,rows):
    a=dict(rows[0],sample_id='NEW-ROLLBACK',batch_id='NEW-BATCH');b=dict(rows[0],comm_failures=200)
    with pytest.raises(ValueError):client.app.state.store.import_rows([clean_row(a),clean_row(b)],'synthetic','test')
    assert client.get('/api/record/NEW-ROLLBACK').status_code==404
    assert client.post('/api/record',json={'row':a,'origin':'synthetic'},headers={'Origin':'https://untrusted.example'}).status_code==403

def test_failed_import_has_no_token_and_no_write(client,rows):
    before=client.get('/api/summary').json()['total']
    out=io.StringIO();w=csv.writer(out);w.writerow(['sample_id','batch_id']);w.writerow(['BROKEN','B'])
    p=client.post('/api/import/preview',files={'file':('missing.csv',out.getvalue().encode())},data={'origin':'synthetic'}).json()
    assert p['error_count']==1 and p['token'] is None
    assert client.get('/api/summary').json()['total']==before
    assert client.get('/api/records',params={'q':"%' OR 1=1 --"}).json()['total']==0

def test_preinspection_no_detection_leak_and_parameter_sensitivity(client,rows):
    features=FIELDS[:34]
    values={k:float(rows[0][k]) for k in features}
    a=client.post('/api/risk/predict',json={'values':values})
    assert a.status_code==200,a.text
    a=a.json();assert a['input_count']==34 and a['inspection_used'] is False
    p=[r['probability'] for r in a['risks']]
    assert all(0<=v<=1 for v in p)
    assert max(p[1:])<=p[0]+1e-6 and p[0]<=sum(p[1:])+1e-6
    contaminated={**values,**{k:9999 for k in OBS},'confirmed_cause_code':'LCD_DRIVER','y_any_fail':1}
    b=client.post('/api/risk/predict',json={'values':contaminated}).json()
    assert a['risks']==b['risks']
    changed={**values,'power_incoming_lot_margin_proxy':0.4,'cleaning_ionic_measurement_sim_ug_cm2':0.6}
    c=client.post('/api/risk/predict',json={'values':changed}).json()
    assert max(abs(r['probability']-s['probability']) for r,s in zip(a['risks'],c['risks']))>1e-5
    assert c['out_of_range']
    values.pop(features[0]);assert client.post('/api/risk/predict',json={'values':values}).status_code==422

def test_preinspection_training_deployment_parity(client):
    reference=PROJECT.parent/'research'/'meter_preinspection_demo_20261119'/'test_predictions.csv'
    with reference.open(encoding='utf-8') as f:rows=list(csv.DictReader(f))[:30]
    maxerr=0
    for r in rows:
        source=client.get('/api/risk/record/'+r['sample_id']).json()
        assert len(source['values'])==34 and not any(k in source['values'] for k in OBS)
        actual=client.post('/api/risk/predict',json={'values':source['values']}).json()
        for item in actual['risks']:maxerr=max(maxerr,abs(item['probability']-float(r[item['key']])))
    assert maxerr<5e-6
    (PROJECT/'remf_acceptance'/'risk_probability_parity.json').write_text(json.dumps({'cases':len(rows),'max_absolute_error':maxerr,'input_features':34,'inspection_fields':0},indent=2),encoding='utf-8')

def test_manufacturing_pending_to_first_inspection(client,rows):
    r=next(r for r in rows if r['y_any_fail']=='1')
    row={k:r[k] for k in FIELDS[:34]};row.update(sample_id='PENDING-001',batch_id='ACCEPTANCE-B')
    before=client.get('/api/summary').json()
    saved=client.post('/api/record',json={'stage':'manufacturing','origin':'synthetic','row':row})
    assert saved.status_code==200,saved.text
    pending=client.get('/api/record/PENDING-001').json()
    assert pending['inspection']['pending'] and not pending['inspection']['ready']
    assert len(pending['values'])==34 and pending['result'] is None
    summary=client.get('/api/summary').json()
    assert summary['pending']==before['pending']+1 and summary['failed']==before['failed']
    assert client.get('/api/records?status=pending').json()['items'][0]['pending']
    assert all(not v['pending'] for v in client.get('/api/records?status=passed').json()['items'])
    assert client.post('/api/infer/PENDING-001').status_code==409
    assert client.post('/api/risk/predict',json={'values':pending['values']}).status_code==200
    obs={k:r[k] for k in OBS}
    response=client.post('/api/record/PENDING-001/inspection',json={'operator':'验收工程师','values':obs})
    assert response.status_code==200,response.text
    complete=response.json();assert complete['inspection']['ready'] and complete['inspection']['failed']
    assert all(complete['values'][k]==pending['values'][k] for k in FIELDS[:34])
    assert complete['events'][-1]['operator']=='验收工程师'
    assert client.post('/api/record/PENDING-001/inspection',json={'operator':'重复提交','values':obs}).status_code==422
    assert client.post('/api/infer/PENDING-001').status_code==200
    summary=client.get('/api/summary').json();assert summary['pending']==before['pending'] and summary['failed']==before['failed']+1

def test_manufacturing_only_csv_and_partial_detection_rejection(rows):
    row={k:rows[0][k] for k in ['sample_id','batch_id']+FIELDS[:34]}
    records,errors=validate_rows([row]);assert not errors and len(records[0]['values'])==34
    row[OBS[0]]=0.1
    records,errors=validate_rows([row]);assert errors and not records

def test_save_scenario_recomputes_and_preserves_record(client,rows):
    source=client.get('/api/risk/record/'+rows[0]['sample_id']).json()
    before=dict(source['values']);after={**before,'power_incoming_lot_margin_proxy':0.6}
    response=client.post('/api/risk/scenario',json={'sample_id':source['sample_id'],'baseline':before,'current':after,'probabilities':[.99]})
    assert response.status_code==200,response.text
    r=client.get('/api/risk/scenario/'+str(response.json()['id'])).json()
    assert r['baseline_values']==before and r['current_values']==after
    assert r['baseline_result']['inspection_used'] is False
    assert len(r['current_result']['risks'])==4
    assert client.get('/api/risk/record/'+source['sample_id']).json()['values']==before
    assert client.get('/api/summary').json()['scenarios']

def test_batch_import_can_complete_pending_without_overwrite(client,rows):
    raw=dict(rows[0],sample_id='PENDING-BULK',batch_id='BULK-B')
    manufacturing=clean_row(raw,manufacturing_only=True);full=clean_row(raw)
    store=client.app.state.store
    assert store.import_rows([manufacturing],'synthetic','manufacturing.csv')['added']==1
    output=io.StringIO();keys=['sample_id','batch_id']+FIELDS
    writer=csv.writer(output);writer.writerow(keys);writer.writerow([raw[k] for k in keys])
    response=client.post('/api/import/preview',files={'file':('full.csv',output.getvalue().encode())},data={'origin':'synthetic'}).json()
    assert response['completed']==1 and response['added']==0 and response['error_count']==0
    response=client.post('/api/import/commit',json={'token':response['token']}).json()
    assert response['completed']==1
    assert store.summary()['imports'][0]['completed']==1
    assert store.get('PENDING-BULK')['inspection']['ready']
    assert store.import_rows([manufacturing],'synthetic','repeat.csv')['skipped']==1
    tampered=clean_row(dict(raw,comm_failures=200))
    with pytest.raises(ValueError):store.import_rows([tampered],'synthetic','overwrite.csv')
    assert store.get('PENDING-BULK')['values']==full['values']

def test_split_manufacturing_sheet_does_not_hide_inspection(rows):
    from openpyxl import Workbook
    wb=Workbook();wb.remove(wb.active)
    r=rows[0]
    for name,keys in [('制造参数',['sample_id','batch_id']+FIELDS[:34]),('出厂检测',['sample_id']+OBS)]:
        ws=wb.create_sheet(name);ws.append(keys);ws.append([r[k] for k in keys])
    data=io.BytesIO();wb.save(data)
    raw,sheets,keys=parse_file(data.getvalue(),'split.xlsx')
    records,errors=validate_rows(raw)
    assert not errors and sheets==['制造参数','出厂检测']
    assert len(records[0]['values'])==46
    assert records[0]['values']==clean_row(r)['values']
