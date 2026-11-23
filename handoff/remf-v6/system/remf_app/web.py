# -*- coding: utf-8 -*-
import csv, io, json, os, time, uuid, threading
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from .schema import ROOT, BUNDLE, FIELDS, FEATURES, LABELS, CAUSES, schema, clean_row
from .store import Store, now, incoming_action
from .ingest import parse_file, validate_rows
from .inference import Inference

def create_app(state=None,seed=True,load_model=True):
    state=Path(state or os.environ.get('REMF_STATE_DIR',ROOT.parent/'.remf_state'));state.mkdir(parents=True,exist_ok=True)
    source=Path(os.environ.get('REMF_SOURCE', ROOT.parents[1]/'data'))
    store=Store(state/'workbench.sqlite3');gpu=Inference(state);previews={};preview_lock=threading.Lock()
    @asynccontextmanager
    async def lifespan(app):
        store.initialize()
        if seed:store.seed(source)
        if load_model:
            try:gpu.start()
            except Exception:pass  # The UI reports actual availability; inference never substitutes fake scores.
        yield
        gpu.close()
    app=FastAPI(title='电表质量追溯 · REMF',version='1.0.0',lifespan=lifespan)
    app.state.store=store;app.state.gpu=gpu
    @app.middleware('http')
    async def local_boundary(request:Request,call_next):
        if request.method in ('POST','PUT','DELETE'):
            origin=request.headers.get('origin')
            if origin and origin!=str(request.base_url).rstrip('/'):
                return Response('跨来源写入不被允许',status_code=403)
        response=await call_next(request)
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Cache-Control']='no-store'
        return response
    @app.get('/')
    def home():return FileResponse(ROOT/'static'/'index.html')
    app.mount('/static',StaticFiles(directory=ROOT/'static'),name='static')
    @app.get('/api/schema')
    def get_schema():return schema()
    @app.get('/api/summary')
    def summary():return store.summary()
    @app.get('/api/model')
    def model():
        return {'name':'REMF V6','seeds':[17,43,101],'health':gpu.health,'features':34,'observations':12,
            'classes':8,'train_single':187,'validation_single':54,'test_single':51,
            'macro_f1':0.5424107142857143,'accuracy':0.7647058823529411,'top3':1.0,
            'scope':'模拟证据训练的单根因候选分类器；尚未完成真实企业验证。',
            'probability_note':'概率在八类单根因候选内归一化，不是独立故障发生率或真实因果确认概率。',
            'source':str(source),'state_dir':str(state),'protocol':schema()['protocol']}
    @app.get('/api/records')
    def records(q:str='',status:str='all',origin:str='all',page:int=Query(1,ge=1)):
        return store.listing(q[:100],status,origin,page)
    def require_record(sid):
        r=store.get(sid)
        if r is None:raise HTTPException(404,'未找到电表记录')
        return r
    @app.get('/api/record/{sid}')
    def record(sid:str):return require_record(sid)
    @app.get('/api/risk/record/{sid}')
    def risk_record(sid:str):
        r=require_record(sid)
        return {'sample_id':r['sample_id'],'batch_id':r['batch_id'],'origin':r['origin'],
                'values':{k:r['values'][k] for k in FEATURES}}
    @app.get('/api/risk/model')
    def risk_model():
        path=BUNDLE/'risk_evaluation.json'
        if not path.exists():raise HTTPException(503,'预测模型尚未训练完成')
        return json.loads(path.read_text(encoding='utf-8'))
    @app.post('/api/risk/predict')
    def risk_predict(payload:dict):
        try:
            supplied=payload.get('values',{})
            if not isinstance(supplied,dict):raise ValueError('制造参数须为字段对象')
            # Explicit allowlist: no T0 or post-inspection fields can reach this model.
            checked=clean_row({'sample_id':'SCENARIO','batch_id':'SCENARIO',**{k:supplied.get(k) for k in FEATURES}},manufacturing_only=True)
        except (ValueError,TypeError,AttributeError) as exc:raise HTTPException(422,str(exc))
        start=time.perf_counter()
        try:r=gpu.predict(checked['values'],task='preinspection')
        except Exception as exc:raise HTTPException(503,'预测模型暂不可用：'+str(exc))
        names=[('any','任一检测异常'),('metering','计量误差超限'),('communication','通信检测异常'),('display','显示缺段异常')]
        return {'model':'制造风险 MLP','version':r['model_version'],
            'risks':[{'key':k,'name':n,'probability':float(p)} for (k,n),p in zip(names,r['probabilities'])],
            'out_of_range':[LABELS[k] for k in r['out_of_range']],
            'tail_features':[LABELS[k] for k in r['tail_features']],
            'elapsed_ms':round((time.perf_counter()-start)*1000,1),'device':gpu.health.get('hardware'),
            'input_count':34,'inspection_used':False,
            'scope':'模拟抽样分布下的检测前风险；参数试算仅表示模型敏感性，不是工艺干预效果。'}
    @app.post('/api/risk/scenario')
    def risk_scenario(payload:dict):
        baseline=payload.get('baseline',{});current=payload.get('current',{})
        a=risk_predict({'values':baseline});b=risk_predict({'values':current})
        sid=payload.get('sample_id')
        if sid and not store.get(sid):raise HTTPException(404,'关联电表不存在')
        contents={'sample_id':sid,'created':now(),'baseline_values':{k:float(baseline[k]) for k in FEATURES},
            'current_values':{k:float(current[k]) for k in FEATURES},'baseline_result':a,'current_result':b,
            'scope':'参数敏感性试算，未修改制造记录，不是实测工艺改善。'}
        return {'id':store.save_scenario(contents)}
    @app.get('/api/risk/scenario/{scenario_id}')
    def scenario_report(scenario_id:int):
        r=store.get_scenario(scenario_id)
        if not r:raise HTTPException(404,'试算记录不存在')
        return Response(r['payload'],media_type='application/json',headers={'Content-Disposition':f'attachment; filename="risk-scenario-{scenario_id}.json"'})
    @app.post('/api/record')
    def manual(payload:dict):
        try:
            origin=payload.get('origin')
            if origin not in ['synthetic','enterprise']:raise ValueError('请选择数据来源')
            stage=payload.get('stage','complete')
            if stage not in ('manufacturing','complete'):raise ValueError('不支持的录入阶段')
            r=clean_row(payload.get('row',{}),manufacturing_only=stage=='manufacturing')
            result=store.import_rows([r],origin,'手工录入')
            return {**result,'sample_id':r['sample_id']}
        except (ValueError,TypeError,AttributeError) as exc:raise HTTPException(422,str(exc))
    @app.post('/api/record/{sid}/inspection')
    def add_inspection(sid:str,payload:dict):
        require_record(sid)
        operator=payload.get('operator','')
        if not isinstance(operator,str) or not 2<=len(operator.strip())<=80:raise HTTPException(422,'请填写 2～80 字的检测记录人')
        try:return store.add_inspection(sid,payload.get('values',{}),operator.strip())
        except (ValueError,TypeError,AttributeError) as exc:raise HTTPException(422,str(exc))
    @app.post('/api/infer/{sid}')
    def infer(sid:str):
        record=require_record(sid)
        if not record['inspection']['ready']:raise HTTPException(409,'该电表尚未检测，请先补录首次检测；检测前可使用风险试算')
        if not record['inspection']['failed']:raise HTTPException(409,'首次检测通过，无需强制根因归类')
        start=time.perf_counter()
        try:r=gpu.predict(record['values'])
        except Exception as exc:raise HTTPException(503,'模型暂不可用：'+str(exc))
        candidates=sorted([{**c,'probability':float(p)} for c,p in zip(CAUSES,r['probabilities'])],key=lambda c:-c['probability'])
        stages={}
        for c in candidates:stages[c['stage']]=stages.get(c['stage'],0)+c['probability']
        result={'model':'REMF V6','created':now(),'sample_id':sid,'candidates':candidates,
            'stages':[{'name':k,'probability':v} for k,v in sorted(stages.items(),key=lambda kv:-kv[1])],
            'low_confidence':r['low_confidence'],'temperature':r['temperature'],'threshold':r['threshold'],
            'elapsed_ms':round((time.perf_counter()-start)*1000,1),'device':gpu.health.get('hardware'),
            'weight_hashes':gpu.health.get('hashes'), 'confirmed':False,
            'scope':'八类单根因候选的校准模型概率；不代表已确认根因，多因与未知原因仍需复核。'}
        store.save_result(sid,result);return result
    @app.get('/api/report/{sid}')
    def report(sid:str):
        r=require_record(sid)
        if not r['result']:raise HTTPException(409,'请先运行模型推理')
        return Response(json.dumps(r,ensure_ascii=False,indent=2),media_type='application/json',
                        headers={'Content-Disposition':'attachment; filename="remf-diagnosis.json"'})
    @app.post('/api/import/preview')
    def preview(file:UploadFile=File(...),origin:str=Form(...)):
        if origin not in ('synthetic','enterprise'):raise HTTPException(422,'请选择数据来源')
        data=file.file.read(30*1024*1024+1)
        try:
            raw,sheets,keys=parse_file(data,file.filename or '')
            rows,errors=validate_rows(raw)
        except Exception as exc:
            raise HTTPException(422,str(exc) if isinstance(exc,ValueError) else '文件无法解析，请核对格式与内容')
        # Preview uses a snapshot. Commit rechecks all identities atomically.
        added=skipped=completed=0
        with store.db() as db:
            existing={r['id']:r for r in db.execute('SELECT id,batch,origin,payload FROM records')}
        for r in rows:
            old=existing.get(r['sample_id'])
            action=incoming_action(old,r,origin)
            if action=='added':added+=1
            elif action=='skipped':skipped+=1
            elif action=='completed':completed+=1
            else:errors.append({'row':r['sample_id'],'message':'与已有制造记录、首次检测或来源冲突，不会覆盖'})
        token=None
        if not errors:
            with preview_lock:
                for key in list(previews):
                    if previews[key]['expires']<time.time():previews.pop(key)
                if len(previews)>=3:raise HTTPException(429,'已有 3 个待确认导入，请先提交或稍后重试')
                token=uuid.uuid4().hex;previews[token]={'rows':rows,'origin':origin,'name':(file.filename or '')[:200],'expires':time.time()+900}
        pending=sum(not all(k in r['values'] for k in FIELDS) for r in rows)
        return {'token':token,'total':len(raw),'valid':len(rows),'added':added,'skipped':skipped,'completed':completed,'pending':pending,'tested':len(rows)-pending,'error_count':len(errors),
            'errors':errors[:50],'sheets':sheets,'matched_fields':len(set(keys)&set(FIELDS)),
            'ignored_fields':len(set(keys)-set(FIELDS)-{'sample_id','batch_id'}),
            'preview':[{'sample_id':r['sample_id'],'batch_id':r['batch_id']} for r in rows[:5]]}
    @app.post('/api/import/commit')
    def commit(payload:dict):
        with preview_lock:
            p=previews.pop(payload.get('token',''),None)
        if not p or p['expires']<time.time():raise HTTPException(409,'预览已过期或已提交，请重新解析文件')
        try:return store.import_rows(p['rows'],p['origin'],p['name'])
        except ValueError as exc:raise HTTPException(409,str(exc))
    @app.get('/api/template.csv')
    def template():
        out=io.StringIO();csv.writer(out).writerow([LABELS[k] for k in ['sample_id','batch_id']+FIELDS])
        return Response(out.getvalue().encode('utf-8-sig'),media_type='text/csv',headers={'Content-Disposition':'attachment; filename="remf-input-template.csv"'})
    @app.get('/api/sample-workbook')
    def workbook():
        path=source/'电表制造全链路样本_10000台_负样本10%_v6.xlsx'
        if not path.exists():raise HTTPException(404,'样本 Excel 不在配置目录中')
        return FileResponse(path,filename=path.name)
    return app
