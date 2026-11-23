# -*- coding: utf-8 -*-
import json, sqlite3, csv
from contextlib import contextmanager
from datetime import datetime
from zoneinfo import ZoneInfo
from .schema import clean_row, inspect_values

def incoming_action(old, record, origin):
    if old is None:return 'added'
    previous=json.loads(old['payload']);values=record['values']
    if old['batch']!=record['batch_id'] or old['origin']!=origin:return 'conflict'
    if previous==values:return 'skipped'
    from .schema import FEATURES, OBS
    if any(previous.get(k)!=values.get(k) for k in FEATURES):return 'conflict'
    old_ready=all(k in previous for k in OBS);new_ready=all(k in values for k in OBS)
    if old_ready and not new_ready:return 'skipped'
    if not old_ready and new_ready:return 'completed'
    return 'conflict'

def now():
    # Fixed UTC+8 avoids depending on Windows IANA timezone data.
    from datetime import timezone, timedelta
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')

class Store:
    def __init__(self,path):self.path=path
    @contextmanager
    def db(self):
        db=sqlite3.connect(self.path,timeout=30);db.row_factory=sqlite3.Row
        try:
            yield db;db.commit()
        finally:db.close()
    def initialize(self):
        with self.db() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, batch TEXT NOT NULL,
                    origin TEXT NOT NULL, payload TEXT NOT NULL, failed INTEGER NOT NULL, domains TEXT NOT NULL,
                    created TEXT NOT NULL, result TEXT);
                CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY AUTOINCREMENT, sample_id TEXT NOT NULL,
                    created TEXT NOT NULL, result TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS imports (id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT NOT NULL,
                    origin TEXT NOT NULL, added INTEGER NOT NULL, skipped INTEGER NOT NULL, created TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, sample_id TEXT NOT NULL,
                    created TEXT NOT NULL, action TEXT NOT NULL, operator TEXT NOT NULL, detail TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS scenarios (id INTEGER PRIMARY KEY AUTOINCREMENT, sample_id TEXT NOT NULL,
                    created TEXT NOT NULL, payload TEXT NOT NULL);
            ''')
            if 'completed' not in {r['name'] for r in db.execute('PRAGMA table_info(imports)')}:
                db.execute('ALTER TABLE imports ADD COLUMN completed INTEGER NOT NULL DEFAULT 0')
    def import_rows(self,records,origin,filename):
        added=skipped=completed=0;stamp=now()
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            for r in records:
                old=db.execute('SELECT payload,batch,origin FROM records WHERE id=?',(r['sample_id'],)).fetchone()
                action=incoming_action(old,r,origin)
                if action=='conflict':raise ValueError('编号已存在且数据或来源冲突：'+r['sample_id']+'。未写入本次导入，请核对数据。')
                if action=='skipped':skipped+=1;continue
                inspection=inspect_values(r['values'])
                if action=='completed':
                    db.execute('UPDATE records SET payload=?,failed=?,domains=? WHERE id=?',
                        (json.dumps(r['values'],ensure_ascii=False),int(inspection['failed']),json.dumps(inspection['domains'],ensure_ascii=False),r['sample_id']))
                    db.execute('INSERT INTO events(sample_id,created,action,operator,detail) VALUES(?,?,?,?,?)',
                        (r['sample_id'],stamp,'首次检测接续补录','本地接入',filename))
                    completed+=1;continue
                db.execute('INSERT INTO records VALUES(?,?,?,?,?,?,?,NULL)',(r['sample_id'],r['batch_id'],origin,
                    json.dumps(r['values'],ensure_ascii=False),int(inspection['failed']) if inspection['ready'] else -1,json.dumps(inspection['domains'],ensure_ascii=False),stamp))
                db.execute('INSERT INTO events(sample_id,created,action,operator,detail) VALUES(?,?,?,?,?)',
                    (r['sample_id'],stamp,'制造与检测接入' if inspection['ready'] else '制造档案接入','本地接入',filename))
                added+=1
            db.execute('INSERT INTO imports(filename,origin,added,skipped,created,completed) VALUES(?,?,?,?,?,?)',(filename,origin,added,skipped,stamp,completed))
        return {'added':added,'skipped':skipped,'completed':completed}
    def seed(self,source):
        with self.db() as db:
            if db.execute('SELECT COUNT(*) FROM records').fetchone()[0]:return
        path=source/'生产与出厂总表_10000.csv'
        if path.exists():
            with path.open(encoding='utf-8-sig',newline='') as f:rows=[clean_row(r) for r in csv.DictReader(f)]
            self.import_rows(rows,'synthetic','内置 V6 样本 · 10000 台')
    def get(self,sid):
        with self.db() as db:
            r=db.execute('SELECT * FROM records WHERE id=?',(sid,)).fetchone()
            events=[dict(e) for e in db.execute('SELECT created,action,operator,detail FROM events WHERE sample_id=? ORDER BY id',(sid,))]
        if not r:return None
        return {'sample_id':r['id'],'batch_id':r['batch'],'origin':r['origin'],'values':json.loads(r['payload']),
            'inspection':inspect_values(json.loads(r['payload'])),'created':r['created'],'events':events,
            'result':json.loads(r['result']) if r['result'] else None}
    def save_result(self,sid,result):
        text=json.dumps(result,ensure_ascii=False,allow_nan=False)
        with self.db() as db:
            db.execute('UPDATE records SET result=? WHERE id=?',(text,sid))
            db.execute('INSERT INTO runs(sample_id,created,result) VALUES(?,?,?)',(sid,now(),text))
    def add_inspection(self,sid,values,operator):
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            r=db.execute('SELECT * FROM records WHERE id=?',(sid,)).fetchone()
            if not r:raise ValueError('电表不存在')
            if r['failed']!=-1:raise ValueError('已有首次检测记录，不能覆盖；请保留原记录并使用新的检测流程')
            from .schema import OBS
            full=clean_row({'sample_id':sid,'batch_id':r['batch'],**json.loads(r['payload']),**{k:values.get(k) for k in OBS}})
            checked=inspect_values(full['values'])
            db.execute('UPDATE records SET payload=?,failed=?,domains=? WHERE id=?',
                (json.dumps(full['values'],ensure_ascii=False),int(checked['failed']),json.dumps(checked['domains'],ensure_ascii=False),sid))
            db.execute('INSERT INTO events(sample_id,created,action,operator,detail) VALUES(?,?,?,?,?)',
                (sid,now(),'补录首次检测',operator,'异常：'+' / '.join(checked['domains']) if checked['failed'] else '首次检测通过'))
        return self.get(sid)
    def save_scenario(self,payload):
        with self.db() as db:
            cursor=db.execute('INSERT INTO scenarios(sample_id,created,payload) VALUES(?,?,?)',
                (payload.get('sample_id') or '自定义参数',now(),json.dumps(payload,ensure_ascii=False)))
            return cursor.lastrowid
    def get_scenario(self,sid):
        with self.db() as db:r=db.execute('SELECT * FROM scenarios WHERE id=?',(sid,)).fetchone()
        return dict(r) if r else None
    def listing(self,q='',status='all',origin='all',page=1):
        clauses=['1=1'];args=[]
        if q:
            escaped=q.replace('\\','\\\\').replace('%','\\%').replace('_','\\_')
            clauses.append("(id LIKE ? ESCAPE '\\' OR batch LIKE ? ESCAPE '\\')");args += ['%'+escaped+'%']*2
        if status=='failed':clauses.append('failed=1')
        elif status=='passed':clauses.append('failed=0')
        elif status=='pending':clauses.append('failed=-1')
        elif status=='analyzed':clauses.append('result IS NOT NULL')
        if origin in ('synthetic','enterprise'):clauses.append('origin=?');args.append(origin)
        where=' AND '.join(clauses)
        with self.db() as db:
            total=db.execute('SELECT COUNT(*) FROM records WHERE '+where,args).fetchone()[0]
            rows=db.execute('SELECT id,batch,origin,failed,domains,result FROM records WHERE '+where+' ORDER BY CASE failed WHEN 1 THEN 0 WHEN -1 THEN 1 ELSE 2 END,id LIMIT 12 OFFSET ?',args+[(page-1)*12]).fetchall()
        return {'total':total,'page':page,'pages':max(1,(total+11)//12),'items':[dict(sample_id=r['id'],batch_id=r['batch'],origin=r['origin'],failed=r['failed']==1,pending=r['failed']==-1,domains=json.loads(r['domains']),analyzed=r['result'] is not None) for r in rows]}
    def summary(self):
        with self.db() as db:
            count=db.execute('SELECT COUNT(*),COALESCE(SUM(failed=1),0),SUM(result IS NOT NULL),COUNT(DISTINCT batch),COALESCE(SUM(failed=-1),0) FROM records').fetchone()
            origins=[dict(r) for r in db.execute('SELECT origin,COUNT(*) AS count FROM records GROUP BY origin')]
            imports=[dict(r) for r in db.execute('SELECT * FROM imports ORDER BY id DESC LIMIT 15')]
            runs=[dict(r) for r in db.execute('SELECT id,sample_id,created,result FROM runs ORDER BY id DESC LIMIT 30')]
            scenarios=[dict(r) for r in db.execute('SELECT id,sample_id,created FROM scenarios ORDER BY id DESC LIMIT 20')]
        for r in runs:
            result=json.loads(r.pop('result'));r['top']=result['candidates'][0]
        return dict(total=count[0],failed=count[1],analyzed=count[2] or 0,batches=count[3],pending=count[4],origins=origins,imports=imports,runs=runs,scenarios=scenarios)
