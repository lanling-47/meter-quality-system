# -*- coding: utf-8 -*-
import os, sys, json, subprocess, queue, threading, uuid
from pathlib import Path
from .schema import BUNDLE

class Inference:
    def __init__(self, state):
        self.state=Path(state); self.lock=threading.RLock(); self.proc=None; self.log=None
        self.health={'ready':False,'state':'not_started'}
    def close(self):
        if self.proc:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait(timeout=5)
            self.proc=None
        if self.log:self.log.close();self.log=None
    def receive(self, seconds):
        try:r=self.messages.get(timeout=seconds)
        except queue.Empty:raise RuntimeError('模型响应超时，请重试并查看运行日志')
        if r is None:raise RuntimeError('模型进程退出，请查看 gpu.log')
        if r.get('error'):raise RuntimeError(r['error'])
        return r
    def _launch(self, python, deps):
        self.close()
        env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1')
        if deps:env['PYTHONPATH']=deps
        else:env.pop('PYTHONPATH',None)
        self.log=(self.state/'gpu.log').open('a',encoding='utf-8')
        self.messages=queue.Queue(); messages=self.messages
        self.proc=subprocess.Popen([python,'-u',str(Path(__file__).with_name('gpu_worker.py')),str(BUNDLE)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,encoding='utf-8',env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        proc=self.proc
        def read():
            try:
                for line in proc.stdout:
                    try:messages.put(json.loads(line))
                    except json.JSONDecodeError:messages.put({'error':'推理返回格式错误，请查看日志'})
            finally:messages.put(None)
        threading.Thread(target=read,daemon=True).start()
        health=self.receive(90)
        if not health.get('ready'):raise RuntimeError('模型加载失败')
        return health
    def start(self):
        with self.lock:
            if self.proc and self.proc.poll() is None:return
            python=os.environ.get('REMF_GPU_PYTHON',sys.executable)
            try:
                self.health=self._launch(python,os.environ.get('REMF_GPU_DEPS',''))
            except Exception as exc:
                first_error=str(exc)
                cpu=os.environ.get('REMF_CPU_PYTHON',sys.executable)
                try:
                    self.health=self._launch(cpu,None)
                    self.health['fallback_note']='首选 CUDA 环境未能启动，现使用同一组权重的 CPU 推理。首选环境错误：'+first_error
                except Exception as cpu_exc:
                    self.close();self.health={'ready':False,'error':f'CUDA: {first_error}; CPU: {cpu_exc}'};raise
    def predict(self, values, task='diagnosis'):
        with self.lock:
            try:
                self.start(); rid=uuid.uuid4().hex
                self.proc.stdin.write(json.dumps({'id':rid,'values':values,'task':task})+'\n');self.proc.stdin.flush()
                r=self.receive(45)
                if r.get('id')!=rid:raise RuntimeError('推理响应不匹配')
                return r
            except Exception as exc:
                self.close();self.health={'ready':False,'error':str(exc)};raise
