"""Start the shipped ASGI entry point on a free loopback port and check both models."""
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
import uvicorn

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
with TemporaryDirectory() as state, socket.socket() as sock:
    os.environ['REMF_STATE_DIR'] = state
    os.environ['REMF_SOURCE'] = str(root.parent / 'data')
    sock.bind(('127.0.0.1', 0))
    url = 'http://127.0.0.1:' + str(sock.getsockname()[1])
    server = uvicorn.Server(uvicorn.Config('remf_main:app', log_level='warning'))
    thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]})
    thread.start()
    try:
        deadline = time.monotonic() + 120
        while not server.started:
            if not thread.is_alive() or time.monotonic() >= deadline:
                raise RuntimeError('HTTP startup failed')
            time.sleep(0.2)
        with httpx.Client(base_url=url, timeout=45, trust_env=False) as client:
            assert 'text/html' in client.get('/').raise_for_status().headers['content-type']
            health = client.get('/api/model').raise_for_status().json()['health']
            summary = client.get('/api/summary').raise_for_status().json()
            assert health['ready'] and health['risk_ready'] and summary['total'] == 10000
            sid = 'SIM6-U00037'
            cause = client.post('/api/infer/' + sid).raise_for_status().json()['candidates'][0]
            values = client.get('/api/risk/record/' + sid).raise_for_status().json()['values']
            def risk():
                result = client.post('/api/risk/predict', json={'values': values}).raise_for_status().json()
                return next(row['probability'] for row in result['risks'] if row['key'] == 'communication')
            before = risk()
            values['rs485_incoming_lot_margin_proxy'] = 0.8
            after = risk()
            assert cause['code'] == 'RS485_JOINT' and abs(cause['probability']-.862)<.001
            assert abs(before-.0218)<.0001 and abs(after-.0211)<.0001
            result = dict(sample_id=sid, device=health['device'], torch=health['torch'],
                          seeded_units=summary['total'], failed_units=summary['failed'], top=cause,
                          communication_risk_before=before, communication_risk_after=after)
            (root/'remf_acceptance'/'live-smoke.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
            print('HTTP smoke passed')
    finally:
        server.should_exit = True
        thread.join()
