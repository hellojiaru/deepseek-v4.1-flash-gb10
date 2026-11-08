"""Measure a short request arriving during one cold long-input request.

Uses an existing synthetic/public request fixture, keeps a new cache namespace,
and never changes service state. Run only on an explicitly reserved endpoint.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.request
import uuid


def main(a):
    key = a.key_file.read_text().strip()
    long_body = json.loads(a.fixture.read_bytes())
    long_body['cache_salt'] = 'interference-' + str(uuid.uuid4())
    long_body.update(model=a.model, max_tokens=16, stream=True, stream_options={'include_usage': True})
    short_body = dict(model=a.model, messages=[dict(role='user', content='Reply with only OK.')],
                      max_tokens=16, temperature=0, seed=42, thinking={'type':'disabled'},
                      stream=True, stream_options={'include_usage':True}, cache_salt='short-'+str(uuid.uuid4()))
    result = dict(passed=False, started=time.time(), requests=[], fixture_sha256=hashlib.sha256(a.fixture.read_bytes()).hexdigest())
    lock = threading.RLock()
    entered = threading.Event()

    def save():
        with lock:
            a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')

    def call(kind, body):
        row = dict(kind=kind, request_id=str(uuid.uuid4()), submitted=False)
        with lock:
            result['requests'].append(row)
            save()
        req = urllib.request.Request(a.base+'/chat/completions',json.dumps(body).encode(),
                {'Authorization':'Bearer '+key,'Content-Type':'application/json','X-Client-Request-Id':row['request_id']})
        http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        row.update(start=time.monotonic(), submitted=True)
        if kind == 'long':
            entered.set()
        text, first, done, usage = '', None, False, None
        try:
            with http.open(req,timeout=180) as response:
                for line in response:
                    if not line.startswith(b'data: '):
                        continue
                    raw=line[6:].strip()
                    if raw==b'[DONE]':
                        done=True
                        break
                    v=json.loads(raw)
                    assert 'error' not in v
                    usage=v.get('usage') or usage
                    for c in v.get('choices',[]):
                        part=c.get('delta',{}).get('content') or ''
                        if part:
                            first=first or time.monotonic()
                            text+=part
            row.update(end=time.monotonic(), first=first, done=done, usage=usage, text=text)
            assert done and usage and first
            row['ttft_s']=first-row['start']
            return row
        finally:
            save()
    try:
        with cf.ThreadPoolExecutor(2) as pool:
            long=pool.submit(call,'long',long_body)
            assert entered.wait(5)
            time.sleep(.75)
            short=pool.submit(call,'short',short_body)
            rlong,rshort=long.result(timeout=180),short.result(timeout=180)
        assert rlong['usage']['prompt_tokens_details']['cached_tokens']==0
        assert rshort['start']<rlong['first'] and 'OK' in rshort['text']
        result.update(passed=True, finished=time.time(),
                      short_arrival_after_long_s=rshort['start']-rlong['start'],
                      short_ttft_s=rshort['ttft_s'], long_ttft_s=rlong['ttft_s'],
                      long_input_tokens=rlong['usage']['prompt_tokens'])
    finally:
        save()
    print(json.dumps({k:v for k,v in result.items() if k!='requests'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--key-file',type=Path,required=True)
    p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
