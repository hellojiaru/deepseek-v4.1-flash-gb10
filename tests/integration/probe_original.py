"""Finite original-weight API acceptance and mixed-workload reference.

No steering asset is needed. Use a reserved endpoint; no retries or service
management. Request identities and partial responses are retained in --output.
"""
import argparse
import base64
import concurrent.futures as cf
import hashlib
import json
from pathlib import Path
import statistics
import struct
import threading
import time
import urllib.error
import urllib.request
import uuid
import zlib

BASE_MODEL = 'deepseek-v4.1-flash-dsgb10-official-1017'
WORKLOADS = {
    'code': 'Implement a Python LRU cache with TTL, thread safety and a controllable clock. Include tests and explain boundary cases. Give complete working code.',
    'prose': 'Write a practical guide for a small library to digitize its collection. Discuss scanning, metadata, backups, access and the order of work. Use detailed natural prose.',
    'agent_plan': 'You are maintaining a Python package whose CI intermittently hangs. Plan how to reproduce, isolate, fix and verify the issue. Explain concrete observations and decisions before each tool action.',
    'json': 'Output only a JSON array of 12 fictional warehouse stock items. Each must have sku, title, quantity, location and reorder_level. Use varied valid values.',
    'math': 'A workshop has 37 kits. Each kit uses 6 bolts. Bolts come in packs of 25 at 4.20 dollars per pack. Find packs needed, spare bolts and total cost. Explain the arithmetic and check it another way.',
    'reasoning': 'Design an experiment to distinguish a slow tokenizer, cold model prefill, queueing and a slow decoder in an HTTP language model server. Explain measurements and ambiguous cases.',
    'translation': 'Translate into English and explain your choices: 这个系统能处理很长的资料，但阅读新资料和继续生成答案是两种不同的工作。我们希望记录用户实际等待的时间，而不是只报告理想条件下的最高速度。',
    'summary': 'Summarize this deployment note into actions and risks: The release changes only the API frontend. The model weights and kernels stay fixed. Requests can remain queued for several minutes, so a short outer timeout can hide a live backend request. The team must drain new admission, let accepted work finish, snapshot configuration, verify process identity, deploy the paired frontend and check tools, images and cache reuse. A successful build alone does not qualify a release.',
}


def run(a):
    key = a.key_file.read_text().strip()
    model = BASE_MODEL if a.label == 'candidate' else BASE_MODEL + '-normal'
    http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    lock = threading.RLock()
    result = dict(passed=False, started=time.time(), label=a.label, model=model,
                  requests=[], mixed_workload=[], checks={}, missing=[])

    def save():
        with lock:
            a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    def request(messages, n=128, **kw):
        body = dict(model=model, messages=messages, max_tokens=n, temperature=0, seed=42,
                    thinking={'type': 'disabled'}, stream=True, stream_options={'include_usage': True}, **kw)
        row = dict(request_id=str(uuid.uuid4()), submitted=False, done=False)
        with lock:
            result['requests'].append(row)
            save()
        req = urllib.request.Request(a.base + '/chat/completions', json.dumps(body).encode(),
                {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key,
                 'X-Client-Request-Id': row['request_id']})
        start = time.monotonic()
        row.update(submitted=True, start=start)
        save()
        text, reasoning, calls, first, usage = '', '', {}, None, None
        try:
            with http.open(req, timeout=180) as response:
                for line in response:
                    if not line.startswith(b'data: '):
                        continue
                    raw = line[6:].strip()
                    if raw == b'[DONE]':
                        row['done'] = True
                        break
                    value = json.loads(raw)
                    assert 'error' not in value
                    usage = value.get('usage') or usage
                    for choice in value.get('choices', []):
                        delta = choice.get('delta', {})
                        if delta.get('content') or delta.get('reasoning_content') or delta.get('tool_calls'):
                            first = first or time.monotonic()
                        text += delta.get('content') or ''
                        reasoning += delta.get('reasoning_content') or ''
                        for call in delta.get('tool_calls', []):
                            c = calls.setdefault(call['index'], dict(id='', name='', arguments=''))
                            c['id'] += call.get('id') or ''
                            c['name'] += call.get('function', {}).get('name') or ''
                            c['arguments'] += call.get('function', {}).get('arguments') or ''
            end = time.monotonic()
            assert row['done'] and usage and first
            row.update(end=end, first=first, ttft_s=first-start, wall_s=end-start, usage=usage,
                       decode_tps=max(0, usage['completion_tokens']-1)/max(1e-9, end-first),
                       text=text, reasoning=reasoning, calls=list(calls.values()),
                       output_sha256=hashlib.sha256((reasoning+text+json.dumps(calls,sort_keys=True)).encode()).hexdigest())
            return row
        except BaseException as error:
            row.update(error=type(error).__name__, end=time.monotonic())
            raise
        finally:
            save()

    def user(text):
        return [dict(role='user', content=text)]

    try:
        for repeat in range(3):
            for name, prompt in WORKLOADS.items():
                row = request(user(prompt), 512)
                result['mixed_workload'].append(dict(workload=name, repeat=repeat, request_id=row['request_id']))
                save()
        if a.label == 'candidate':
            req = urllib.request.Request(a.base + '/models', headers={'Authorization': 'Bearer ' + key})
            with http.open(req, timeout=10) as f:
                models = json.load(f)['data']
            assert [x['id'] for x in models] == [BASE_MODEL]
            result['checks']['single_original_model_listed'] = True
            rejected = urllib.request.Request(a.base + '/chat/completions',
                        json.dumps(dict(model=BASE_MODEL+'-creative',messages=user('Reply OK.'),max_tokens=1)).encode(),
                        {'Authorization':'Bearer '+key,'Content-Type':'application/json'})
            try:
                with http.open(rejected, timeout=10):
                    raise AssertionError('steered alias accepted without an asset')
            except urllib.error.HTTPError as error:
                assert error.code == 400
                result['checks']['creative_alias_rejected'] = error.code
            tool = dict(type='function', function=dict(name='lookup_stock', description='Look up a stock item',
                        parameters=dict(type='object',properties=dict(sku=dict(type='string')),required=['sku'])))
            messages = user('Use lookup_stock to look up SKU R42.')
            row = request(messages, 128, tools=[tool], tool_choice=dict(type='function',function=dict(name='lookup_stock')))
            assert len(row['calls']) == 1 and row['calls'][0]['name'] == 'lookup_stock'
            call = row['calls'][0]
            assert json.loads(call['arguments'])['sku'] == 'R42'
            messages += [dict(role='assistant',content=row['text'] or None,tool_calls=[dict(id=call['id'],type='function',function=dict(name=call['name'],arguments=call['arguments']))]),
                         dict(role='tool',tool_call_id=call['id'],content='{"quantity":37}'),dict(role='user',content='Reply with only the quantity returned by the tool.')]
            assert '37' in request(messages, 32)['text']
            result['checks']['tool_roundtrip'] = True
            def chunk(k, data):
                return struct.pack('>I',len(data))+k+data+struct.pack('>I',zlib.crc32(k+data)&0xffffffff)
            pixels = b''.join(b'\0'+b'\xff\0\0'*32+b'\0\0\xff'*32 for _ in range(32))
            png = b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',64,32,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(pixels))+chunk(b'IEND',b'')
            image = 'data:image/png;base64,'+base64.b64encode(png).decode()
            row = request([dict(role='user',content=[dict(type='text',text='Name the color of the right half, using only its English name.'),dict(type='image_url',image_url=dict(url=image))])],32)
            assert 'blue' in row['text'].lower()
            result['checks']['image_semantics'] = True
            prefix = user('alpha beta gamma delta\n'*13000+'\nReply with only OK.')
            cold = request(prefix, 16, cache_salt='stock-cache-'+a.output.stem)
            hot = request(prefix, 16, cache_salt='stock-cache-'+a.output.stem)
            assert 'OK' in hot['text'] and hot['usage']['prompt_tokens_details']['cached_tokens'] >= 64000
            result['checks']['cached_prefix_tokens'] = hot['usage']['prompt_tokens_details']['cached_tokens']
        rows = result['requests'][:24]
        result['mixed_summary'] = dict(responses=len(rows), output_tokens=sum(x['usage']['completion_tokens'] for x in rows),
             output_over_decode_time=sum(x['usage']['completion_tokens']-1 for x in rows)/sum(x['end']-x['first'] for x in rows),
             output_over_request_time=sum(x['usage']['completion_tokens'] for x in rows)/sum(x['wall_s'] for x in rows),
             median_request_tps=statistics.median(x['decode_tps'] for x in rows),
             ttft_median_s=statistics.median(x['ttft_s'] for x in rows))
        result.update(passed=True, finished=time.time())
    finally:
        save()
    print(json.dumps(dict(passed=True,label=a.label,summary=result['mixed_summary'],checks=result['checks'])),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', required=True)
    p.add_argument('--key-file', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--label', choices=['candidate','baseline-before','baseline-after'], required=True)
    run(p.parse_args())
