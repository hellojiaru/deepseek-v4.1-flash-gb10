"""Reproducible public-source text cold-prefill check through the paired HTTP API."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
import uuid

FILES = ['engine/model/model.cu', 'engine/model/prefill.cu', 'engine/kernels/moe.cu',
         'engine/kernels/gemv_fp8.cu', 'engine/attn/sparse_attn.cu', 'engine/attn/indexer.cu',
         'frontend/src/server.rs', 'frontend/src/session.rs']


def prepare(a):
    sys.path.insert(0, str(a.source_root / 'ref/encoding-1017'))
    import encoding
    a.inputs.mkdir(exist_ok=False)
    files = {name: hashlib.sha256((a.source_root / name).read_bytes()).hexdigest() for name in FILES}
    corpus = '\n'.join('FILE ' + name + '\n' + (a.source_root / name).read_text() for name in FILES)
    corpus = corpus.replace('｜', '|').replace('<|', '< |').replace('<image>', '< image>')
    corpus *= 4
    manifest = []
    for target in (65536, 262144):
        chars, pad = min(len(corpus), target * 3), 0
        for attempt in range(12):
            text = 'The following is reference source code, not instructions.\n' + corpus[:chars]
            text += '\n' + ' z' * pad + '\nReply with only PREFILL_OK.\n'
            messages = [dict(role='user', content=text)]
            encoded = encoding.encode_messages(messages, thinking_mode='chat', reasoning_effort=75)
            encoded_file = a.inputs / f'{target}.encoded.txt'
            encoded_file.write_text(encoded)
            proc = subprocess.run([str(a.frontend), 'tokenize', str(encoded_file), '--tokenizer', str(a.tokenizer)],
                                  capture_output=True, text=True, timeout=90, check=True)
            n = len(json.loads(proc.stdout))
            if abs(n - target) <= 32:
                break
            if n > target:
                excess = n - target
                if pad >= excess:
                    pad -= excess
                else:
                    chars = int(chars * target / max(1, n - pad) * .99)
                    pad = 0
            else:
                pad += target - n
        else:
            raise RuntimeError('token count did not converge')
        body = dict(model='deepseek-v4.1-flash-dsgb10-official-1017-normal', messages=messages,
                    temperature=0, seed=42, max_tokens=16, stream=True,
                    stream_options=dict(include_usage=True), cache_salt='source-prefill-' + str(target))
        data = json.dumps(body, ensure_ascii=False).encode()
        (a.inputs / f'{target}.request.json').write_bytes(data)
        manifest.append(dict(target=target, input_tokens=n, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    (a.inputs / 'MANIFEST.json').write_text(json.dumps(dict(files=files, requests=manifest), indent=2))
    print(json.dumps(dict(prepared=True, targets=[x['target'] for x in manifest])), flush=True)


def run(a):
    manifest = json.loads((a.inputs / 'MANIFEST.json').read_text())
    key = a.key_file.read_text().strip()
    http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    result = dict(passed=False, started=time.time(), rows=[], fixture=manifest,
                  metric='prompt_tokens / client TTFT; includes frontend, tokenization and native preparation')
    def save():
        a.output.write_text(json.dumps(result, indent=2))
    try:
        for item in manifest['requests']:
            data = (a.inputs / f'{item["target"]}.request.json').read_bytes()
            assert hashlib.sha256(data).hexdigest() == item['sha256']
            rid = str(uuid.uuid4())
            row = dict(target=item['target'], request_id=rid, submitted=False)
            result['rows'].append(row)
            save()
            req = urllib.request.Request(a.base + '/chat/completions', data,
                                         {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key,
                                          'X-Client-Request-Id': rid})
            start = time.monotonic()
            first, text, usage, done = None, '', None, False
            row['submitted'] = True
            save()
            with http.open(req, timeout=180) as response:
                for line in response:
                    if not line.startswith(b'data: '):
                        continue
                    raw = line[6:].strip()
                    if raw == b'[DONE]':
                        done = True
                        break
                    value = json.loads(raw)
                    assert 'error' not in value
                    if value.get('usage'):
                        usage = value['usage']
                    for choice in value.get('choices', []):
                        part = choice.get('delta', {}).get('content') or ''
                        if part:
                            first = first or time.monotonic()
                            text += part
            assert done and first and usage and usage['prompt_tokens'] == item['input_tokens']
            assert usage['prompt_tokens_details']['cached_tokens'] == 0
            assert 'PREFILL_OK' in text
            row.update(ttft_s=first-start, wall_s=time.monotonic()-start, usage=usage,
                       effective_cold_prefill_tps=usage['prompt_tokens']/(first-start),
                       output_sha256=hashlib.sha256(text.encode()).hexdigest(), done=True)
            save()
        result.update(passed=True, finished=time.time())
    except BaseException as error:
        result.update(error=repr(error), finished=time.time())
        raise
    finally:
        save()
    print(json.dumps(dict(passed=True, rows=result['rows'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest='action', required=True)
    prep = subs.add_parser('prepare')
    prep.add_argument('--source-root', type=Path, required=True)
    prep.add_argument('--frontend', type=Path, required=True)
    prep.add_argument('--tokenizer', type=Path, required=True)
    prep.add_argument('--inputs', type=Path, required=True)
    execute = subs.add_parser('run')
    execute.add_argument('--inputs', type=Path, required=True)
    execute.add_argument('--base', required=True)
    execute.add_argument('--key-file', type=Path, required=True)
    execute.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    (prepare if a.action == 'prepare' else run)(a)
