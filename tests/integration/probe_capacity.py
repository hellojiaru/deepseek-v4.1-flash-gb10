"""C8 near-full-window capacity check; synthetic input, not a quality benchmark.

prepare uses the supplied checkpoint tokenizer and reference chat encoder, without
GPU use. run submits eight distinct cache namespaces and keeps partial evidence.
"""
from pathlib import Path
import argparse
import concurrent.futures as cf
import hashlib
import json
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

TARGET_INPUT = 1048320
OUTPUT = 256
MODEL = 'deepseek-v4.1-flash-dsgb10-official-1017-normal'


def prepare(a):
    sys.path.insert(0, str(a.encoding_source))
    import encoding
    a.inputs.mkdir(parents=True, exist_ok=False)
    records = []
    for lane in range(8):
        markers = [hashlib.sha256(f'capacity-{lane}-{k}-20261108'.encode()).hexdigest()[:12] for k in range(3)]
        lines, pad = (TARGET_INPUT - 240) // 5, 0
        for attempt in range(5):
            third = lines // 3
            text = f'Capacity lane {lane}. Keep the three keys from this document.\nKEY_A={markers[0]}\n'
            text += 'alpha beta gamma delta\n' * third + f'KEY_B={markers[1]}\n'
            text += 'alpha beta gamma delta\n' * third + f'KEY_C={markers[2]}\n'
            text += 'alpha beta gamma delta\n' * (lines - 2 * third)
            text += ' z' * pad
            text += '\nFirst print the three KEY values as JSON, then write a complete, well-commented Python SQLite append-only event store with optimistic concurrency, snapshots, idempotency and tests. Include full code and continue until the output limit.\n'
            messages = [dict(role='user', content=text)]
            prompt = encoding.encode_messages(messages, thinking_mode='chat', reasoning_effort=75)
            prompt_file = a.inputs / f'{lane}.encoded.txt'
            prompt_file.write_text(prompt, encoding='utf-8')
            proc = subprocess.run([str(a.frontend), 'tokenize', str(prompt_file), '--tokenizer', str(a.tokenizer)],
                                  capture_output=True, text=True, timeout=90, check=True)
            ids = json.loads(proc.stdout)
            count = len(ids)
            del ids
            if count == TARGET_INPUT:
                break
            difference = TARGET_INPUT - count
            if difference >= 0:
                pad += difference
            else:
                remove = min(pad, -difference)
                pad -= remove
                if difference + remove < 0:
                    lines -= (-difference - remove + 4) // 5
        else:
            raise RuntimeError(f'token count did not converge: {lane}, {count}')
        body = dict(model=MODEL, messages=messages, max_tokens=OUTPUT, temperature=0, seed=42,
                    stream=True, stream_options=dict(include_usage=True), cache_salt=f'capacity-20261108-lane-{lane}',
                    ignore_eos=True)
        data = json.dumps(body, ensure_ascii=False).encode()
        (a.inputs / f'{lane}.request.json').write_bytes(data)
        records.append(dict(lane=lane, input_tokens=count, max_output=OUTPUT, body_bytes=len(data),
                            sha256=hashlib.sha256(data).hexdigest(), markers=markers))
        (a.inputs / 'MANIFEST.json').write_text(json.dumps(records, indent=2))
        print(json.dumps(dict(prepared_lane=lane, tokens=count, bytes=len(data))), flush=True)
    print(json.dumps(dict(prepared=True, total_input_tokens=sum(x['input_tokens'] for x in records))), flush=True)


def run(a):
    manifest = json.loads((a.inputs / 'MANIFEST.json').read_text())
    key = a.key_file.read_text().strip()
    result = dict(passed=False, started=time.time(), synthetic_capacity_pattern=True,
                  target_input_per_lane=TARGET_INPUT, max_output_per_lane=OUTPUT, requests=[], snapshots=[])
    lock = threading.Lock()
    finished = threading.Event()
    barrier = threading.Barrier(8)

    def save():
        with lock:
            temp = a.output.with_suffix('.tmp')
            temp.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            temp.replace(a.output)

    def observe():
        iteration = 0
        while not finished.is_set():
            try:
                snapshot = json.loads(a.metrics_file.read_text())
                snapshot['sample_time'] = time.time()
                with lock:
                    result['snapshots'].append(snapshot)
            except (OSError, json.JSONDecodeError):
                pass
            if iteration % 10 == 0:
                save()
            iteration += 1
            finished.wait(.5)

    def one(row):
        lane = row['lane']
        data = (a.inputs / f'{lane}.request.json').read_bytes()
        assert hashlib.sha256(data).hexdigest() == row['sha256']
        request_id = str(uuid.uuid4())
        req = urllib.request.Request(a.base + '/chat/completions', data,
                                     {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key,
                                      'X-Client-Request-Id': request_id})
        http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        barrier.wait(timeout=30)
        start = time.monotonic()
        current = dict(lane=lane, request_id=request_id, start=start, input_target=TARGET_INPUT,
                       first=None, last=None, usage=None, text='', done=False)
        try:
            with http.open(req, timeout=1650) as response:
                for line in response:
                    if not line.startswith(b'data: '):
                        continue
                    raw = line[6:].strip()
                    if raw == b'[DONE]':
                        current['done'] = True
                        break
                    value = json.loads(raw)
                    assert 'error' not in value, 'upstream returned an error'
                    if value.get('usage'):
                        current['usage'] = value['usage']
                    for choice in value.get('choices', []):
                        delta = choice.get('delta', {})
                        part = (delta.get('content') or '') + (delta.get('reasoning_content') or '')
                        if part:
                            now = time.monotonic()
                            current['first'] = current['first'] or now
                            current['last'] = now
                            current['text'] += part
            current['end'] = time.monotonic()
            assert current['done'] and current['usage'] and current['first'] is not None
            assert current['usage']['prompt_tokens'] == TARGET_INPUT
            assert current['usage']['prompt_tokens_details']['cached_tokens'] == 0, 'requires a cold instance or fresh cache namespace'
            assert current['usage']['completion_tokens'] == OUTPUT
            current['full_window_tokens'] = TARGET_INPUT + OUTPUT
            current['ttft_s'] = current['first'] - start
            current['decode_tps'] = (OUTPUT - 1) / (current['end'] - current['first'])
            current['marker_recall'] = [x in current['text'] for x in row['markers']]
            current['passed_capacity'] = True
        except BaseException as error:
            current.update(error=type(error).__name__, end=time.monotonic(), passed_capacity=False)
            raise
        finally:
            with lock:
                result['requests'].append(current)
            save()
        return current

    monitor = threading.Thread(target=observe)
    monitor.start()
    try:
        with cf.ThreadPoolExecutor(8) as pool:
            rows = list(pool.map(one, manifest))
        with lock:
            result['overlap_all_streams_s'] = min(x['end'] for x in rows) - max(x['first'] for x in rows)
            result['peak_active_context_tokens'] = max(x.get('active_context_tokens', 0) for x in result['snapshots'])
            result['peak_active_sessions'] = max(x.get('active_sessions', 0) for x in result['snapshots'])
        assert result['overlap_all_streams_s'] > 0
        assert result['peak_active_sessions'] == 8
        assert result['peak_active_context_tokens'] >= 8 * TARGET_INPUT
        with lock:
            result.update(passed=True, finished=time.time())
    except BaseException as error:
        with lock:
            result.update(error=repr(error), finished=time.time())
        raise
    finally:
        finished.set()
        monitor.join(10)
        save()
    print(json.dumps(dict(passed=True, requests=8, peak_active_context_tokens=result['peak_active_context_tokens'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--inputs', type=Path, required=True)
    prep.add_argument('--frontend', type=Path, required=True)
    prep.add_argument('--tokenizer', type=Path, required=True)
    prep.add_argument('--encoding-source', type=Path, required=True)
    execute = sub.add_parser('run')
    execute.add_argument('--inputs', type=Path, required=True)
    execute.add_argument('--base', required=True)
    execute.add_argument('--key-file', type=Path, required=True)
    execute.add_argument('--metrics-file', type=Path, required=True)
    execute.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.action == 'prepare' else run)(args)
