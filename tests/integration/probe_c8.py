"""Finite C8 reference: actual stream overlap and process-epoch telemetry."""
from pathlib import Path
import concurrent.futures as cf
import hashlib
import json
import sys
import threading
import time
import urllib.request
from trial_smoke import call, CODE
import native_capability_smoke as cap

PROSE = 'Write a detailed engineering handbook chapter about operating a four-node GPU inference cluster. Cover topology, memory accounting, numerical validation, request queues, cancellation, cache namespaces, tracing, benchmarking and recovery. Use varied prose and concrete examples.'


def main():
    base, key_file, output, label = sys.argv[1:5]
    key = Path(key_file).read_text().strip()
    out = Path(output)
    result = dict(passed=False, started=time.time(), label=label, slots=(8 if label == 'candidate' else 4), single=[], groups=[])
    http = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def save():
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    def run(prompt, count):
        v = call(base, key, prompt, count)
        assert v['usage']['completion_tokens'] == count
        v['sha256'] = hashlib.sha256(v['text'].encode()).hexdigest()
        return v

    try:
        run('Reply with only OK.', 1)
        for rep in range(2):
            for name, prompt, count in [('code', CODE, 2048), ('prose', PROSE, 1024)]:
                result['single'].append(dict(workload=name, rep=rep, **run(prompt, count)))
                save()
            for name, prompt in [('code', CODE), ('prose', PROSE)]:
                observations, errors = [], []
                done = threading.Event()

                def observe():
                    while not done.is_set():
                        try:
                            req = urllib.request.Request(base.removesuffix('/v1') + '/telemetry', headers={'Authorization': 'Bearer ' + key})
                            with http.open(req, timeout=2) as response:
                                d = json.load(response)
                            observations.append(dict(time=time.monotonic(), pending=d['pending']))
                        except Exception as error:
                            errors.append(repr(error))
                        done.wait(.2)

                monitor = threading.Thread(target=observe)
                monitor.start()
                try:
                    barrier = threading.Barrier(8)

                    def worker(k):
                        barrier.wait(timeout=15)
                        return run(f'Independent worker {k}:\n' + prompt, 512)

                    with cf.ThreadPoolExecutor(8) as pool:
                        rows = list(pool.map(worker, range(8)))
                finally:
                    done.set()
                    monitor.join(5)
                overlap = min(r['end'] for r in rows) - max(r['first'] for r in rows)
                span = max(r['end'] for r in rows) - min(r['first'] for r in rows)
                group = dict(workload=name, rep=rep, rows=rows, overlap_all_streams_s=overlap,
                             pending_peak=max((v['pending'] for v in observations), default=0),
                             telemetry=observations, telemetry_errors=errors,
                             aggregate_decode_tps=sum(r['usage']['completion_tokens'] - 1 for r in rows) / span)
                result['groups'].append(group)
                save()
                assert overlap > 0 and group['pending_peak'] == 8, 'C8 overlap not demonstrated'
        if label == 'candidate':
            result['capabilities'] = cap.smoke(base, key)
        result.update(passed=True, finished=time.time())
        save()
        print(json.dumps(dict(passed=True, label=label)), flush=True)
    except BaseException as error:
        result.update(error=repr(error), finished=time.time())
        save()
        raise


if __name__ == '__main__':
    main()
