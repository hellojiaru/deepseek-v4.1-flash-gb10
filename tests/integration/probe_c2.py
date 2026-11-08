"""C2 reference using the existing fixed code/prose benchmark prompts."""
from pathlib import Path
import concurrent.futures as cf
import hashlib
import json
import sys
import threading
import time
from trial_smoke import call, CODE
from probe import PROSE


def main():
    base, key_file, output = sys.argv[1:4]
    key = Path(key_file).read_text().strip()
    output = Path(output)
    result = dict(passed=False, started=time.time(), slots_in_service=4, concurrent_requests=2, groups=[])
    try:
        for repeat in range(3):
            for name, prompt in [('code', CODE), ('prose', PROSE)]:
                barrier = threading.Barrier(2)
                def one(lane):
                    barrier.wait(timeout=10)
                    row = call(base,key,f'Independent worker {lane}:\n'+prompt,512)
                    assert row['usage']['completion_tokens']==512
                    row['sha256']=hashlib.sha256(row['text'].encode()).hexdigest()
                    return row
                with cf.ThreadPoolExecutor(2) as pool:
                    rows=list(pool.map(one,range(2)))
                result['groups'].append(dict(workload=name,repeat=repeat,rows=rows,
                    aggregate_decode_tps=sum(x['usage']['completion_tokens']-1 for x in rows)/(max(x['end'] for x in rows)-min(x['first'] for x in rows)),
                    end_to_end_tps=sum(x['usage']['completion_tokens'] for x in rows)/(max(x['end'] for x in rows)-min(x['start'] for x in rows))))
                output.write_text(json.dumps(result,indent=2),encoding='utf-8')
        result.update(passed=True,finished=time.time())
    finally:
        output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=True,groups=len(result['groups']))),flush=True)


if __name__=='__main__':
    main()
