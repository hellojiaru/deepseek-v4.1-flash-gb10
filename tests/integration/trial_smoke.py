"""Bounded real mixed-length completion, cancellation, cache and queue acceptance."""
from pathlib import Path
import concurrent.futures as cf,json,threading,time,urllib.request,uuid
import native_capability_smoke as cap
smoke=cap.smoke
CODE='Write a complete, well-commented Python implementation of an append-only event store with SQLite, optimistic concurrency, snapshots, idempotency keys, recovery after a crash, and unit tests. Include all code; do not abbreviate.'

def call(base,key,prompt,n,mode='normal',first_event=None,cancel_s=None):
    public='8083' in base
    b=dict(model=('local/deepseek-v41-' if public else cap.MODEL)+mode,messages=[dict(role='user',content=prompt)],
           max_tokens=n,temperature=0,seed=42,thinking={'type':'disabled'},stream=True,stream_options={'include_usage':True})
    request_id=str(uuid.uuid4());req=urllib.request.Request(base+'/chat/completions',json.dumps(b).encode(),
          {'Content-Type':'application/json','Authorization':'Bearer '+key,'X-Client-Request-Id':request_id})
    start=time.monotonic();first=last=None;text='';usage=None;done=False;cancelled=False
    http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with http.open(req,timeout=180) as f:
        for line in f:
            if not line.startswith(b'data: '):continue
            raw=line[6:].strip()
            if raw==b'[DONE]':done=True;break
            d=json.loads(raw);assert 'error' not in d,d.get('error')
            if d.get('usage'):usage=d['usage']
            for c in d.get('choices',[]):
                v=c.get('delta',{});part=(v.get('content') or '')+(v.get('reasoning_content') or '')
                if part:
                    now=time.monotonic();first=first or now;last=now;text+=part
                    if first_event:first_event.set()
            if cancel_s is not None and first is not None and time.monotonic()-first>=cancel_s:
                cancelled=True;break
    end=time.monotonic()
    r=dict(request_id=request_id,start=start,first=first,last=last,end=end,text=text,usage=usage,done=done,cancelled=cancelled)
    if cancelled:return r
    assert done and usage and first is not None,r
    r.update(ttft=first-start,done_tail=end-last,decode_tps=(usage['completion_tokens']-1)/(end-first))
    return r

def performance(base,key):
    rows=[call(base,key,CODE,384) for _ in range(3)]
    long=call(base,key,CODE,2048)
    assert long['usage']['completion_tokens']==2048
    return dict(short=rows,long=long)

def qualification(base,key):
    perf=performance(base,key)
    # Start the long stream first, then force short work to join a live stream.
    first=threading.Event()
    with cf.ThreadPoolExecutor(4) as pool:
        long=pool.submit(call,base,key,CODE,2048,'normal',first)
        assert first.wait(20),'long stream did not begin'
        shorts=[pool.submit(call,base,key,'Reply with only OK.',16) for _ in range(3)]
        short=[x.result(timeout=90) for x in shorts];long=long.result(timeout=150)
    assert all('OK' in r['text'] and r['end']<long['end'] and r['done_tail']<5 for r in short),'short Done retained behind long peer'
    first=threading.Event()
    with cf.ThreadPoolExecutor(2) as pool:
        peer=pool.submit(call,base,key,CODE,1024,'normal',first)
        assert first.wait(20)
        cancelled=pool.submit(call,base,key,CODE,2048,'normal',None,.3).result(timeout=60)
        peer=peer.result(timeout=90)
    assert cancelled['cancelled'] and peer['done'] and peer['usage']['completion_tokens']==1024
    caps=cap.smoke(base,key)
    # Four approximately 64K fixed-prefix sessions, then repeat after N/C/N.
    prompts=[f'Worker {k}.\n'+('alpha beta gamma delta\n'*13000)+'\nReply with only OK.' for k in range(4)]
    def group():
        with cf.ThreadPoolExecutor(4) as pool:return list(pool.map(lambda p:call(base,key,p,16),prompts))
    cold=group();hot=group()
    call(base,key,'Reply with only OK.',16,'creative');call(base,key,'Reply with only OK.',16,'normal')
    restored=group()
    for rows in (cold,hot,restored):assert all('OK' in r['text'] and r['usage']['prompt_tokens']>=64000 for r in rows)
    for rows in (hot,restored):assert all(r['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0)>=64000 for r in rows)
    return dict(passed=True,performance=perf,mixed=dict(long=long,short=short),cancel=dict(cancelled=cancelled,peer=peer),capabilities=caps,
                long_cache=dict(cold=cold,hot=hot,restored=restored),scope='bounded scheduling/function acceptance; no cold-cache bit equivalence or 4x1M endurance claim')

def public_queue(key,out,observe=None):
    base='http://127.0.0.1:8083/v1';events=[];stopped=threading.Event()
    def monitor():
        while not stopped.is_set():
            if observe:events.append(dict(time=time.time(),**observe()))
            stopped.wait(.5)
    m=threading.Thread(target=monitor);m.start()
    try:
        first=[threading.Event() for _ in range(4)]
        with cf.ThreadPoolExecutor(5) as pool:
            futures=[pool.submit(call,base,key,f'Worker {k}:\n'+CODE,2048,'normal',first[k]) for k in range(4)]
            assert all(e.wait(25) for e in first),'full four-way load did not start'
            queued=pool.submit(call,base,key,'Reply with only OK.',16)
            rows=[f.result(timeout=170) for f in futures];q=queued.result(timeout=170)
        assert all(r['usage']['completion_tokens']==2048 for r in rows),'queue fixture ended early'
        assert q['ttft']>30 and 'OK' in q['text'],'fifth request did not demonstrably wait past 30s'
    finally:
        stopped.set();m.join(20)
        (Path(out)/'queue-observations.json').write_text(json.dumps(events,indent=2))
    assert events and max(v['gate']['in_flight'] for v in events)<=4
    assert max(v['load']['current_in_use'] for v in events)<=4
    assert max(v['load']['waiting_in_queue'] for v in events)>=1
    return dict(passed=True,rows=rows,queued=q,observations=events,execution_limit=4)

if __name__=='__main__':
    import sys
    root=Path(sys.argv[1]);out=Path(sys.argv[2]);key=(root/'service/api-key').read_text().strip()
    try:
        result=performance('http://127.0.0.1:18893/v1',key) if len(sys.argv)>3 and sys.argv[3]=='baseline' else qualification('http://127.0.0.1:18893/v1',key)
        out.write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(json.dumps(dict(passed=True)))
    except BaseException as e:
        out.write_text(json.dumps(dict(passed=False,error=repr(e))));raise
