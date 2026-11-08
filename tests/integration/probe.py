"""Matched finite Normal-only C1/C4 measurements; saves every request and output hash."""
from pathlib import Path
import concurrent.futures as cf,hashlib,json,statistics,sys,time
from trial_smoke import call,CODE
import native_capability_smoke as cap
PROSE='Write a detailed engineering handbook chapter about operating a four-node GPU inference cluster. Cover topology, memory accounting, numerical validation, request queues, cancellation, cache namespaces, tracing, benchmarking and recovery. Use varied prose and concrete examples.'
def main():
 base=sys.argv[1];key=Path(sys.argv[2]).read_text().strip();out=Path(sys.argv[3]);mode=sys.argv[4]
 result=dict(passed=False,started=time.time(),label=mode,single=[],groups=[])
 def save():out.write_text(json.dumps(result,ensure_ascii=False,indent=2))
 def run(p,n):
  v=call(base,key,p,n);v['sha256']=hashlib.sha256(v['text'].encode()).hexdigest();assert v['usage']['completion_tokens']==n;return v
 try:
  run('Reply with only OK.',1)
  for rep in range(2):
   for name,prompt,n in [('code',CODE,2048),('prose',PROSE,1024)]:
    result['single'].append(dict(workload=name,rep=rep,**run(prompt,n)));save()
   for name,prompt in [('code',CODE),('prose',PROSE)]:
    with cf.ThreadPoolExecutor(4) as pool:rows=list(pool.map(lambda k:run(f'Independent worker {k}:\n'+prompt,512),range(4)))
    decspan=max(r['end'] for r in rows)-min(r['first'] for r in rows)
    result['groups'].append(dict(workload=name,rep=rep,rows=rows,aggregate_decode_tps=sum(r['usage']['completion_tokens']-1 for r in rows)/decspan));save()
  if mode=='candidate':
   result['scheduling_and_capabilities']=__import__('trial_smoke').qualification(base,key)
  result.update(passed=True,finished=time.time());save();print(json.dumps(dict(passed=True,label=mode)),flush=True)
 except BaseException as e:result.update(error=repr(e),finished=time.time());save();raise
if __name__=='__main__':main()
