"""Small capability checks for the user-authorized C4 trial, never a benchmark matrix."""
from pathlib import Path
import base64, concurrent.futures as cf, hashlib, json, struct, time, urllib.request, zlib

MODEL='deepseek-v4.1-flash-dsgb10-official-1017-'


def smoke(base, key, public=False):
    http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    rows=[]
    def call(mode,messages,n=64,**kw):
        body=dict(model=('local/deepseek-v41-' if public else MODEL)+mode,messages=messages,
                  temperature=0,seed=42,thinking={'type':'disabled'},max_tokens=n,stream=False,**kw)
        req=urllib.request.Request(base+'/chat/completions',json.dumps(body).encode(),
                  {'Authorization':'Bearer '+key,'Content-Type':'application/json'})
        start=time.monotonic()
        with http.open(req,timeout=75) as f:d=json.load(f)
        assert 'error' not in d and d['choices'], 'generation error'
        c=d['choices'][0]
        r=dict(mode=mode,seconds=time.monotonic()-start,usage=d.get('usage'),
               message=c['message'],finish_reason=c['finish_reason'])
        assert r['usage'] and r['usage']['completion_tokens']>0
        rows.append(r)
        return r
    def user(text):return [dict(role='user',content=text)]
    # N -> C -> N really exercises request-boundary switching.
    for mode in ('normal','creative','normal'):
        r=call(mode,user('Reply with only OK.'),16)
        assert 'OK' in (r['message'].get('content') or ''),mode
    # Sufficient output for a real overlapping four-way native/public burst.
    def worker(i):return call('normal',user(f'Worker {i}: Write a complete Python append-only SQLite event store. Include all code and comments.'),128)
    with cf.ThreadPoolExecutor(4) as pool:burst=list(pool.map(worker,range(4)))
    assert all(r['usage']['completion_tokens']==128 and r['finish_reason']=='length' for r in burst)
    if not public:
        tools=[dict(type='function',function=dict(name='get_temperature',description='Read temperature in a city',
               parameters=dict(type='object',properties=dict(city=dict(type='string')),required=['city'])))]
        def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
        pixels=b''.join(b'\0'+b'\xff\0\0'*32+b'\0\0\xff'*32 for _ in range(32))
        png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',64,32,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(pixels))+chunk(b'IEND',b'')
        url='data:image/png;base64,'+base64.b64encode(png).decode()
        for mode in ('normal','creative'):
            messages=user('Call get_temperature for Beijing.')
            r=call(mode,messages,128,tools=tools,tool_choice=dict(type='function',function=dict(name='get_temperature')))
            tc=r['message'].get('tool_calls',[])
            assert len(tc)==1 and tc[0]['function']['name']=='get_temperature'
            assert json.loads(tc[0]['function']['arguments'])['city'].lower()=='beijing'
            messages += [r['message'],dict(role='tool',tool_call_id=tc[0]['id'],content='{"temperature":22}'),
                         dict(role='user',content='Reply with only the temperature number returned by the tool.')]
            r=call(mode,messages)
            assert '22' in (r['message'].get('content') or '')
            r=call(mode,[dict(role='user',content=[dict(type='text',text='What color is the RIGHT half? Reply with only the English color name.'),
                         dict(type='image_url',image_url=dict(url=url))])])
            assert 'blue' in (r['message'].get('content') or '').lower()
        # Repeated prefix and cross-mode return; no claim of cold/cache bit equality.
        messages=user(('alpha beta gamma delta\n'*600)+'\nReply with only OK.')
        for mode in ('normal','normal','creative','normal'):
            r=call(mode,messages,16)
            assert 'OK' in (r['message'].get('content') or '')
        cached=[r['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0) for r in rows[-4:]]
        assert cached[1]>1000 and cached[3]>1000,cached
    return dict(passed=True,time=time.time(),public=public,rows=rows,
                tests='mode switch, single, C4'+(', tools, image, prefix reuse' if not public else ''),
                full_long_agent_tested=False,full_4x1m_tested=False)


if __name__=='__main__':
    import sys
    root=Path(sys.argv[1]);target=Path(sys.argv[2])
    try:
        result=smoke('http://127.0.0.1:18893/v1',(root/'service/api-key').read_text().strip())
    except BaseException as e:
        target.write_text(json.dumps(dict(passed=False,error=repr(e))))
        raise
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(dict(passed=True,requests=len(result['rows']))))
