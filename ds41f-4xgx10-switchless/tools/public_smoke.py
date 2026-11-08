#!/usr/bin/env python3
"""Public synthetic multimodal smoke fixture. Dry-run unless --execute is set.

Uses no captured conversations, images, credentials or deployment configuration.
HTTP execution is opt-in and intended for a separately reserved test server.
"""
import argparse
import base64
import hashlib
import json
import pathlib
import struct
import time
import urllib.error
import urllib.request
import zlib


def png_rgb(rgb, size=256):
    def chunk(name, data):
        return (struct.pack('>I', len(data)) + name + data
                + struct.pack('>I', zlib.crc32(name + data) & 0xffffffff))
    rows = (b'\0' + bytes(rgb) * size) * size
    return (b'\x89PNG\r\n\x1a\n'
            + chunk(b'IHDR', struct.pack('>IIBBBBB', size, size, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(rows, 9)) + chunk(b'IEND', b''))


def fixtures(model, blocks):
    image = png_rgb((0, 255, 0))
    context = ''.join(f'Archive entry {n:05d}: the numbered tile is part of a synthetic interface test.\n'
                      for n in range(blocks))
    tools = [{'type': 'function', 'function': {
        'name': 'inspect_image', 'description': 'Read the synthetic screenshot.',
        'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}}]
    prefix = [{'role': 'user', 'content': context + '\nA screenshot will follow in the next turn. Reply READY for now.'}]
    image_messages = prefix + [
        {'role': 'assistant', 'content': 'READY'},
        {'role': 'user', 'content': 'Now inspect the screenshot with inspect_image.'},
        {'role': 'assistant', 'content': None, 'tool_calls': [{
            'id': 'synthetic_image_1', 'type': 'function',
            'function': {'name': 'inspect_image', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'synthetic_image_1', 'content': [
            {'type': 'text', 'text': 'Synthetic screenshot:'},
            {'type': 'image_url', 'image_url': {
                'url': 'data:image/png;base64,' + base64.b64encode(image).decode('ascii')}}]},
        {'role': 'user', 'content': 'What is the dominant color of this screenshot? Reply with one color word.'}]
    common = {'model': model, 'temperature': 1, 'top_p': 1,
              'max_tokens': 256, 'stream': False,
              'chat_template_kwargs': {'thinking': False}}
    text = dict(common, messages=prefix, tools=tools)
    visual = dict(common, messages=image_messages, tools=tools)
    return [('text', text), ('image_tool', visual), ('image_repeat', visual)], image


def encode(value):
    return json.dumps(value, separators=(',', ':'), ensure_ascii=False).encode('utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='test-model')
    parser.add_argument('--text-blocks', type=int, default=32)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--base-url', help='Explicit test server base ending in /v1')
    parser.add_argument('--api-key-env', default='OPENAI_API_KEY')
    parser.add_argument('--output', type=pathlib.Path)
    args = parser.parse_args()
    if not 1 <= args.text_blocks <= 8192:
        parser.error('--text-blocks must be in 1..8192 (not a token-count guarantee)')
    if args.execute and not args.base_url:
        parser.error('--execute requires an explicit --base-url for your reserved test service')
    cases, image = fixtures(args.model, args.text_blocks)
    report = {'schema': 1, 'executed': args.execute,
              'image_sha256': hashlib.sha256(image).hexdigest(), 'cases': []}
    failed = False
    for name, body in cases:
        payload = encode(body)
        row = {'name': name, 'request_sha256': hashlib.sha256(payload).hexdigest(),
               'request_bytes': len(payload)}
        if args.execute:
            import os
            headers = {'Content-Type': 'application/json'}
            key = os.environ.get(args.api_key_env)
            if key:
                headers['Authorization'] = 'Bearer ' + key
            request = urllib.request.Request(args.base_url.rstrip('/') + '/chat/completions',
                                             data=payload, headers=headers)
            started = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    result = json.load(response)
                row['http_completion_seconds'] = time.monotonic() - started
                choice = result['choices'][0]
                answer = choice.get('message', {}).get('content') or ''
                row['finish_reason'] = choice.get('finish_reason')
                row['usage'] = result.get('usage', {})
                row['answer_sha256'] = hashlib.sha256(answer.encode('utf-8')).hexdigest()
                row['completed'] = choice.get('finish_reason') == 'stop'
                failed |= not row['completed']
                if name == 'text':
                    row['ready_answer'] = answer.strip().upper() == 'READY'
                    failed |= not row['ready_answer']
                if name.startswith('image'):
                    row['green_answer'] = answer.strip().lower().rstrip('.!') == 'green'
                    failed |= not row['green_answer']
            except urllib.error.HTTPError as exc:
                row['http_status'] = exc.code
                failed = True
            except (urllib.error.URLError, TimeoutError, KeyError, IndexError, ValueError, TypeError):
                row['error'] = 'request_or_response_failed'
                failed = True
        report['cases'].append(row)
    report['passed'] = None if not args.execute else not failed
    data = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(data, encoding='utf-8')
    print(data, end='')
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
