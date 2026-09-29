"""Opt-in local hosted-model/scorer smoke, one frozen rl_val prompt per route.

This checks real generation -> shared verifier wiring, not learning or judge
qualification. No main_test data, optimizer updates, or remote GPU access.
"""

import argparse
import concurrent.futures
import json
import math
import time
from pathlib import Path

from slime_plugins.chimera_mixrl.core import digest, load_split, write_json
from slime_plugins.chimera_mixrl import tasks as task_file
from slime_plugins.chimera_mixrl.routes import validate_route
from slime_plugins.chimera_mixrl.runtime import request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--model-url', default='http://127.0.0.1:8010/v1')
    parser.add_argument('--model', default='eval-long2b')
    parser.add_argument('--scorer-url', default='http://127.0.0.1:8020')
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--concurrency', type=int, default=2)
    parser.add_argument('--tasks', default='', help='Optional comma-separated route subset for targeted rechecks')
    parser.add_argument('--tasks-file', default=str(task_file.DEFAULT_PATH))
    args = parser.parse_args()
    rows, manifest = load_split(args.data_dir, 'rl_val')
    protocol = request(args.scorer_url + '/health', timeout=10)['protocol_id']
    catalog = task_file.routes(task_file.load(args.tasks_file)['tasks'])
    routes = args.tasks.split(',') if args.tasks else list(catalog)
    if set(routes) - catalog.keys():
        raise ValueError('Unknown training route')
    selected = [min((r for r in rows if r['task'] == route),
                    key=lambda r: len(json.dumps(r['messages']))) for route in routes]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    identity = digest({'manifest': manifest, 'protocol': protocol, 'args': vars(args)})

    def run(row):
        path = output / (row['task'] + '.json')
        start = time.monotonic()
        record = {'row_id': row['id'], 'task': row['task'], 'identity': identity}
        try:
            validate_route(row, catalog)
            if path.exists():
                record = json.loads(path.read_text())
                if record['identity'] != identity:
                    raise ValueError('Smoke identity changed; choose new output directory')
            if 'response' not in record:
                result = request(args.model_url + '/chat/completions', {
                    'model': args.model, 'messages': row['messages'], 'max_tokens': args.max_tokens,
                    'temperature': .7, 'top_p': 1., 'seed': 42,
                    'chat_template_kwargs': {'enable_thinking': False}}, timeout=180)
                choice = result['choices'][0]
                record['response'] = {'text': choice['message']['content'],
                                      'finish_reason': choice['finish_reason']}
                record['usage'] = result.get('usage')
                write_json(path, record)
            result = request(args.scorer_url + '/score', {
                'request_id': f'{identity}:{row["id"]}', 'protocol_id': protocol,
                'split': 'rl_val', 'row_id': row['id'], 'row_hash': digest(row),
                'response': record['response']}, timeout=600)
            grade = result['grade']
            if grade.get('status') != 'valid' or not math.isfinite(grade['score']) or not 0 <= grade['score'] <= 1:
                raise ValueError('Invalid normalized grade')
            record['grade'] = grade
            record['ok'] = True
            record.pop('error', None)
        except Exception as exc:
            record.update(ok=False, error=f'{type(exc).__name__}: {exc}')
        record['seconds'] = time.monotonic() - start
        write_json(path, record)
        print(json.dumps({k: record[k] for k in ('task', 'ok', 'seconds')}), flush=True)
        return record

    with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
        results = list(pool.map(run, selected))
    summary = {'scope': 'live plumbing only; not capability/learning/judge qualification',
               'passed': sum(r['ok'] for r in results), 'total': len(results),
               'capped': sum(r.get('response', {}).get('finish_reason') == 'length' for r in results),
               'identity': identity, 'protocol': protocol}
    write_json(output / 'summary.json', summary)
    print(json.dumps(summary), flush=True)
    if summary['passed'] != summary['total']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
