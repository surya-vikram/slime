"""Real tokenizer/data admission diagnostic, without pretending to launch training."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from slime_plugins.chimera_mixrl import runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--hf-checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--config', help='Resolved launcher manifest; audits exactly that route/cap configuration')
    parser.add_argument('--context', type=int, help='Diagnostic comparison budget; does not change launcher config')
    parser.add_argument('--response-multiplier', type=int, default=1)
    parser.add_argument('--headroom', type=int, default=0)
    args = parser.parse_args()
    c = dict(data_dir=args.data_dir, run_dir=args.output_dir,
             quotas={'mcqa': 2, 'gsm8k_train': 1, 'openqa': 1},
             caps={'mcqa': 2048, 'gsm8k_train': 4096, 'openqa': 2048},
             context=8192, seed=42, chat_template_kwargs={'enable_thinking': False})
    if args.config:
        c = json.loads(Path(args.config).read_text())
        c.update(data_dir=args.data_dir, run_dir=args.output_dir)
    if args.context:
        c['context'] = args.context
    if args.response_multiplier < 1 or not 0 <= args.headroom < c['context']:
        parser.error('Invalid response multiplier or headroom')
    c['caps'] = {k: v * args.response_multiplier for k, v in c['caps'].items()}
    c['context_headroom'] = args.headroom
    with patch.object(runtime, 'config', return_value=c):
        source = runtime.DataSource(SimpleNamespace(hf_checkpoint=args.hf_checkpoint))
    print(json.dumps({'scope': 'CPU tokenizer/data admission only; no model execution',
                      'train_by_route': {r: len(p) for r, p in source.sampler.pools.items()},
                      'val_by_route': {r: sum(x['task'] == r for x in source.val) for r in c['quotas']},
                      'excluded': len(source.excluded), 'identity': source.identity,
                      'manifest': str(Path(args.output_dir) / 'manifests/mixrl_admission.json')}, indent=2))


if __name__ == '__main__':
    main()
