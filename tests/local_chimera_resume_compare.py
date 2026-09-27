"""Compare all saved tensors, including distributed Adam state, from local probes."""
import argparse
import json
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('uninterrupted')
    parser.add_argument('resumed')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    import torch
    from megatron.core import dist_checkpointing
    from slime.utils.eval_config import EvalDatasetConfig
    from slime_plugins.chimera_mixrl.core import write_json

    with tempfile.TemporaryDirectory(prefix='chimera-resume-') as temp:
        torch.distributed.init_process_group('gloo', rank=0, world_size=1,
                                            init_method='file://' + str(Path(temp) / 'rdzv'))
        # These are our locally generated trusted checkpoint probes, not external
        # untrusted pickles. Allow only the known Slime metadata class, not unrestricted loading.
        with torch.serialization.safe_globals([EvalDatasetConfig]):
            a = dist_checkpointing.load_plain_tensors(args.uninterrupted)
            b = dist_checkpointing.load_plain_tensors(args.resumed)
        def tensors(node, prefix=''):
            if isinstance(node, torch.Tensor):
                return {prefix: node}
            result = {}
            children = node.items() if isinstance(node, dict) else enumerate(node) if isinstance(node, (list, tuple)) else []
            for key, value in children:
                result.update(tensors(value, prefix + '/' + str(key)))
            return result
        a, b = tensors(a), tensors(b)
        if set(a) != set(b):
            raise RuntimeError('Resumed checkpoint tensor key set differs')
        differences = [name for name in a if a[name].dtype != b[name].dtype or
                       a[name].shape != b[name].shape or not torch.equal(a[name], b[name])]
        optimizer = [name for name in a if 'optimizer' in name]
        report = dict(tensors=len(a), optimizer_tensors=len(optimizer), differences=differences,
                      exact=not differences, scope=__doc__)
        write_json(Path(args.output), report)
        print(json.dumps(report), flush=True)
        torch.distributed.destroy_process_group()
    if differences or not optimizer:
        raise RuntimeError('Resume must preserve both model and optimizer tensors exactly')


if __name__ == '__main__':
    main()
