"""Live fixed-weight SGLang/HF response-logprob and captured-route diagnostic.

Random canonical tiny checkpoint: mechanics only. Forces recorded routes in HF
for the second comparison; never alters checkpoint weights or training routing.
"""

import argparse
import base64
import json
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', default='http://127.0.0.1:19030')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--transformers-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-logprob-error', type=float, default=.05)
    parser.add_argument('--mean-logprob-error', type=float, default=.01)
    args = parser.parse_args()
    import numpy as np
    import torch
    from slime_plugins.models.chimera import register_transformers
    from slime_plugins.chimera_mixrl.core import write_json
    register_transformers(args.transformers_root)
    from transformers.models.chimera import ChimeraForCausalLM
    model = ChimeraForCausalLM.from_pretrained(args.checkpoint, dtype=torch.bfloat16,
                                             attn_implementation='eager').cuda().eval()
    assert model.config.num_hidden_layers == 8 and model.config.num_experts_per_tok == 2
    results = []
    for case in range(6):
        prompt = list(range(12 + case, 12 + case + (4, 8, 16)[case % 3]))
        payload = dict(input_ids=prompt, sampling_params=dict(max_new_tokens=8, temperature=0,
                       ignore_eos=True), return_logprob=True, return_routed_experts=True)
        request = urllib.request.Request(args.endpoint.rstrip('/') + '/generate',
                  data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)
        tokens = prompt + result['output_ids']
        paths = np.frombuffer(base64.b64decode(result['meta_info']['routed_experts']), dtype=np.int32)
        paths = torch.from_numpy(paths.copy().reshape(len(tokens) - 1, 8, 2)).cuda().long()
        assert not paths[:, :2].any()
        assert ((paths[:, 2:] >= 0) & (paths[:, 2:] < 8)).all()
        assert (paths[:, 2:, 0] != paths[:, 2:, 1]).all()
        ids = torch.tensor([tokens[:-1]], device='cuda')
        targets = torch.tensor(tokens[len(prompt):], device='cuda')
        behavior = torch.tensor([x[0] for x in result['meta_info']['output_token_logprobs']], device='cuda')
        selected_rows = slice(len(prompt) - 1, len(tokens) - 1)
        def logprobs():
            with torch.no_grad():
                logits = model(input_ids=ids, use_cache=False).logits[0, selected_rows].float()
                return logits.log_softmax(-1).gather(1, targets[:, None]).squeeze(1)
        natural = logprobs()
        disagreements = []
        originals = []
        for layer in range(2, 8):
            mlp = model.model.layers[layer].mlp
            original = mlp.route_tokens_to_experts
            originals.append((mlp, original))
            def replay(logits, layer=layer, original=original):
                fresh, _ = original(logits)
                chosen = paths[:, layer]
                disagreements.append((fresh.sort(-1).values != chosen.sort(-1).values).any(-1).float().mean().item())
                scores = torch.sigmoid(logits.float()).gather(1, chosen)
                return chosen, scores / (scores.sum(-1, keepdim=True) + 1e-20) * 2.5
            mlp.route_tokens_to_experts = replay
        try:
            forced = logprobs()
        finally:
            for mlp, original in originals:
                mlp.route_tokens_to_experts = original
        difference = (forced - behavior).abs()
        report = dict(case=case, prompt_tokens=len(prompt), response_tokens=len(targets),
                      captured_rows=len(tokens) - 1,
                      natural_max_logprob_error=(natural - behavior).abs().max().item(),
                      replay_max_logprob_error=difference.max().item(),
                      replay_mean_logprob_error=difference.mean().item(),
                      route_set_disagreement_by_moe_layer=disagreements)
        results.append(report)
        print(json.dumps(report), flush=True)
    passed = all(r['replay_max_logprob_error'] <= args.max_logprob_error and
                 r['replay_mean_logprob_error'] <= args.mean_logprob_error for r in results)
    write_json(Path(args.output), dict(scope=__doc__, passed=passed, cases=results,
               thresholds=dict(max_error=args.max_logprob_error, mean_error=args.mean_logprob_error)))
    if not passed:
        raise RuntimeError('Live logprob diagnostic exceeded predeclared thresholds; inspect report')


if __name__ == '__main__':
    main()
