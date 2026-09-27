"""Canonical production/tiny geometry; tiny is validation-only, never inferred."""


def expected_geometry(size='full'):
    common = dict(vocab_size=50176, num_key_value_heads=2, first_k_dense_replace=2,
                  last_k_dense_replace=0, n_shared_experts=0, shared_expert_intermediate_size=0,
                  rms_norm_eps=1e-5, scoring_func='sigmoid', routed_scaling_factor=2.5,
                  qk_layernorm=True, tie_word_embeddings=False, load_with_bias=True)
    if size == 'full':
        common.update(hidden_size=2048, intermediate_size=8192, moe_intermediate_size=2048,
                      num_hidden_layers=25, num_attention_heads=16, head_dim=256,
                      n_routed_experts=32, num_experts_per_tok=4)
    elif size == 'tiny':
        common.update(hidden_size=512, intermediate_size=2048, moe_intermediate_size=256,
                      num_hidden_layers=8, num_attention_heads=8, head_dim=64,
                      n_routed_experts=8, num_experts_per_tok=2)
    else:
        raise ValueError('CHIMERA_MODEL_SIZE must be full or tiny')
    return common


def validate_geometry(config, size='full'):
    mismatches = {k: {'expected': v, 'actual': config.get(k)}
                  for k, v in expected_geometry(size).items() if config.get(k) != v}
    if mismatches:
        raise ValueError(f'Checkpoint does not match canonical Chimera {size}: {mismatches}')


def expected_mcore_geometry(size='full'):
    hf = expected_geometry(size)
    mapping = {'num_layers': 'num_hidden_layers', 'hidden_size': 'hidden_size',
               'ffn_hidden_size': 'intermediate_size', 'num_attention_heads': 'num_attention_heads',
               'num_query_groups': 'num_key_value_heads', 'kv_channels': 'head_dim',
               'num_moe_experts': 'n_routed_experts', 'moe_router_topk': 'num_experts_per_tok',
               'moe_ffn_hidden_size': 'moe_intermediate_size', 'vocab_size': 'vocab_size',
               'layernorm_epsilon': 'rms_norm_eps'}
    return {key: hf[field] for key, field in mapping.items()}


def validate_mcore_geometry(model, size='full'):
    for key, value in expected_mcore_geometry(size).items():
        if model.get(key) != value:
            raise ValueError(f'Imported checkpoint architecture mismatch: {key}')
