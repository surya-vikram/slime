"""Bridge export metadata for synchronous Chimera saves, from actual source + run.

Never invent metadata from a repository template or modify the imported checkpoint.
The native Slime checkpoint remains authoritative for optimizer/RNG restoration.
"""

import copy
import dataclasses
import inspect
from contextlib import contextmanager
from pathlib import Path

from slime_plugins.models.chimera_geometry import validate_mcore_geometry
from .core import digest, write_json


def initialize_lazy_optimizer_state(optimizer):
    """Make a pre-first-update checkpoint resumable without taking a dummy step.

    Precision-aware Adam creates master parameters/moments lazily. A fully
    masked first batch must still save the sampler and initial optimizer state.
    Megatron's initializer only touches empty parameter states and is idempotent.
    """
    if optimizer is None:
        return
    children = getattr(optimizer, 'chained_optimizers', None)
    if children is not None:
        for child in children:
            initialize_lazy_optimizer_state(child)
        return
    initialize = getattr(optimizer, 'init_state_fn', None)
    inner = getattr(optimizer, 'optimizer', None)
    if initialize is not None and inner is not None:
        native = getattr(inner, 'initialize_state', None)
        if (getattr(optimizer.config, 'use_precision_aware_optimizer', False)
                and native is not None
                and 'store_param_remainders' in inspect.signature(native).parameters):
            # Newer TE requires this argument; the pinned image's MCore callback
            # still uses the old one-argument API. Match TE's own step() logic,
            # but do not take a step or reset existing moments/master weights.
            import torch
            for group in inner.param_groups:
                for param in group['params']:
                    if not inner.state[param]:
                        native(param, store_param_remainders=(
                            inner.store_param_remainders and param.dtype == torch.bfloat16))
            return
        initialize(inner, optimizer.config)


@contextmanager
def portable_save_args(args):
    """Avoid a Slime-only pickle class in checkpoints consumed by Bridge.

    The save is synchronous. Restore the exact live eval objects afterwards;
    native resume reconstructs them from its immutable launcher configuration.
    """
    if args.async_save:
        raise ValueError('Portable Chimera checkpoint arguments require synchronous saving')
    original = getattr(args, 'eval_datasets', None)
    if original is None:
        yield
        return
    try:
        args.eval_datasets = [dataclasses.asdict(item) if dataclasses.is_dataclass(item) else item
                              for item in original]
        yield
    finally:
        args.eval_datasets = original


def write_metadata(config, save_dir, iteration):
    import yaml

    if config['model_profile'] != 'chimera':
        return
    if config['execution_mode'] != 'sync' or type(iteration) is not int or iteration < 0:
        raise ValueError('Chimera export metadata requires a synchronous completed checkpoint')
    context = config['checkpoint_context']
    provenance = context['mcore_context_provenance']
    source = Path(provenance['source'])
    document = yaml.safe_load(source.read_text())
    if not isinstance(document, dict) or not isinstance(document.get('model'), dict):
        raise ValueError('Source checkpoint export metadata is missing or malformed')
    document = copy.deepcopy(document)
    model = document['model']
    validate_mcore_geometry(model, config['chimera_model_size'])
    yarn = context['yarn']
    model.update(chimera_context_phase=context['phase'], position_embedding_type='yarn',
                 seq_length=context['model_max_context'], rotary_base=yarn['rotary_base'],
                 rotary_scaling_factor=yarn['scaling_factor'],
                 yarn_rotary_scaling_factor=yarn['scaling_factor'],
                 yarn_original_max_position_embeddings=yarn['original_max_position_embeddings'],
                 yarn_beta_fast=yarn['beta_fast'], yarn_beta_slow=yarn['beta_slow'],
                 yarn_mscale=yarn['mscale'], yarn_mscale_all_dim=yarn['mscale_all_dim'],
                 yarn_correction_range_round_to_int=yarn['correction_range_round_to_int'],
                 moe_router_load_balancing_type='none', moe_router_bias_update_rate=0.,
                 moe_aux_loss_coeff=0., moe_z_loss_coeff=0.,
                 tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
                 expert_model_parallel_size=config.get('expert_model_parallel_size', 1),
                 expert_tensor_parallel_size=1,
                 context_parallel_size=1)
    root = Path(save_dir)
    iteration_dir = root / f'iter_{iteration:07d}'
    if not iteration_dir.is_dir():
        raise ValueError('Native checkpoint iteration directory missing after save')
    encoded = yaml.safe_dump(document, sort_keys=False)
    for directory in (iteration_dir, root):
        destination = directory / 'run_config.yaml'
        temporary = directory / 'run_config.yaml.tmp'
        temporary.write_text(encoded)
        temporary.replace(destination)
    write_json(iteration_dir / 'chimera_mixrl_identity.json', {
        'config_hash': digest(config), 'iteration': iteration,
        'checkpoint_context': context, 'train_sequence_length': config['context'],
        'router_frozen': True, 'weight_decay': config['weight_decay'],
        'note': 'Native checkpoint restores optimizer/RNG; run_config.yaml supplies Bridge model export metadata.'})
