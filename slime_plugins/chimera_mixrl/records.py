"""JSON completion persistence with compact, non-pickle storage of integer arrays
(expert paths and top-p candidate sets)."""

import base64
import json
import math
import zlib

from .core import write_json

CODEC = 'zlib-base64-int32le-v1'
TOP_P_FIELDS = ('rollout_top_p_token_ids', 'rollout_top_p_token_offsets')


def _pack(values):
    import numpy as np
    if hasattr(values, 'detach'):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype='<i4')
    return {'codec': CODEC, 'shape': list(array.shape),
            'data': base64.b64encode(zlib.compress(array.tobytes())).decode('ascii')}


def _unpack(packed, name):
    import numpy as np
    import torch
    shape = packed.get('shape')
    if packed.get('codec') != CODEC or not isinstance(shape, list):
        raise ValueError(f'Saved {name} codec mismatch')
    length = math.prod(shape) * 4
    stream = zlib.decompressobj()
    raw = stream.decompress(base64.b64decode(packed['data'], validate=True), length + 1)
    if len(raw) != length or not stream.eof or stream.unused_data:
        raise ValueError(f'Saved {name} payload length mismatch')
    return torch.from_numpy(np.frombuffer(raw, dtype='<i4').copy().reshape(shape))


def save_sample(path, sample):
    record = sample.to_dict()
    for name in ('rollout_routed_experts', *TOP_P_FIELDS):
        if record.get(name) is not None:
            record[name] = _pack(record[name])
    write_json(path, record)


def load_sample(path, *, num_layers=25, router_topk=4):
    from slime.utils.types import Sample

    record = json.loads(path.read_text())
    experts = record.get('rollout_routed_experts')
    if isinstance(experts, dict):
        if experts.get('shape') != [len(record['tokens']) - 1, num_layers, router_topk]:
            raise ValueError('Saved expert path shape/codec mismatch')
        record['rollout_routed_experts'] = _unpack(experts, 'expert path')
    if any(isinstance(record.get(name), dict) for name in TOP_P_FIELDS):
        ids, offsets = (_unpack(record[name], name) for name in TOP_P_FIELDS)
        if len(offsets) != record['response_length'] + 1 or int(offsets[-1]) != len(ids):
            raise ValueError('Saved top-p candidate sets do not match the response')
        record.update(zip(TOP_P_FIELDS, (ids, offsets)))
    return Sample.from_dict(record)
