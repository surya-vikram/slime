"""JSON completion persistence with compact, non-pickle expert-path storage."""

import base64
import json
import math
import zlib

from .core import write_json


def save_sample(path, sample):
    record = sample.to_dict()
    experts = record.get('rollout_routed_experts')
    if experts is not None:
        import numpy as np
        if hasattr(experts, 'detach'):
            experts = experts.detach().cpu().numpy()
        array = np.asarray(experts, dtype='<i4')
        record['rollout_routed_experts'] = {
            'codec': 'zlib-base64-int32le-v1', 'shape': list(array.shape),
            'data': base64.b64encode(zlib.compress(array.tobytes())).decode('ascii')}
    write_json(path, record)


def load_sample(path, *, num_layers=25, router_topk=4):
    import numpy as np
    import torch
    from slime.utils.types import Sample

    record = json.loads(path.read_text())
    experts = record.get('rollout_routed_experts')
    if isinstance(experts, dict):
        shape = experts['shape']
        if (experts.get('codec') != 'zlib-base64-int32le-v1' or len(shape) != 3 or
            shape != [len(record['tokens']) - 1, num_layers, router_topk]):
            raise ValueError('Saved expert path shape/codec mismatch')
        length = math.prod(shape) * 4
        stream = zlib.decompressobj()
        raw = stream.decompress(base64.b64decode(experts['data'], validate=True), length + 1)
        if len(raw) != length or not stream.eof or stream.unused_data:
            raise ValueError('Saved expert path payload length mismatch')
        record['rollout_routed_experts'] = torch.from_numpy(np.frombuffer(raw, dtype='<i4').copy().reshape(shape))
    return Sample.from_dict(record)
