"""Make Chimera visible without replacing the Slime image's Transformers."""

from slime_plugins.models.chimera import register_transformers


register_transformers()

import os
for flag, installer in (
    ('CHIMERA_MATCH_RMSNORM', 'install_rms_alignment'),
    ('CHIMERA_MATCH_DENSE_SWIGLU', 'install_dense_swiglu_alignment'),
    ('CHIMERA_SGLANG_FULL_BF16_REDUCTION', 'install_full_bf16_reduction'),
):
    if os.environ.get(flag, '0') != '1':
        continue
    # This also runs in CLI helpers whose stdout is a JSON protocol.
    import contextlib
    import sys
    with contextlib.redirect_stdout(sys.stderr):
        from importlib import import_module
        chimera_sglang_precision = import_module('slime_plugins.models.chimera_sglang_precision')
        getattr(chimera_sglang_precision, installer)()
