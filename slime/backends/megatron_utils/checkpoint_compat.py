"""Known metadata-only compatibility for checkpoints from newer Megatron versions.

This does not make arbitrary newer checkpoints compatible, change model math,
or relax checkpoint validation. Only load trusted checkpoints: Megatron common
state is pickle-based regardless of this compatibility registration.
"""

import enum
import logging


def register_checkpoint_metadata_compat():
    """Supply the exact upstream enum identity when the pinned image lacks it.

    Newer Megatron serializes InferenceCudaGraphScope in saved args/configs.
    Older images cannot unpickle those configs even during weights-only loads.
    Registering its original name and values preserves metadata without enabling
    inference graphs or replacing an enum supplied by the installed Megatron.
    Unknown enum values intentionally still raise ValueError.
    """
    from megatron.core.transformer import enums

    if hasattr(enums, "InferenceCudaGraphScope"):
        return False
    enums.InferenceCudaGraphScope = enum.Enum(
        "InferenceCudaGraphScope",
        {"none": 1, "layer": 2, "block": 3},
        module=enums.__name__,
    )
    logging.getLogger(__name__).info(
        "Registered metadata-only InferenceCudaGraphScope compatibility for older Megatron"
    )
    return True
