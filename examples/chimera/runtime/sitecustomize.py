"""Make Chimera visible without replacing the Slime image's Transformers."""

from slime_plugins.models.chimera import register_transformers


register_transformers()
