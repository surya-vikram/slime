"""CPU-only regression coverage for newer checkpoint enum metadata."""

import enum
import pickle
import sys
import types

import pytest

from slime.backends.megatron_utils.checkpoint_compat import register_checkpoint_metadata_compat


@pytest.fixture
def enums(monkeypatch):
    names = ["megatron", "megatron.core", "megatron.core.transformer", "megatron.core.transformer.enums"]
    modules = {name: types.ModuleType(name) for name in names}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(modules[parent], child, module)
    return modules[names[-1]]


def test_newer_metadata_loads_without_changing_values(enums):
    original = enum.Enum("InferenceCudaGraphScope", {"none": 1, "layer": 2, "block": 3}, module=enums.__name__)
    enums.InferenceCudaGraphScope = original
    saved = pickle.dumps({"scopes": list(original), "iteration": 3478})
    del enums.InferenceCudaGraphScope
    with pytest.raises(AttributeError, match="InferenceCudaGraphScope"):
        pickle.loads(saved)
    assert register_checkpoint_metadata_compat()
    loaded = pickle.loads(saved)
    assert [(x.name, x.value) for x in loaded["scopes"]] == [("none", 1), ("layer", 2), ("block", 3)]
    assert loaded["iteration"] == 3478
    assert pickle.loads(pickle.dumps(loaded)) == loaded
    assert not register_checkpoint_metadata_compat()


def test_existing_enum_is_never_replaced(enums):
    original = enum.Enum("InferenceCudaGraphScope", {"none": 1, "future": 4})
    enums.InferenceCudaGraphScope = original
    assert not register_checkpoint_metadata_compat()
    assert enums.InferenceCudaGraphScope is original


def test_unknown_values_are_not_silently_coerced(enums):
    register_checkpoint_metadata_compat()
    with pytest.raises(ValueError):
        enums.InferenceCudaGraphScope(4)


def test_megatron_load_registers_before_deserialization():
    from pathlib import Path

    source = (Path(__file__).parents[1] / "slime/backends/megatron_utils/checkpoint.py").read_text()
    assert source.index("register_checkpoint_metadata_compat()") < source.index("return _load_checkpoint_megatron(")
