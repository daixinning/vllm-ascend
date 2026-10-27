# SPDX-License-Identifier: Apache-2.0
"""The Qwen3.5 GDN in_proj_qkvz -> in_proj_qkv + in_proj_z split patch."""

from unittest.mock import patch

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention

from vllm_ascend.patch.worker.patch_qwen3_5_gdn_split import _make_split_qkv_init

_SWITCH = "vllm_ascend.patch.worker.patch_qwen3_5_gdn_split._qkv_split_enabled"


def _install_split_over(wrapped):
    """Install the split wrapper on top of ``wrapped``, as the patch does."""
    with patch.object(QwenGatedDeltaNetAttention, "__init__", wrapped):
        return _make_split_qkv_init(QwenGatedDeltaNetAttention.__init__)


def test_split_wrapper_chains_onto_the_installed_init():
    # Regression: capturing __init__ at module import time instead of at
    # install time dropped the packed-conv-weight wrapper, so
    # ascend_conv1d_weight was never registered.
    calls = []

    def previously_installed_init(self, *args, **kwargs):
        calls.append("packed_conv_weight")

    init = _install_split_over(previously_installed_init)
    with patch(_SWITCH, return_value=False):
        init(object())

    assert calls == ["packed_conv_weight"]


def test_split_wrapper_is_inert_when_config_disables_it():
    class _Layer:
        def __init__(self):
            self.in_proj_qkvz = object()
            self.gqa_interleaved_layout = False

    init = _install_split_over(lambda self: None)
    layer = _Layer()
    with patch(_SWITCH, return_value=False):
        init(layer)

    # The upstream fused layout is left exactly as it was.
    assert hasattr(layer, "in_proj_qkvz")
    assert not hasattr(layer, "in_proj_qkv")


def test_interleaved_layout_is_never_split():
    # Qwen3-Next stores qkvz as one fused tensor whose GQA interleaving has to
    # survive TP sharding, so it keeps the upstream module even when enabled.
    class _Layer:
        def __init__(self):
            self.in_proj_qkvz = object()
            self.gqa_interleaved_layout = True

    init = _install_split_over(lambda self: None)
    layer = _Layer()
    with patch(_SWITCH, return_value=True):
        init(layer)

    assert hasattr(layer, "in_proj_qkvz")
    assert not hasattr(layer, "in_proj_qkv")
