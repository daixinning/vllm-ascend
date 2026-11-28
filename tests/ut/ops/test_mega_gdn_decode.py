# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

import vllm_ascend.ops.mega_gdn_decode as mega


def make_case(batch=4):
    nk, nv, channels, slots = 2, 4, 1024, 8
    layer = SimpleNamespace(
        num_spec=0,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_size=4,
        num_k_heads=nk,
        num_v_heads=nv,
        tp_size=1,
        activation="silu",
        conv1d=SimpleNamespace(bias=None),
        norm=SimpleNamespace(
            eps=1e-6,
            norm_before_gate=True,
            group_size=None,
            activation="silu",
            weight=torch.ones(128, dtype=torch.bfloat16),
        ),
        A_log=torch.zeros(nv),
        _mega_gdn_dt_bias=torch.zeros(nv),
        kv_cache=(torch.zeros(slots, 3, channels, dtype=torch.bfloat16), torch.zeros(slots, nv, 128, 128)),
    )
    meta = SimpleNamespace(
        num_prefills=0,
        num_spec_decodes=0,
        spec_sequence_masks=None,
        num_decodes=batch,
        non_spec_state_indices_tensor=torch.arange(1, batch + 1, dtype=torch.int32),
    )
    qkv = torch.zeros(batch, channels, dtype=torch.bfloat16)
    z = torch.zeros(batch, nv, 128, dtype=torch.bfloat16)
    b = torch.zeros(batch, nv, dtype=torch.bfloat16)
    weight = torch.zeros(4, channels, dtype=torch.bfloat16)
    return layer, meta, qkv, z, b, b.clone(), weight, torch.zeros_like(z)


def test_supported_and_padded_page_state_layouts():
    args = make_case()
    assert mega.supports_mega_gdn_decode(*args)
    layer = args[0]
    conv, ssm = layer.kv_cache
    conv = torch.empty_strided(conv.shape, (conv[0].numel() + 64, *conv.stride()[1:]), dtype=conv.dtype)
    ssm = torch.empty_strided(ssm.shape, (ssm[0].numel() + 64, *ssm.stride()[1:]), dtype=ssm.dtype)
    layer.kv_cache = conv, ssm
    assert mega.supports_mega_gdn_decode(*args)


@pytest.mark.parametrize("field,value", [("num_spec", 1), ("head_k_dim", 64), ("activation", "relu")])
def test_unsupported_model_keeps_old_path(field, value):
    args = make_case()
    setattr(args[0], field, value)
    assert not mega.supports_mega_gdn_decode(*args)


@pytest.mark.parametrize("field,value", [("num_prefills", 1), ("num_spec_decodes", 1), ("num_decodes", 0)])
def test_mixed_speculative_or_idle_batches_keep_old_path(field, value):
    args = make_case()
    setattr(args[1], field, value)
    assert not mega.supports_mega_gdn_decode(*args)


def test_batch_and_dtype_guards():
    assert not mega.supports_mega_gdn_decode(*make_case(33))
    args = list(make_case())
    args[2] = args[2].float()
    assert not mega.supports_mega_gdn_decode(*args)


@pytest.mark.parametrize("fused", [True, False])
def test_output_norm_applied_exactly_once(monkeypatch, fused):
    args = make_case()
    layer, _, qkv, z, b, a, _, output = args
    layer.norm = Mock(side_effect=lambda x, gate: x + 3)

    def core(qkv, b, a, output, mega_z=None):
        output.fill_(7 if fused else 4)
        return fused

    layer._forward_core = core
    monkeypatch.setattr(mega, "get_forward_context", lambda: SimpleNamespace(no_compile_layers={"layer": layer}))
    monkeypatch.setattr(mega, "get_weight_prefetch_method", lambda: None)
    mega.ascend_gdn_core_with_norm(qkv, z, b, a, output, "layer")
    assert torch.all(output == 7)
    assert layer.norm.call_count == (0 if fused else 1)


def test_bias_repack_keeps_pointer_after_reload():
    layer = torch.nn.Module()
    layer.register_parameter("dt_bias", torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.bfloat16)))
    mega.prepare_mega_gdn_weights(layer)
    ptr = layer._mega_gdn_dt_bias.data_ptr()
    with torch.no_grad():
        layer.dt_bias.fill_(3)
    mega.prepare_mega_gdn_weights(layer)
    assert layer._mega_gdn_dt_bias.data_ptr() == ptr
    assert layer._mega_gdn_dt_bias.dtype == torch.float32
    assert torch.equal(layer._mega_gdn_dt_bias, torch.tensor([3.0, 3.0]))


def test_single_token_projection_views_survive_shared_dynamic_graph(monkeypatch):
    """Trace with multiple tokens, then run that same FX graph with one token.

    This mirrors vLLM's graph reuse without allowing Dynamo to retrace and
    hide the regression behind a separate graph specialized for one token.
    """
    layer = SimpleNamespace(key_dim=2048, value_dim=6144, tp_size=4, head_v_dim=128)
    seen = []

    def core(qkv, b, a, output, mega_z=None):
        seen.append((qkv, mega_z, b, a))
        output.copy_(mega_z)
        return True

    layer._forward_core = core
    monkeypatch.setattr(mega, "get_forward_context", lambda: SimpleNamespace(no_compile_layers={"layer": layer}))
    monkeypatch.setattr(mega, "get_weight_prefetch_method", lambda: None)

    def forward(qkvz, ba):
        output = qkvz.new_empty((qkvz.shape[0], 12, 128))
        torch.ops.vllm.ascend_gdn_core_with_projection(qkvz, ba, output, "layer")
        return output

    lib = torch.library.Library("vllm", "IMPL", "CPU")
    try:
        if not torch._C._dispatch_has_kernel_for_dispatch_key("vllm::ascend_gdn_core_with_projection", "CPU"):
            lib.impl("ascend_gdn_core_with_projection", mega.ascend_gdn_core_with_projection)
        graph = make_fx(forward, tracing_mode="symbolic")(torch.empty(4, 4096), torch.empty(4, 24))
        assert "ascend_gdn_core_with_projection" in graph.code
        assert "ascend_split_gdn_projection" not in graph.code
        qkvz = torch.arange(4096, dtype=torch.float32).reshape(1, 4096)
        ba = torch.arange(24, dtype=torch.float32).reshape(1, 24)
        output = graph(qkvz, ba)
    finally:
        lib._destroy()

    assert len(seen) == 1
    qkv, z, b, a = seen[0]
    assert qkv.data_ptr() == qkvz.data_ptr()
    assert z.data_ptr() == qkvz.data_ptr() + 2560 * qkvz.element_size()
    assert b.data_ptr() == ba.data_ptr()
    assert a.data_ptr() == ba.data_ptr() + 12 * ba.element_size()
    torch.testing.assert_close(output, qkvz[:, 2560:].reshape(1, 12, 128))
