# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Test packaged operator dispatcher contracts without the DSL compiler."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops import packaged_attention as ops


@pytest.mark.parametrize("name", ["quant_lightning_indexer", "quant_sparse_lightning_indexer"])
@pytest.mark.parametrize("layout", ["legacy", "wheel", "source"])
def test_dsl_module_layouts(monkeypatch, name, layout):
    compute, metadata = object(), object()
    prefix = "" if layout == "source" else "ops."
    suffix = "_dsl" if layout == "legacy" else ""
    modules = {
        f"{prefix}{name}{suffix}": SimpleNamespace(**{name: compute}),
        f"{prefix}{name}_metadata{suffix}": SimpleNamespace(**{f"{name}_metadata": metadata}),
    }

    def import_module(module_name):
        if module_name in modules:
            return modules[module_name]
        missing = "ops" if layout == "source" and module_name.startswith("ops.") else module_name
        raise ModuleNotFoundError(name=missing)

    monkeypatch.setattr(ops, "import_module", import_module)
    assert ops._get_dsl_ops.__wrapped__(name) == (compute, metadata)


@pytest.mark.parametrize("fail_metadata", [False, True])
def test_dsl_internal_dependency_error(monkeypatch, fail_metadata):
    dependency_error = ModuleNotFoundError(name="dsl_compiler_dependency")
    calls = []

    def import_module(module_name):
        calls.append(module_name)
        if fail_metadata and len(calls) == 1:
            return SimpleNamespace()
        raise dependency_error

    monkeypatch.setattr(ops, "import_module", import_module)
    with pytest.raises(ModuleNotFoundError) as error:
        ops._get_dsl_ops.__wrapped__("quant_lightning_indexer")
    assert error.value is dependency_error
    assert len(calls) == (2 if fail_metadata else 1)


@pytest.mark.parametrize("return_value,candidate_blocks", [(False, -1), (True, 2048)])
def test_indexer_meta_pipeline(return_value, candidate_blocks):
    q = torch.empty((7, 32, 64), dtype=torch.uint8, device="meta")
    k = torch.empty((3, 128, 1, 72), dtype=torch.uint8, device="meta")
    w = torch.empty((7, 32), dtype=torch.float32, device="meta")
    scale = torch.empty((7, 32, 1), dtype=torch.float32, device="meta")
    k_scale = torch.empty((3, 128, 1), dtype=torch.float32, device="meta")
    indices, values, candidates, lengths = ops.quant_lightning_indexer(
        q,
        k,
        w,
        scale,
        k_scale,
        512,
        0,
        layout_k="PA_BBND",
        return_value=return_value,
        candidate_topk_blocks=candidate_blocks,
        candidate_block_size=8,
    )
    assert indices.shape == (7, 1, 512)
    assert indices.dtype == torch.int32
    assert values.shape == ((7, 1, 512) if return_value else (0,))
    assert values.dtype == torch.bfloat16
    assert all(t.device.type == "meta" for t in (indices, values, candidates, lengths))
    assert candidates.dtype == lengths.dtype == torch.int32
    if candidate_blocks < 0:
        assert candidates.shape == lengths.shape == (0,)
        return

    assert candidates.shape == (7, 1, 2048)
    assert lengths.shape == (7, 1)
    sparse_indices, sparse_values = ops.quant_sparse_lightning_indexer(
        q,
        k,
        w,
        scale,
        candidates,
        lengths,
        512,
        0,
        8,
        descale_k=k_scale,
        layout_k="PA_BBND",
        return_value=return_value,
    )
    assert sparse_indices.shape == indices.shape
    assert sparse_indices.dtype == indices.dtype
    assert sparse_values.shape == values.shape
    assert sparse_values.dtype == values.dtype
    assert sparse_indices.device.type == sparse_values.device.type == "meta"


@pytest.mark.parametrize(
    "name",
    [
        "quant_lightning_indexer_metadata",
        "quant_sparse_lightning_indexer_metadata",
        "mixed_quant_sparse_flash_mla_metadata",
    ],
)
def test_metadata_meta_abi(name):
    lengths = torch.empty((7, 1), dtype=torch.int32, device="meta")
    cu = torch.empty((3,), dtype=torch.int32, device="meta")
    if name == "mixed_quant_sparse_flash_mla_metadata":
        metadata = getattr(ops, name)(
            lengths,
            lengths,
            cu_seqlens_q=cu,
            num_heads_q=64,
            num_heads_kv=1,
            head_dim=512,
            quant_mode=0,
        )
    else:
        kwargs = dict(cu_seqlens_q=cu, num_heads_q=32, num_heads_k=1, head_dim=128, topk=512)
        if name == "quant_sparse_lightning_indexer_metadata":
            kwargs.update(candidate_block_length=lengths, quant_mode=0, candidate_block_size=8)
        metadata = getattr(ops, name)(**kwargs)
    assert metadata.shape == (1024,)
    assert metadata.dtype == torch.int32
    assert metadata.device.type == "meta"


@pytest.mark.parametrize("return_lse", [False, True])
def test_attention_meta_outputs(return_lse):
    q = torch.empty((7, 64, 512), dtype=torch.bfloat16, device="meta")
    kv = torch.empty((3, 128, 1, 584), dtype=torch.uint8, device="meta")
    output, lse = ops.mixed_quant_sparse_flash_mla(
        q,
        ori_kv=kv,
        quant_mode=0,
        return_softmax_lse=return_lse,
    )
    assert output.shape == (7, 64, 512)
    assert output.dtype == torch.bfloat16
    assert lse.shape == ((1, 7, 64) if return_lse else (0,))
    assert lse.dtype == torch.float32
    assert output.device.type == lse.device.type == "meta"
