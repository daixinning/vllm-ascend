#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""One quantize shared across projections, and the SP all-gather behind it."""

from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from vllm_ascend.ops import shared_quant_proj as proj_ops
from vllm_ascend.ops.shared_quant_proj import project_with_shared_quant

_NAMES = ("in_proj_qkvz", "in_proj_ba")


class _FakeWrapper:
    """CVLinearWrapper stand-in recording what each step is handed."""

    def __init__(self, events: list, column: int):
        self.events = events
        self.column = column
        self.linear = SimpleNamespace(bias=None)

    def quantize(self, x: torch.Tensor):
        self.events.append(("quantize", tuple(x.shape)))
        # One byte per element plus a per-token scale, like the real schemes.
        return x.to(torch.int8), torch.ones(x.shape[0])

    def matmul(self, quantized: torch.Tensor, scale=None, bias=None):
        self.events.append(("matmul", tuple(quantized.shape), tuple(scale.shape)))
        return quantized.to(torch.float32) + self.column


class _RecordingProjection(nn.Module):
    """Linear stand-in returning the ``(output, bias)`` tuple vLLM layers do."""

    def __init__(self, seen: list):
        super().__init__()
        self.seen = seen

    def forward(self, x: torch.Tensor):
        self.seen.append(tuple(x.shape))
        return x, None


def _owner_with_shared_wrappers(events: list, names: tuple[str, ...] = _NAMES) -> nn.Module:
    owner = nn.Module()
    wrappers = [_FakeWrapper(events, column) for column in range(len(names))]
    setattr(owner, proj_ops._SHARED_QUANT_CACHE, {names: wrappers})
    return owner


def _owner_without_sharing(seen: list, names: tuple[str, ...] = _NAMES) -> nn.Module:
    owner = nn.Module()
    # Cache a None verdict: the projections cannot share a quantized input.
    setattr(owner, proj_ops._SHARED_QUANT_CACHE, {names: None})
    for name in names:
        owner.add_module(name, _RecordingProjection(seen))
    return owner


def _tp2_all_gather(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    assert dim == 0
    return torch.cat([x, x], dim=0)


def test_quantize_runs_on_the_shard_before_the_sp_gather():
    events: list = []
    owner = _owner_with_shared_wrappers(events)
    shard = torch.ones(3, 4)

    with patch.object(proj_ops, "tensor_model_parallel_all_gather", side_effect=_tp2_all_gather) as all_gather:
        outputs = project_with_shared_quant(owner, shard, _NAMES, sp_num_tokens=5)

    # Quantization runs once, on the 3-row shard rather than the gathered rows.
    assert events[0] == ("quantize", (3, 4))
    assert sum(event[0] == "quantize" for event in events) == 1
    # The collective carries the quantized payload and its scale.
    assert all_gather.call_count == 2
    assert all_gather.call_args_list[0].args[0].dtype == torch.int8
    # Both are trimmed from the 6 gathered rows back to the real token count.
    matmuls = [event for event in events if event[0] == "matmul"]
    assert matmuls == [("matmul", (5, 4), (5,))] * len(_NAMES)
    assert [tuple(out.shape) for out in outputs] == [(5, 4)] * len(_NAMES)


def test_shared_quantize_without_sequence_parallelism_runs_no_collective():
    events: list = []
    owner = _owner_with_shared_wrappers(events)

    with (
        patch.object(proj_ops, "tensor_model_parallel_all_gather") as quantized_gather,
        patch.object(proj_ops, "sp_all_gather") as bf16_gather,
    ):
        outputs = project_with_shared_quant(owner, torch.ones(5, 4), _NAMES)

    quantized_gather.assert_not_called()
    bf16_gather.assert_not_called()
    # Still one quantize for the whole group - the point of sharing it.
    assert events[0] == ("quantize", (5, 4))
    assert sum(event[0] == "quantize" for event in events) == 1
    assert len(outputs) == len(_NAMES)


def test_unshareable_projections_gather_bf16_in_front():
    seen: list = []
    owner = _owner_without_sharing(seen)

    with (
        patch.object(proj_ops, "sp_all_gather", side_effect=_tp2_all_gather) as bf16_gather,
        patch.object(proj_ops, "tensor_model_parallel_all_gather") as quantized_gather,
    ):
        outputs = project_with_shared_quant(owner, torch.ones(3, 4), _NAMES, sp_num_tokens=5)

    # The bf16 gather stays in front of the projections, trimmed to the tokens.
    bf16_gather.assert_called_once()
    assert bf16_gather.call_args.args[0].dtype == torch.float32
    quantized_gather.assert_not_called()
    assert seen == [(5, 4), (5, 4)]
    assert len(outputs) == len(_NAMES)


def test_lone_projection_outside_sp_keeps_its_own_forward():
    # Nothing to share and no collective to move, so the quantize/matmul split
    # would only add overhead: the projection's own forward runs instead.
    seen: list = []
    owner = nn.Module()
    owner.add_module("qkv_proj", _RecordingProjection(seen))
    probe = SimpleNamespace(called=False)

    def _unexpected(*args, **kwargs):
        probe.called = True
        return None

    with patch.object(proj_ops, "_get_shared_quant_wrappers", side_effect=_unexpected):
        (output,) = project_with_shared_quant(owner, torch.ones(5, 4), ("qkv_proj",))

    assert not probe.called
    assert seen == [(5, 4)]
    assert tuple(output.shape) == (5, 4)


def test_lone_projection_under_sp_still_quantizes_before_the_gather():
    events: list = []
    owner = _owner_with_shared_wrappers(events, ("qkv_proj",))

    with patch.object(proj_ops, "tensor_model_parallel_all_gather", side_effect=_tp2_all_gather):
        (output,) = project_with_shared_quant(owner, torch.ones(3, 4), ("qkv_proj",), sp_num_tokens=5)

    assert events == [("quantize", (3, 4)), ("matmul", (5, 4), (5,))]
    assert tuple(output.shape) == (5, 4)
