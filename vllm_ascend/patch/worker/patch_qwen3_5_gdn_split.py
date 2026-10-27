#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Split the Qwen3.5 GDN ``in_proj_qkvz`` into ``in_proj_qkv`` + ``in_proj_z``.

Qwen3.5 checkpoints ship q/k/v/z as four separate weights and
``Qwen3_5Model.hf_to_vllm_mapper`` fuses them into one ``in_proj_qkvz``
module. The fused output then needs a split and a reshape on every forward.
Keeping z in its own projection drops those at the cost of one extra matmul,
which is the better trade on the GDN input projection.

Only the non-interleaved Qwen3.5 layout can be split this way. The Qwen3-Next
interleaved layout (``gqa_interleaved_layout=True``) stores qkvz as a single
fused tensor whose GQA interleaving has to survive TP sharding, so it keeps
the upstream module.

``AscendGatedDeltaNetAttention.forward`` picks the matching branch by probing
for the ``in_proj_qkv`` attribute, so this module only has to build the two
projections and leave the weight mapping consistent with them.

The switch is ``additional_config.ascend_config.enable_gdn_qkv_split``, read
when a GDN layer is constructed rather than when this module is imported: the
worker imports its patches (``adapt_patch()``) before it calls
``init_ascend_config()``, so nothing can read the config at import time. The
weight-mapping rewrite rides along on the first layer construction, which is
early enough for both of its consumers:

- ``configure_quant_config`` runs before any layer is built, but it reads
  ``hf_to_vllm_mapper.get_unstacked_mapper()``, which drops the stacked maps
  this patch edits, and it takes ``packed_modules_mapping`` by reference, so a
  later ``pop`` is still visible through ``quant_config``.
- The stacked mapper itself is only read by ``load_weights()``, after the
  model is fully constructed.
"""

from functools import lru_cache, wraps

from vllm.model_executor.layers.linear import ColumnParallelLinear, MergedColumnParallelLinear
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention


@lru_cache(maxsize=1)
def _qkv_split_enabled() -> bool:
    """Read the switch once; it cannot change within a process."""
    from vllm_ascend.ascend_config import get_ascend_config

    return get_ascend_config().enable_gdn_qkv_split


def _make_split_qkv_init(wrapped_init):
    """Wrap ``wrapped_init``, which must be whatever ``__init__`` is installed
    when this runs - not the one imported at module load. The caller patches
    ``__init__`` after the packed-conv-weight wrapper, so capturing it here
    keeps that wrapper in the chain instead of dropping it.
    """

    @wraps(wrapped_init)
    def _gdn_init_with_split_qkv(self, *args, **kwargs):
        wrapped_init(self, *args, **kwargs)

        # LoRA already creates the split layout upstream, and the interleaved
        # Qwen3-Next layout cannot be split at all.
        if not _qkv_split_enabled() or not hasattr(self, "in_proj_qkvz") or self.gqa_interleaved_layout:
            return

        _rewrite_weight_mapping()
        del self.in_proj_qkvz

        # q/k/v each need their own shard: rearrange_mixed_qkv splits the local
        # tensor as [key_dim // tp, key_dim // tp, value_dim // tp], which only
        # holds when every segment is partitioned independently. A plain
        # ColumnParallelLinear would slice the fused [q|k|v] dim as one block
        # and hand rank i the wrong rows for tp > 1.
        self.in_proj_qkv = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[self.key_dim, self.key_dim, self.value_dim],
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{self.prefix}.in_proj_qkv",
        )
        self.in_proj_z = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.value_dim,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{self.prefix}.in_proj_z",
        )

    return _gdn_init_with_split_qkv


def _rewrite_weight_mapping() -> None:
    """Point the weight loader at the split modules. Idempotent.

    Dropping the ``in_proj_qkvz`` entry is the marker: once it is gone the
    rewrite has already run, so no module-level flag is needed.
    """
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5Model
    from vllm.model_executor.models.qwen3_next import Qwen3NextModel
    from vllm.model_executor.models.utils import WeightsMapper

    if "in_proj_qkvz" not in Qwen3_5ForCausalLM.packed_modules_mapping:
        return

    # Keep the in_proj_b / in_proj_a fusion, drop the qkvz one: the two
    # projections now load ".in_proj_qkv" and ".in_proj_z" under their own
    # checkpoint names.
    Qwen3_5Model.hf_to_vllm_mapper = Qwen3NextModel.hf_to_vllm_mapper | WeightsMapper(
        orig_to_new_stacked={
            ".in_proj_b": (".in_proj_ba", 0),
            ".in_proj_a": (".in_proj_ba", 1),
        }
    )
    # The dict is inherited from Qwen3_5ForCausalLMBase, so this also covers
    # the MoE variant, and quant_config holds the same object by reference.
    Qwen3_5ForCausalLM.packed_modules_mapping.pop("in_proj_qkvz")


def apply_gdn_qkv_split_patch() -> None:
    """Install the wrapper. It no-ops unless the config enables the split.

    Call this after every other ``__init__`` patch is in place: it wraps the
    currently installed one, so anything patched afterwards would be lost.
    """
    QwenGatedDeltaNetAttention.__init__ = _make_split_qkv_init(QwenGatedDeltaNetAttention.__init__)
