"""给 AttentionSpec 注入 page_size_padded，配合 VLLM_ASCEND_TARGET_BLOCK_SIZE。

强制小 block 时 attention 的 real page 小于 mamba page，而 KVCacheManager 只能管理
单一 page 大小；upstream 的 unify_kv_cache_spec_page_size() 会通过**抬高 block_size**
来对齐，正好抵消我们要的小 block。这里改为把 attention 的 page 显式 pad 到共同值，
unify 看到两边相等即直接返回。

allocation 侧无需改动：worker/gpu/attn_utils.py 见到 page_size_padded 就用
torch.as_strided 以 padded page 为 block 跨步建视图，block 内部布局不变。

不设 VLLM_ASCEND_TARGET_BLOCK_SIZE 时本 patch 完全不生效（no-op）。
"""
import os

from vllm.logger import logger
from vllm.model_executor.layers.attention.attention import Attention

_orig_get_kv_cache_spec = Attention.get_kv_cache_spec


def _patched_get_kv_cache_spec(self, vllm_config):
    spec = _orig_get_kv_cache_spec(self, vllm_config)
    padded = os.environ.get("_VLLM_ASCEND_ATTN_PAGE_PADDED")
    if not padded or spec is None:
        return spec
    padded = int(padded)
    real = spec.page_size_bytes
    if padded > real:
        # frozen dataclass，用 object.__setattr__（upstream kv_cache_utils.py 亦如此）
        object.__setattr__(spec, "page_size_padded", padded)
        logger.info_once(
            "Padding attention page %d -> %d B (x%.2f) so it matches the mamba page, "
            "keeping block_size=%d instead of letting unify raise it.",
            real, padded, padded / real, spec.block_size,
        )
    return spec


Attention.get_kv_cache_spec = _patched_get_kv_cache_spec
