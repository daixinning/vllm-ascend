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

from collections import defaultdict

import vllm.v1.engine.core as _engine_core_mod
from vllm.logger import logger
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.patch.platform.patch_balance_schedule import BalanceDPEngineCoreProc, _balance_scheduling_enabled
from vllm_ascend.patch.platform.patch_dyntra_lb_core import DyntraLBDPEngineCoreProc, _get_dyntra_lb_config
from vllm_ascend.patch.platform.patch_dyntra_lb_core import _print_rank_0 as dyntra_print_rank_0
from vllm_ascend.patch.platform.patch_pp_mtp import _patch_engine_core as pp_mtp_patch_post_step
from vllm_ascend.patch.platform.patch_profiling_chunk import _apply_profiling_patches

_PATCHED = False

_OriginalRunEngineCore = EngineCoreProc.run_engine_core

_OriginalExecuteDummyBatch = EngineCore.execute_dummy_batch

_OriginalProcessEngineStep = EngineCoreProc._process_engine_step


def _patch_dp_engine_core_proc(vllm_config, dp_rank: int):
    dyntra_lb_config = _get_dyntra_lb_config(vllm_config)
    if dyntra_lb_config.enabled:
        dyntra_print_rank_0(
            "Enable DyntraLB DP load balancing.",
            dp_rank,
            dyntra_lb_config.enable_diagnostics,
        )
        _engine_core_mod.DPEngineCoreProc = DyntraLBDPEngineCoreProc
    elif _balance_scheduling_enabled(vllm_config):
        _engine_core_mod.DPEngineCoreProc = BalanceDPEngineCoreProc


class _KVTransferOutputShim:
    __slots__ = ("kv_connector_output",)

    def __init__(self, kv_connector_output: KVConnectorOutput | None):
        self.kv_connector_output = kv_connector_output


_WAITING_QUEUE_NAMES = ("waiting", "skipped_waiting", "kv_holding_waiting")


def _any_request_awaiting_remote_kv(scheduler) -> bool:
    found_queue = False
    for queue_name in _WAITING_QUEUE_NAMES:
        queue = getattr(scheduler, queue_name, None)
        if queue is None:
            continue
        found_queue = True
        if any(req.status == RequestStatus.WAITING_FOR_REMOTE_KVS for req in queue):
            return True
    if found_queue:
        return False
    requests = getattr(scheduler, "requests", None)
    if requests is None:
        return False
    return any(req.status == RequestStatus.WAITING_FOR_REMOTE_KVS for req in requests.values())


def _fail_kv_load_requests(engine_core, scheduler, failed_req_ids: set[str]) -> None:
    finished_reqs = scheduler.finish_requests(failed_req_ids, RequestStatus.FINISHED_ERROR)
    if not finished_reqs:
        return
    by_client = defaultdict[int, list[str]](list)
    for request in finished_reqs:
        by_client[request.client_index].append(request.request_id)
    send_error = getattr(engine_core, "_send_error_outputs_to_client", None)
    for client_index, req_ids in by_client.items():
        if send_error is not None:
            send_error(req_ids, client_index)
        else:
            logger.warning("[dummy_kv_poll] KV load failed for requests %s", req_ids)


def _poll_kv_transfer_finished(engine_core) -> None:
    if not envs_ascend.VLLM_ASCEND_ENABLE_DUMMY_KV_POLL:
        return
    if engine_core.vllm_config.kv_transfer_config is None:
        logger.debug("[dummy_kv_poll] skipped: no kv connector configured")
        return
    scheduler = getattr(engine_core, "scheduler", None)
    if scheduler is None:
        return
    if not _any_request_awaiting_remote_kv(scheduler):
        logger.debug("[dummy_kv_poll] skipped: no request awaiting remote KV.")
        return

    try:
        results = engine_core.model_executor.collective_rpc("get_kv_transfer_finished")
    except Exception:
        logger.warning("[dummy_kv_poll] worker hook unavailable", exc_info=True)
        return
    if not results or any(result is None for result in results):
        logger.debug("[dummy_kv_poll] skipped: worker without a KV connector.")
        return

    shims = []
    for result in results:
        sending, recving = result[0], result[1]
        failed = result[2] if len(result) > 2 else ()
        if not (sending or recving or failed):
            shims.append(_KVTransferOutputShim(None))
            continue
        shims.append(
            _KVTransferOutputShim(
                KVConnectorOutput(
                    finished_sending=set(sending or ()),
                    finished_recving=set(recving or ()),
                    failed_recving=set(failed or ()),
                )
            )
        )

    kv_output: KVConnectorOutput | None
    aggregator = getattr(engine_core.model_executor, "kv_output_aggregator", None)
    if aggregator is not None:
        kv_output = getattr(aggregator.aggregate(shims, output_rank=0), "kv_connector_output", None)
    else:
        kv_output = KVConnectorOutput(finished_sending=set(), finished_recving=set(), failed_recving=set())
        for shim in shims:
            worker_output = shim.kv_connector_output
            if worker_output is None:
                continue
            kv_output.finished_sending |= worker_output.finished_sending or set()
            kv_output.finished_recving |= worker_output.finished_recving or set()
            kv_output.failed_recving |= worker_output.failed_recving or set()
    if kv_output is None:
        return

    failed_recving = set(kv_output.failed_recving or ())
    finished_sending = set(kv_output.finished_sending or ())
    # Failed loads are also completed transfers. Keep their completion signal
    # so the scheduler can retry them or release blocks after finishing them.
    finished_recving = set(kv_output.finished_recving or ())
    if not (finished_sending or finished_recving or failed_recving):
        return

    if failed_recving:
        handle_failed = getattr(scheduler, "_handle_failed_recving", None)
        error_req_ids = handle_failed(failed_recving) if callable(handle_failed) else set()
        if error_req_ids:
            _fail_kv_load_requests(engine_core, scheduler, error_req_ids)

    scheduler._update_from_kv_xfer_finished(kv_output)
    logger.debug(
        "[dummy_kv_poll] promoted %d recving, %d sending, %d failed",
        len(finished_recving),
        len(finished_sending),
        len(failed_recving),
    )


def _execute_dummy_batch_with_kv_poll(self) -> None:
    _OriginalExecuteDummyBatch(self)
    _poll_kv_transfer_finished(self)


def _process_engine_step_with_kv_poll(self) -> bool:
    executed = _OriginalProcessEngineStep(self)
    if executed:
        return executed
    if isinstance(self, _engine_core_mod.DPEngineCoreProc):
        return executed
    _poll_kv_transfer_finished(self)
    return executed


def _run_engine_core_patch_func(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
    vllm_config = kwargs.get("vllm_config")
    ascend_config = init_ascend_config(vllm_config)

    # Call _apply_profiling_patches to patch EngineCore.__init__
    # when the child unpickles the patch.
    if ascend_config.scheduler_config.profiling_chunk_config.enabled:
        _apply_profiling_patches()

    _patch_dp_engine_core_proc(vllm_config, dp_rank)

    return _OriginalRunEngineCore(*args, dp_rank=dp_rank, local_dp_rank=local_dp_rank, **kwargs)


def _apply_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    # Patch EngineCore.post_step
    pp_mtp_patch_post_step()
    EngineCore.execute_dummy_batch = _execute_dummy_batch_with_kv_poll
    EngineCoreProc._process_engine_step = _process_engine_step_with_kv_poll
    # Patch EngineCoreProc.run_engine_core
    # And in _run_engine_core_patch_func:
    # 1. Patch EngineCore.__init__ by _apply_profiling_patches
    # 2. Patch class DPEngineCoreProc by _patch_dp_engine_core_proc
    EngineCoreProc.run_engine_core = staticmethod(_run_engine_core_patch_func)


_apply_patch()
