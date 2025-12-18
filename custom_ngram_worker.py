"""
Custom NGRAM Worker with Acceptance Confirmation

This module extends the SGLang NGRAMWorker to add acceptance confirmation messages
whenever a token is accepted from the draft memory during speculative decoding.
"""

import logging
from typing import Optional, Union

from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.ngram_worker import NGRAMWorker

logger = logging.getLogger(__name__)


class CustomNGRAMWorker(NGRAMWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
        verbose_acceptance: bool = True,
        acceptance_log_level: str = "INFO",
    ):
        super().__init__(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            moe_ep_rank=moe_ep_rank,
            nccl_port=nccl_port,
            target_worker=target_worker,
        )

        self.verbose_acceptance = verbose_acceptance
        self.acceptance_count = 0
        self.total_accepted_tokens = 0

        self.acceptance_logger = logging.getLogger(f"{__name__}.acceptance")
        if verbose_acceptance:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("[CUSTOM-NGRAM] %(message)s")
            handler.setFormatter(formatter)
            self.acceptance_logger.addHandler(handler)
            self.acceptance_logger.setLevel(
                getattr(logging, acceptance_log_level.upper(), logging.INFO)
            )

    def forward_batch_generation(self, batch) -> GenerationBatchResult:
        if hasattr(batch, "forward_mode"):
            if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
                result = self.target_worker.forward_batch_generation(batch)
                return result
            else:

                class MinimalScheduleBatch:
                    def __init__(self, model_worker_batch):
                        self.forward_mode = model_worker_batch.forward_mode
                        self.reqs = model_worker_batch.reqs
                        self.batch_size = lambda: len(model_worker_batch.reqs)
                        self.get_model_worker_batch = lambda: model_worker_batch
                        self.return_logprob = model_worker_batch.return_logprob
                        self.sampling_info = model_worker_batch.sampling_info
                        self.top_logprobs_nums = model_worker_batch.top_logprobs_nums
                        self.token_ids_logprobs = model_worker_batch.token_ids_logprobs
                        self.page_size = (
                            self.page_size if hasattr(self, "page_size") else 16
                        )

                try:
                    minimal_batch = MinimalScheduleBatch(batch)
                    result = super().forward_batch_generation(minimal_batch)

                    if (
                        result.num_accepted_tokens is not None
                        and result.num_accepted_tokens > 0
                        and self.verbose_acceptance
                    ):
                        self._print_acceptance_confirmation(
                            batch, result.num_accepted_tokens
                        )

                    return result
                except Exception as e:
                    print(f"DEBUG: NGRAM parent call failed: {e}")
                    result = self.target_worker.forward_batch_generation(batch)
                    return result
        else:
            result = super().forward_batch_generation(batch)

            if (
                result.num_accepted_tokens is not None
                and result.num_accepted_tokens > 0
                and self.verbose_acceptance
            ):
                self._print_acceptance_confirmation(batch, result.num_accepted_tokens)

            return result

    def _print_acceptance_confirmation(
        self, batch: Union[ScheduleBatch, ModelWorkerBatch], num_accepted: int
    ):
        self.acceptance_count += 1
        self.total_accepted_tokens += num_accepted

        confirmation_msg = f"accepted {num_accepted} token(s) from draft memory"

        self.acceptance_logger.info(confirmation_msg)

        if self.acceptance_logger.isEnabledFor(logging.DEBUG):
            if hasattr(batch, "batch_size") and hasattr(batch, "reqs"):
                batch_info = f"batch_size={batch.batch_size()}, request_ids={[req.rid for req in batch.reqs]}"
            else:
                batch_info = (
                    f"request_count={getattr(batch, 'req_pool_indices', 'unknown')}"
                )
            self.acceptance_logger.debug(
                f"Acceptance details: {confirmation_msg} | {batch_info}"
            )

    def get_acceptance_stats(self):
        return {
            "total_acceptance_events": self.acceptance_count,
            "total_accepted_tokens": self.total_accepted_tokens,
            "average_tokens_per_acceptance": (
                self.total_accepted_tokens / self.acceptance_count
                if self.acceptance_count > 0
                else 0
            ),
            "verbose_acceptance": self.verbose_acceptance,
        }

    def set_verbose_acceptance(self, verbose: bool):
        self.verbose_acceptance = verbose

    def print_stats(self):
        stats = self.get_acceptance_stats()
        print("\n=== CustomNGRAMWorker Acceptance Statistics ===")
        print(f"Total acceptance events: {stats['total_acceptance_events']}")
        print(f"Total accepted tokens: {stats['total_accepted_tokens']}")
        print(
            f"Average tokens per acceptance: {stats['average_tokens_per_acceptance']:.2f}"
        )


def create_custom_ngram_worker(
    server_args: ServerArgs,
    gpu_id: int,
    tp_rank: int,
    dp_rank: Optional[int],
    moe_ep_rank: int,
    nccl_port: int,
    target_worker: TpModelWorker,
    **custom_kwargs,
) -> CustomNGRAMWorker:
    verbose_acceptance = custom_kwargs.get("verbose_acceptance", True)
    acceptance_log_level = custom_kwargs.get("acceptance_log_level", "INFO")

    return CustomNGRAMWorker(
        server_args=server_args,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
        dp_rank=dp_rank,
        moe_ep_rank=moe_ep_rank,
        nccl_port=nccl_port,
        target_worker=target_worker,
        verbose_acceptance=verbose_acceptance,
        acceptance_log_level=acceptance_log_level,
    )
