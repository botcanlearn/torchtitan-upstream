# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""Higher-precision Triton RMSNorm override.

``triton_rms_norm`` replaces every exact ``RMSNorm.Config`` selected by the
override framework with ``TritonRMSNorm``, which keeps the ``nn.RMSNorm``
parameter layout (so checkpoints are interchangeable) but computes the norm
with Triton kernels:

- Forward: the sum of squares is accumulated in FP64, or in FP32 with Kahan
  (Fast2Sum) compensation. ``use_kahan_summation`` selects between them and
  defaults to True (Kahan), which avoids FP64 arithmetic on GPUs with low FP64
  throughput. In FP64 mode ``rstd`` stays FP64 and ``x * rstd * weight`` is
  evaluated in FP64; in Kahan mode it is evaluated in FP32. Either way the output
  is rounded once to the output dtype.
- Backward: FP32 math. ``dx`` is a per-row two-pass kernel; ``dw`` is a
  deterministic two-stage reduction over rows (no atomics).
- Dtypes: ``input_dtype`` and ``out_dtype`` are separate settings.
  ``input_dtype`` rounds the input to that dtype inside the kernel. This equals
  casting the input to ``input_dtype`` before the norm, without materializing the
  cast copy; the norm math then runs on the rounded values. Backward re-rounds
  from the saved original input. ``out_dtype`` is the dtype of the returned
  tensor. If unset, it is ``input_dtype`` when that is set, and otherwise the
  input's dtype. Setting only ``out_dtype`` normalizes the unrounded input and
  rounds once, at the output. ``dx`` always has the original input's dtype.
"""

from dataclasses import dataclass, replace
from typing import Literal

import torch
import triton
import triton.language as tl

from torchtitan.config import derive, override, TORCH_DTYPE_MAP
from torchtitan.models.common.nn_modules import RMSNorm

__all__ = [
    "TritonRMSNorm",
    "rms_norm_backward_kernel",
    "rms_norm_forward_kernel",
    "triton_rms_norm",
    "triton_rms_norm_backward_op",
    "triton_rms_norm_op",
]

_TRITON_DTYPES = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


# Elements of x held per program in the row kernels (BLOCK_T * BLOCK_D).
_ROW_BLOCK_ELEMENTS = 8192
_MAX_BLOCK_D = 8192
_ROW_KERNEL_NUM_WARPS = 8
# dw kernel tile: rows per inner step and columns per program.
_DW_BLOCK_T = 32
_DW_BLOCK_D = 128
# Row-chunks per SM for the dw partial sums.
_DW_CHUNKS_PER_SM = 4


@triton.jit
def _fast2sum(a, b):
    """Return ``(s, e)`` with ``a + b == s + e`` exactly; requires ``|a| >= |b|``."""
    s = a + b
    e = b - (s - a)
    return s, e


@triton.jit
def _round_input(x, ROUND_INPUT: tl.constexpr, INPUT_DTYPE: tl.constexpr):
    """Round FP32 ``x`` through ``INPUT_DTYPE`` when ``ROUND_INPUT``."""
    if ROUND_INPUT:
        x = x.to(INPUT_DTYPE).to(tl.float32)
    return x


@triton.jit
def _rms_norm_forward_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    rstd_ptr,
    NUM_ROWS,
    eps,
    x_row_stride,
    D: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    USE_KAHAN: tl.constexpr,
    ROUND_INPUT: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    """``y = x * rsqrt(mean(x^2) + eps) * w`` for rows of ``x [NUM_ROWS, D]``."""
    rows = (tl.program_id(0).to(tl.int64) * BLOCK_T) + tl.arange(0, BLOCK_T)
    row_mask = rows < NUM_ROWS

    if USE_KAHAN:
        sum_sq = tl.zeros([BLOCK_T], dtype=tl.float32)
    else:
        sum_sq = tl.zeros([BLOCK_T], dtype=tl.float64)
    compensation = tl.zeros([BLOCK_T], dtype=tl.float32)
    for offset in tl.range(0, D, BLOCK_D):
        cols = offset + tl.arange(0, BLOCK_D)
        x = tl.load(
            x_ptr + rows[:, None] * x_row_stride + cols[None, :],
            mask=row_mask[:, None] & (cols[None, :] < D),
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        x = _round_input(x, ROUND_INPUT, INPUT_DTYPE)
        block_sum = tl.sum(x * x, axis=1)
        if USE_KAHAN:
            sum_sq, compensation = _fast2sum(sum_sq, block_sum + compensation)
        else:
            sum_sq += block_sum

    # rstd is FP64 in FP64 mode, so the multiplies below are FP64 too. Only the
    # stored copy (consumed by backward) is rounded to FP32.
    rstd = tl.rsqrt(sum_sq * (1.0 / D) + eps)
    tl.store(rstd_ptr + rows, rstd.to(tl.float32), mask=row_mask)

    for offset in tl.range(0, D, BLOCK_D):
        cols = offset + tl.arange(0, BLOCK_D)
        mask = row_mask[:, None] & (cols[None, :] < D)
        x = tl.load(
            x_ptr + rows[:, None] * x_row_stride + cols[None, :],
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x = _round_input(x, ROUND_INPUT, INPUT_DTYPE)
        y = x * rstd[:, None]
        if HAS_WEIGHT:
            w = tl.load(w_ptr + cols, mask=cols < D, other=1.0).to(tl.float32)
            y = y * w[None, :]
        # y is contiguous [NUM_ROWS, D].
        tl.store(y_ptr + rows[:, None] * D + cols[None, :], y, mask=mask)


@triton.jit
def _rms_norm_backward_dx_kernel(
    dy_ptr,
    x_ptr,
    w_ptr,
    rstd_ptr,
    dx_ptr,
    NUM_ROWS,
    dy_row_stride,
    x_row_stride,
    D: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    ROUND_INPUT: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    """``dx = rstd * (dy * w - x_hat * mean(dy * w * x_hat))``, ``x_hat = x * rstd``."""
    rows = (tl.program_id(0).to(tl.int64) * BLOCK_T) + tl.arange(0, BLOCK_T)
    row_mask = rows < NUM_ROWS
    rstd = tl.load(rstd_ptr + rows, mask=row_mask, other=0.0)

    correction = tl.zeros([BLOCK_T], dtype=tl.float32)
    for offset in tl.range(0, D, BLOCK_D):
        cols = offset + tl.arange(0, BLOCK_D)
        mask = row_mask[:, None] & (cols[None, :] < D)
        dy = tl.load(
            dy_ptr + rows[:, None] * dy_row_stride + cols[None, :],
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        x = tl.load(
            x_ptr + rows[:, None] * x_row_stride + cols[None, :],
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        x = _round_input(x, ROUND_INPUT, INPUT_DTYPE)
        if HAS_WEIGHT:
            w = tl.load(w_ptr + cols, mask=cols < D, other=1.0).to(tl.float32)
            dy = dy * w[None, :]
        correction += tl.sum(dy * x * rstd[:, None], axis=1)

    for offset in tl.range(0, D, BLOCK_D):
        cols = offset + tl.arange(0, BLOCK_D)
        mask = row_mask[:, None] & (cols[None, :] < D)
        dy = tl.load(
            dy_ptr + rows[:, None] * dy_row_stride + cols[None, :],
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        x = tl.load(
            x_ptr + rows[:, None] * x_row_stride + cols[None, :],
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x = _round_input(x, ROUND_INPUT, INPUT_DTYPE)
        if HAS_WEIGHT:
            w = tl.load(w_ptr + cols, mask=cols < D, other=1.0).to(tl.float32)
            dy = dy * w[None, :]
        x_hat = x * rstd[:, None]
        dx = rstd[:, None] * (dy - x_hat * (correction[:, None] * (1.0 / D)))
        # dx is contiguous [NUM_ROWS, D].
        tl.store(dx_ptr + rows[:, None] * D + cols[None, :], dx, mask=mask)


@triton.jit
def _rms_norm_backward_dw_kernel(
    dy_ptr,
    x_ptr,
    rstd_ptr,
    partial_dw_ptr,
    NUM_ROWS,
    ROWS_PER_CHUNK,
    dy_row_stride,
    x_row_stride,
    D: tl.constexpr,
    ROUND_INPUT: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    """Write ``partial_dw[chunk] = sum_{t in chunk} dy[t] * x[t] * rstd[t]``."""
    chunk = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    col_mask = cols < D

    row_begin = chunk * ROWS_PER_CHUNK
    row_end = tl.minimum(row_begin + ROWS_PER_CHUNK, NUM_ROWS)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for start in tl.range(row_begin, row_end, BLOCK_T):
        rows = start + tl.arange(0, BLOCK_T)
        row_mask = rows < row_end
        mask = row_mask[:, None] & col_mask[None, :]
        dy = tl.load(
            dy_ptr + rows[:, None] * dy_row_stride + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        x = tl.load(
            x_ptr + rows[:, None] * x_row_stride + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        x = _round_input(x, ROUND_INPUT, INPUT_DTYPE)
        rstd = tl.load(rstd_ptr + rows, mask=row_mask, other=0.0)
        acc += tl.sum(dy * x * rstd[:, None], axis=0)
    tl.store(partial_dw_ptr + chunk * D + cols, acc, mask=col_mask)


def _resolve_dtypes(
    x: torch.Tensor,
    input_dtype: torch.dtype | None,
    out_dtype: torch.dtype | None,
) -> tuple[bool, tl.dtype, torch.dtype]:
    """Return ``(round_input, triton_input_dtype, y_dtype)``."""
    for dtype in (input_dtype, out_dtype):
        if dtype is not None and dtype not in _TRITON_DTYPES:
            raise ValueError(f"Unsupported norm dtype {dtype}")
    round_input = input_dtype is not None and input_dtype != x.dtype
    return (
        round_input,
        _TRITON_DTYPES[input_dtype if round_input else x.dtype],
        out_dtype or input_dtype or x.dtype,
    )


def _as_rows(tensor: torch.Tensor) -> torch.Tensor:
    """View ``[..., D]`` as ``[T, D]`` with unit stride along ``D``."""
    rows = tensor.reshape(-1, tensor.shape[-1])
    return rows if rows.stride(-1) == 1 else rows.contiguous()


def _row_block_shape(dim: int) -> tuple[int, int]:
    block_d = min(triton.next_power_of_2(dim), _MAX_BLOCK_D)
    block_t = max(1, _ROW_BLOCK_ELEMENTS // block_d)
    return block_t, block_d


def rms_norm_forward_kernel(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    use_kahan_summation: bool = True,
    input_dtype: torch.dtype | None = None,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(y, rstd)`` for ``x [..., D]``; ``rstd`` is FP32 of shape ``x.shape[:-1]``.

    ``y`` has dtype ``out_dtype``, else ``input_dtype``, else ``x.dtype``.
    """
    dim = x.shape[-1]
    if weight is not None and weight.shape != (dim,):
        raise ValueError(f"weight must have shape ({dim},), got {tuple(weight.shape)}")
    round_input, triton_input_dtype, y_dtype = _resolve_dtypes(
        x, input_dtype, out_dtype
    )
    x_rows = _as_rows(x)
    num_rows = x_rows.shape[0]
    y = torch.empty(x.shape, dtype=y_dtype, device=x.device)
    rstd = torch.empty(x.shape[:-1], dtype=torch.float32, device=x.device)

    block_t, block_d = _row_block_shape(dim)
    _rms_norm_forward_kernel[(triton.cdiv(num_rows, block_t),)](
        x_rows,
        weight if weight is not None else x_rows,
        y,
        rstd,
        num_rows,
        eps,
        x_rows.stride(0),
        D=dim,
        HAS_WEIGHT=weight is not None,
        USE_KAHAN=use_kahan_summation,
        ROUND_INPUT=round_input,
        INPUT_DTYPE=triton_input_dtype,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=_ROW_KERNEL_NUM_WARPS,
    )
    return y, rstd


def rms_norm_backward_kernel(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor | None,
    rstd: torch.Tensor,
    input_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(dx, dw)``; ``dw`` is an empty tensor when ``weight`` is None.

    ``x`` is the original forward input; ``dx`` has ``x.dtype``.
    """
    dim = x.shape[-1]
    round_input, triton_input_dtype, _ = _resolve_dtypes(x, input_dtype, None)
    x_rows = _as_rows(x)
    dy_rows = _as_rows(dy)
    num_rows = x_rows.shape[0]
    dx = torch.empty(x.shape, dtype=x.dtype, device=x.device)

    block_t, block_d = _row_block_shape(dim)
    _rms_norm_backward_dx_kernel[(triton.cdiv(num_rows, block_t),)](
        dy_rows,
        x_rows,
        weight if weight is not None else x_rows,
        rstd,
        dx,
        num_rows,
        dy_rows.stride(0),
        x_rows.stride(0),
        D=dim,
        HAS_WEIGHT=weight is not None,
        ROUND_INPUT=round_input,
        INPUT_DTYPE=triton_input_dtype,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=_ROW_KERNEL_NUM_WARPS,
    )
    if weight is None:
        return dx, x.new_empty(0)

    # Two-stage dw: each program reduces a fixed chunk of rows in order, then the
    # chunks are summed by a fixed-order reduction, so the result is deterministic.
    num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    rows_per_chunk = max(
        _DW_BLOCK_T, triton.cdiv(num_rows, _DW_CHUNKS_PER_SM * num_sms)
    )
    num_chunks = triton.cdiv(num_rows, rows_per_chunk)
    partial_dw = torch.empty((num_chunks, dim), dtype=torch.float32, device=x.device)
    _rms_norm_backward_dw_kernel[(num_chunks, triton.cdiv(dim, _DW_BLOCK_D))](
        dy_rows,
        x_rows,
        rstd,
        partial_dw,
        num_rows,
        rows_per_chunk,
        dy_rows.stride(0),
        x_rows.stride(0),
        D=dim,
        ROUND_INPUT=round_input,
        INPUT_DTYPE=triton_input_dtype,
        BLOCK_T=_DW_BLOCK_T,
        BLOCK_D=_DW_BLOCK_D,
        num_warps=4,
    )
    return dx, partial_dw.sum(0).to(weight.dtype)


@torch.library.custom_op(
    "torchtitan::triton_rms_norm",
    mutates_args=(),
    device_types="cuda",
)
def triton_rms_norm_op(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    use_kahan_summation: bool = True,
    input_dtype: torch.dtype | None = None,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm over the last dim; returns ``(y, rstd)`` with FP32 ``rstd``."""
    return rms_norm_forward_kernel(
        x, weight, eps, use_kahan_summation, input_dtype, out_dtype
    )


@triton_rms_norm_op.register_fake
def _triton_rms_norm_op_fake(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    use_kahan_summation: bool = True,
    input_dtype: torch.dtype | None = None,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, y_dtype = _resolve_dtypes(x, input_dtype, out_dtype)
    return (
        torch.empty(x.shape, dtype=y_dtype, device=x.device),
        torch.empty(x.shape[:-1], dtype=torch.float32, device=x.device),
    )


@torch.library.custom_op(
    "torchtitan::triton_rms_norm_backward",
    mutates_args=(),
    device_types="cuda",
)
def triton_rms_norm_backward_op(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor | None,
    rstd: torch.Tensor,
    input_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gradients ``(dx, dw)`` of ``torchtitan::triton_rms_norm``."""
    return rms_norm_backward_kernel(dy, x, weight, rstd, input_dtype)


@triton_rms_norm_backward_op.register_fake
def _triton_rms_norm_backward_op_fake(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor | None,
    rstd: torch.Tensor,
    input_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    dw = x.new_empty(0) if weight is None else torch.empty_like(weight)
    return torch.empty(x.shape, dtype=x.dtype, device=x.device), dw


def _triton_rms_norm_setup_context(ctx, inputs, output) -> None:
    x, weight, _, _, input_dtype, _ = inputs
    _, rstd = output
    ctx.input_dtype = input_dtype
    ctx.save_for_backward(x, weight, rstd)
    ctx.mark_non_differentiable(rstd)


def _triton_rms_norm_autograd_backward(ctx, grad_y, grad_rstd):
    x, weight, rstd = ctx.saved_tensors
    dx, dw = triton_rms_norm_backward_op(grad_y, x, weight, rstd, ctx.input_dtype)
    return dx, (None if weight is None else dw), None, None, None, None


triton_rms_norm_op.register_autograd(
    _triton_rms_norm_autograd_backward, setup_context=_triton_rms_norm_setup_context
)


class TritonRMSNorm(RMSNorm):
    """RMSNorm with FP64 / Kahan-FP32 sum-of-squares accumulation (Triton)."""

    @dataclass(kw_only=True, slots=True)
    class Config(RMSNorm.Config):
        # True: FP32 accumulation with Kahan compensation. False: FP64 accumulation.
        use_kahan_summation: bool = True
        # Round the input to this dtype in-kernel before normalizing, like casting
        # it first without the copy. None keeps the input unrounded.
        input_dtype: torch.dtype | None = None
        # Output dtype (e.g. FP32 for the LM head or residual add). None means
        # ``input_dtype`` if set, else the input's dtype.
        out_dtype: torch.dtype | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        self.use_kahan_summation = config.use_kahan_summation
        self.input_dtype = config.input_dtype
        self.out_dtype = config.out_dtype

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        y, _ = triton_rms_norm_op(
            input,
            self.weight,
            self.eps,
            self.use_kahan_summation,
            self.input_dtype,
            self.out_dtype,
        )
        return y


@override(
    target=RMSNorm.Config,
    exact=True,
    description="RMSNorm with FP64 / Kahan-FP32 sum-of-squares accumulation in Triton.",
)
def triton_rms_norm(
    cfg: RMSNorm.Config,
    *,
    use_kahan_summation: bool = True,
    input_dtype: Literal["float16", "bfloat16", "float32"] | None = None,
    out_dtype: Literal["float16", "bfloat16", "float32"] | None = None,
) -> TritonRMSNorm.Config:
    # Dtypes are names so they can be passed as JSON on the --override CLI.
    sharding_config = cfg.sharding_config
    if sharding_config is not None:
        # The Triton op is opaque to SPMD type propagation, so run it as a local
        # region typed by the norm's declared input and output layouts.
        sharding_config = replace(sharding_config, local_spmd=True)
    return derive(
        cfg,
        TritonRMSNorm.Config,
        use_kahan_summation=use_kahan_summation,
        input_dtype=None if input_dtype is None else TORCH_DTYPE_MAP[input_dtype],
        out_dtype=None if out_dtype is None else TORCH_DTYPE_MAP[out_dtype],
        sharding_config=sharding_config,
    )
