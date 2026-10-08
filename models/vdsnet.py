"""VDSNet architecture used by the TIP 2026 manuscript.

This module isolates the paper model from the historical ablation collection in
``ab_arch.py``.  The implementation corresponds to ``RD_3`` with:

* a four-level U-Net backbone;
* value-driven, single-direction Mamba scanning;
* grouped dynamic convolution in the local Mixer branch;
* HiLo cross-feature bridges;
* multi-granularity DINO guidance during training only; and
* the lightweight latent FFN used by the final 7.02M-parameter model.

DINOv2 is optional and is deliberately not part of inference by default.  Pass
``dino_model_path`` while training to enable the paper's MVGL supervision.
"""

import math
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from mamba_ssm.ops.selective_scan_interface import (
    selective_scan_fn,
    selective_scan_ref,
)


class CudaLatencyProfiler:
    """Accumulates nested CUDA event intervals for one model forward."""

    STAGES = (
        "model_total_ms",
        "vdrs_total_ms",
        "value_prediction_ms",
        "descending_argsort_ms",
        "inverse_argsort_ms",
        "gather_restore_ms",
        "selective_scan_ms",
    )

    def __init__(self):
        self.reset()

    def reset(self):
        self._events = defaultdict(list)
        self._block_metadata = {}

    @staticmethod
    def start():
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA latency profiling requires an available CUDA device")
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def end(self, stage, start_event, block_name=None):
        if stage not in self.STAGES:
            raise ValueError(f"unknown latency profiling stage: {stage}")
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        self._events[stage].append((start_event, end_event, block_name))

    def register_block(self, block_name, channels, height, width):
        self._block_metadata[block_name] = {
            "channels": channels,
            "height": height,
            "width": width,
            "tokens": height * width,
        }

    def retained_event_count(self):
        return 2 * sum(len(intervals) for intervals in self._events.values())

    def summary(self, synchronize=True):
        if synchronize:
            torch.cuda.synchronize()
        result = {}
        per_block = defaultdict(lambda: defaultdict(float))
        for stage in self.STAGES:
            result[stage] = 0.0
            for start, end, block_name in self._events.get(stage, ()):
                elapsed = start.elapsed_time(end)
                result[stage] += elapsed
                if block_name is not None:
                    per_block[block_name][stage] += elapsed
        result["vdrs_call_count"] = len(self._events.get("vdrs_total_ms", ()))
        result["recorded_event_count"] = self.retained_event_count()
        result["sorting_total_ms"] = (
            result["descending_argsort_ms"] + result["inverse_argsort_ms"]
        )
        known = (
            result["value_prediction_ms"]
            + result["sorting_total_ms"]
            + result["gather_restore_ms"]
            + result["selective_scan_ms"]
        )
        result["other_vdrs_ms"] = result["vdrs_total_ms"] - known
        result["per_block"] = {}
        for block_name, stages in per_block.items():
            block = {stage: stages.get(stage, 0.0) for stage in self.STAGES}
            block["sorting_total_ms"] = (
                block["descending_argsort_ms"] + block["inverse_argsort_ms"]
            )
            block["other_vdrs_ms"] = (
                block["vdrs_total_ms"]
                - block["value_prediction_ms"]
                - block["sorting_total_ms"]
                - block["gather_restore_ms"]
                - block["selective_scan_ms"]
            )
            block.update(self._block_metadata.get(block_name, {}))
            result["per_block"][block_name] = block
        self.reset()
        result["retained_event_count_after_summary"] = self.retained_event_count()
        return result


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)

    def forward(self, x):
        return self.proj(x)


class LayScale(nn.Module):
    def __init__(self, dim, init_value=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1) * init_value)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, groups=x.shape[1])


class DownSample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.proj(x)


class UpSample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.proj(x)


class HiLo_Conv(nn.Module):
    """Cross-feature bridge used by the three decoder skip connections."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        window_size=2,
        alpha=0.5,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        head_dim = dim // num_heads
        self.dim = dim
        self.reduce = nn.Conv2d(2 * dim, dim, 1, bias=False)
        self.l_heads = int(alpha * num_heads)
        self.l_dim = self.l_heads * head_dim
        self.h_heads = num_heads - self.l_heads
        self.h_dim = self.h_heads * head_dim
        self.window_size = window_size
        self.scale = qk_scale or head_dim**-0.5

        if self.l_heads > 0:
            self.l_q = nn.Linear(dim, self.l_dim, bias=qkv_bias)
            self.l_kv = nn.Linear(dim, self.l_dim * 2, bias=qkv_bias)
            self.l_proj = nn.Linear(self.l_dim, self.l_dim)
        if self.h_heads > 0:
            self.h_qkv = nn.Linear(dim, self.h_dim * 3, bias=qkv_bias)
            self.h_proj = nn.Linear(self.h_dim, self.h_dim)

    def hifi(self, x):
        batch, channels, height, width = x.shape
        ws = self.window_size
        if height % ws or width % ws:
            raise ValueError(f"feature size {(height, width)} must be divisible by {ws}")

        x = x.permute(0, 2, 3, 1)
        h_group, w_group = height // ws, width // ws
        total_group = h_group * w_group
        x = x.reshape(batch, h_group, ws, w_group, ws, channels).transpose(2, 3)
        qkv = self.h_qkv(x).reshape(
            batch, total_group, -1, 3, self.h_heads, self.h_dim // self.h_heads
        )
        q, k, v = qkv.permute(3, 0, 1, 4, 2, 5)
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        x = (attn @ v).transpose(2, 3).reshape(
            batch, h_group, w_group, ws, ws, self.h_dim
        )
        x = x.transpose(2, 3).reshape(batch, height, width, self.h_dim)
        return self.h_proj(x).permute(0, 3, 1, 2)

    def lowfi(self, low, high):
        batch, channels, height, width = high.shape
        q = self.l_q(high.permute(0, 2, 3, 1)).reshape(
            batch, height * width, self.l_heads, self.l_dim // self.l_heads
        )
        q = q.permute(0, 2, 1, 3)
        low = low.reshape(batch, channels, -1).permute(0, 2, 1)
        kv = self.l_kv(low).reshape(
            batch, -1, 2, self.l_heads, self.l_dim // self.l_heads
        )
        k, v = kv.permute(2, 0, 3, 1, 4)
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(batch, height, width, self.l_dim)
        return self.l_proj(x).permute(0, 3, 1, 2)

    def forward(self, low_feature, high_feature):
        if self.l_heads == 0:
            return self.hifi(high_feature)
        high_out = self.hifi(high_feature)
        low_out = self.lowfi(self.reduce(low_feature), high_feature)
        return torch.cat((low_out, high_out), dim=1)


class DynamicConv2d(nn.Module):
    """Depthwise dynamic convolution mixed from several learned kernel bases."""

    def __init__(
        self,
        dim,
        hw,
        kernel_shape=3,
        reduction_ratio=4,
        num_dynamic_kernels=1,
    ):
        super().__init__()
        if num_dynamic_kernels <= 1:
            raise ValueError("DynamicConv2d requires num_dynamic_kernels > 1")
        self.num_dynamic_kernels = num_dynamic_kernels
        self.K = kernel_shape
        self.P = nn.Parameter(
            torch.empty(num_dynamic_kernels, dim, kernel_shape, kernel_shape)
        )
        self.pool = nn.AdaptiveAvgPool2d((kernel_shape, kernel_shape))
        hidden = dim // reduction_ratio
        self.proj = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, dim * num_dynamic_kernels, 1, bias=False),
        )
        nn.init.trunc_normal_(self.P, std=0.02)

    def forward(self, x):
        batch, channels, height, width = x.shape
        scale = self.proj(self.pool(x)).reshape(
            batch, self.num_dynamic_kernels, channels, self.K, self.K
        )
        scale = scale.softmax(dim=1)
        kernel = (scale * self.P.unsqueeze(0)).sum(dim=1)
        kernel = kernel.reshape(-1, 1, self.K, self.K)
        x = F.conv2d(
            x.reshape(1, -1, height, width),
            kernel,
            padding=self.K // 2,
            groups=batch * channels,
        )
        return x.reshape(batch, channels, height, width)


class DeformableConvOffsetGenerator2(nn.Module):
    """Generate an independent deformable-sampling offset field per channel group."""

    def __init__(
        self, in_channels, kernel_size=1, stride=1, num_offset_groups=1
    ):
        super().__init__()
        if in_channels % num_offset_groups:
            raise ValueError(
                f"in_channels ({in_channels}) must be divisible by "
                f"num_offset_groups ({num_offset_groups})"
            )
        self.kernel_size = kernel_size
        self.num_offset_groups = num_offset_groups
        group_channels = in_channels // num_offset_groups
        self.conv3x3 = nn.Conv2d(
            in_channels,
            group_channels,
            3,
            stride=stride,
            padding=1,
            groups=num_offset_groups,
        )
        self.gelu = nn.GELU()
        self.conv1x1 = nn.Conv2d(
            group_channels,
            2 * kernel_size * kernel_size * num_offset_groups,
            1,
        )

    def forward(self, x):
        offset = self.conv1x1(self.gelu(self.conv3x3(x)))
        batch, _, height, width = offset.shape
        offset = offset.view(
            batch,
            self.num_offset_groups,
            2,
            self.kernel_size * self.kernel_size,
            height,
            width,
        )
        return offset[:, :, 0], offset[:, :, 1]


def deformable_sampling_frequency(
    x, offset_x, offset_y, num_offset_groups=1
):
    """Accumulate bilinear sampling hits into a pixel-aligned value map."""

    batch, channels, height, width = x.shape
    if channels % num_offset_groups:
        raise ValueError(
            f"channels ({channels}) must be divisible by "
            f"num_offset_groups ({num_offset_groups})"
        )

    kernel_points = offset_x.shape[2]
    kernel_size = int(kernel_points**0.5)
    y_offset = torch.linspace(
        -(kernel_size - 1) / 2,
        (kernel_size - 1) / 2,
        steps=kernel_size,
        device=x.device,
    )
    x_offset = y_offset
    ky, kx = torch.meshgrid(y_offset, x_offset, indexing="ij")

    yy, xx = torch.meshgrid(
        torch.arange(height, device=x.device, dtype=torch.float32),
        torch.arange(width, device=x.device, dtype=torch.float32),
        indexing="ij",
    )
    sample_y = (
        yy.view(1, 1, 1, height, width)
        + ky.reshape(1, 1, kernel_points, 1, 1)
        + offset_y
    ).clamp(0, height - 1)
    sample_x = (
        xx.view(1, 1, 1, height, width)
        + kx.reshape(1, 1, kernel_points, 1, 1)
        + offset_x
    ).clamp(0, width - 1)

    y0, x0 = sample_y.floor().long(), sample_x.floor().long()
    y1, x1 = (y0 + 1).clamp(max=height - 1), (x0 + 1).clamp(max=width - 1)
    dy, dx = sample_y - y0.float(), sample_x - x0.float()
    weights = (
        (y0, x0, (1 - dx) * (1 - dy)),
        (y0, x1, dx * (1 - dy)),
        (y1, x0, (1 - dx) * dy),
        (y1, x1, dx * dy),
    )

    frequency = x.new_zeros(batch, num_offset_groups, height, width)
    for batch_index in range(batch):
        for group_index in range(num_offset_groups):
            target = frequency[batch_index, group_index].view(-1)
            for y_index, x_index, weight in weights:
                flat_index = (
                    y_index[batch_index, group_index].reshape(-1) * width
                    + x_index[batch_index, group_index].reshape(-1)
                )
                target.scatter_add_(
                    0, flat_index, weight[batch_index, group_index].reshape(-1)
                )
    return frequency


def reorder_by_value(
    x,
    value_map,
    num_value_groups=1,
    profiler=None,
    block_name=None,
    descending=True,
):
    batch, channels, height, width = x.shape
    group_channels = channels // num_value_groups
    x = x.view(batch, num_value_groups, group_channels, height * width)

    start = profiler.start() if profiler is not None else None
    indices = value_map.view(batch, num_value_groups, -1).argsort(
        dim=-1, descending=descending
    )
    if profiler is not None:
        profiler.end("descending_argsort_ms", start, block_name)

    start = profiler.start() if profiler is not None else None
    gather_indices = indices.unsqueeze(2).expand(-1, -1, group_channels, -1)
    x = x.gather(dim=-1, index=gather_indices)
    if profiler is not None:
        profiler.end("gather_restore_ms", start, block_name)
    return x.view(batch, channels, height, width), indices


def restore_from_value_order(
    x, indices, num_value_groups=1, profiler=None, block_name=None
):
    batch, channels, height, width = x.shape
    group_channels = channels // num_value_groups
    x = x.view(batch, num_value_groups, group_channels, height * width)

    inverse_start = profiler.start() if profiler is not None else None
    inverse = indices.argsort(dim=-1).unsqueeze(2).expand(
        -1, -1, group_channels, -1
    )
    if profiler is not None:
        profiler.end("inverse_argsort_ms", inverse_start, block_name)

    restore_start = profiler.start() if profiler is not None else None
    x = x.gather(dim=-1, index=inverse).view(batch, channels, height, width)
    if profiler is not None:
        profiler.end("gather_restore_ms", restore_start, block_name)
    return x


class ValueDrivenSS2D(nn.Module):
    """Single-direction selective scan ordered by the learned value map."""

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=1,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        num_offset_groups=1,
        value_kernel_size=1,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.num_offset_groups = max(1, num_offset_groups)
        self.mvgl_temperature = 0.1

        self.in_proj = nn.Linear(
            d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )
        self.conv2d = nn.Conv2d(
            self.d_inner,
            self.d_inner,
            d_conv,
            padding=(d_conv - 1) // 2,
            groups=self.d_inner,
            bias=conv_bias,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + 2 * d_state,
            bias=False,
            **factory_kwargs,
        )
        self.x_proj_weight = nn.Parameter(x_proj.weight.unsqueeze(0))

        dt_proj = self._init_dt(
            self.dt_rank,
            self.d_inner,
            dt_scale,
            dt_init,
            dt_min,
            dt_max,
            dt_init_floor,
            **factory_kwargs,
        )
        self.dt_projs_weight = nn.Parameter(dt_proj.weight.unsqueeze(0))
        self.dt_projs_bias = nn.Parameter(dt_proj.bias.unsqueeze(0))
        self.A_logs = self._init_a_log(d_state, self.d_inner)
        self.Ds = self._init_d(self.d_inner)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(
            self.d_inner, d_model, bias=bias, **factory_kwargs
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.offset = DeformableConvOffsetGenerator2(
            d_model,
            kernel_size=value_kernel_size,
            num_offset_groups=self.num_offset_groups,
        )
        self.af = None
        self.df = None
        self._latency_profiler = None
        self._latency_block_name = None
        # Optional, analysis-only callback.  It is deliberately not a Module,
        # Parameter, or buffer, so enabling the audit never changes state_dict.
        self._state_propagation_debug_callback = None
        self._state_propagation_debug_name = None

    @staticmethod
    def _init_dt(
        dt_rank,
        d_inner,
        dt_scale,
        dt_init,
        dt_min,
        dt_max,
        dt_init_floor,
        **factory_kwargs,
    ):
        projection = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(projection.weight, std)
        elif dt_init == "random":
            nn.init.uniform_(projection.weight, -std, std)
        else:
            raise NotImplementedError(f"unsupported dt_init: {dt_init}")
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        with torch.no_grad():
            projection.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        projection.bias._no_reinit = True
        return projection

    @staticmethod
    def _init_a_log(d_state, d_inner):
        values = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=d_inner,
        )
        parameter = nn.Parameter(values.log())
        parameter._no_weight_decay = True
        return parameter

    @staticmethod
    def _init_d(d_inner):
        parameter = nn.Parameter(torch.ones(d_inner))
        parameter._no_weight_decay = True
        return parameter

    def _project_scan_parameters(self, x):
        batch, _, height, width = x.shape
        length = height * width
        u = x.float().view(batch, -1, length)
        x_dbl = torch.einsum(
            "b k d l,k c d -> b k c l",
            x.view(batch, 1, -1, length),
            self.x_proj_weight,
        )
        dts, Bs, Cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2
        )
        dts = torch.einsum(
            "b k r l,k d r -> b k d l", dts, self.dt_projs_weight
        )
        return u, dts.float().view(batch, -1, length), Bs.float(), Cs.float()

    def _run_projected_scan(self, u, dts, Bs, Cs, include_direct=True):
        scan = selective_scan_fn if u.is_cuda else selective_scan_ref
        output = scan(
            u,
            dts,
            -torch.exp(self.A_logs.float()).view(-1, self.d_state),
            Bs,
            Cs,
            self.Ds.float().view(-1) if include_direct else None,
            z=None,
            delta_bias=self.dt_projs_bias.float().view(-1),
            delta_softplus=True,
            return_last_state=False,
        )
        return output

    def forward_core(self, x, include_direct=True):
        batch, _, height, width = x.shape
        projected = self._project_scan_parameters(x)
        output = self._run_projected_scan(*projected, include_direct=include_direct)
        return output.view(batch, -1, height, width)

    @staticmethod
    def _permute_projected_scan(projected, indices):
        """Apply one shared spatial permutation to precomputed SSM inputs."""
        u, dts, Bs, Cs = projected
        if indices.shape[1] != 1:
            raise NotImplementedError(
                "shared-projection state analysis currently requires "
                "num_offset_groups=1"
            )
        token_indices = indices[:, 0]

        def gather_channels(tensor):
            return tensor.gather(
                -1, token_indices.unsqueeze(1).expand(-1, tensor.shape[1], -1)
            )

        def gather_group_state(tensor):
            return tensor.gather(
                -1,
                token_indices[:, None, None, :].expand(
                    -1, tensor.shape[1], tensor.shape[2], -1
                ),
            )

        return (
            gather_channels(u),
            gather_channels(dts),
            gather_group_state(Bs),
            gather_group_state(Cs),
        )

    def _debug_recurrent_responses(self, x, value_map, vdrs_indices):
        """Return C_t h_t under three scan orders without changing inference.

        ``x`` is the single shared post-in-projection/post-depthwise-convolution
        tensor that is passed to the SSM.  Passing ``D=None`` to the same
        selective scan implementation removes the direct ``D * x`` term
        exactly, leaving the recurrent readout ``C_t h_t``.  No output norm,
        z-gating, output projection, local-branch fusion, residual, or MLP is
        included in the returned tensors.
        """
        batch, channels, height, width = x.shape
        identity = torch.arange(height * width, device=x.device).view(1, 1, -1)
        identity = identity.expand(batch, self.num_offset_groups, -1)

        projected = self._project_scan_parameters(x)
        standard = self._run_projected_scan(
            *projected, include_direct=False
        ).view(batch, channels, height, width)
        standard = restore_from_value_order(
            standard, identity, self.num_offset_groups
        )

        reverse_indices = value_map.view(
            batch, self.num_offset_groups, -1
        ).argsort(dim=-1, descending=False)
        reverse = self._run_projected_scan(
            *self._permute_projected_scan(projected, reverse_indices),
            include_direct=False,
        ).view(batch, channels, height, width)
        reverse = restore_from_value_order(
            reverse, reverse_indices, self.num_offset_groups
        )

        vdrs = self._run_projected_scan(
            *self._permute_projected_scan(projected, vdrs_indices),
            include_direct=False,
        ).view(batch, channels, height, width)
        vdrs = restore_from_value_order(vdrs, vdrs_indices, self.num_offset_groups)

        responses = {
            "standard": standard,
            "reverse": reverse,
            "vdrs": vdrs,
        }
        expected_shape = x.shape
        if any(response.shape != expected_shape for response in responses.values()):
            raise RuntimeError(
                f"state-propagation debug shape mismatch in "
                f"{self._state_propagation_debug_name}: expected {expected_shape}"
            )
        self._state_propagation_debug_callback(
            {
                "module_name": self._state_propagation_debug_name,
                "value_map": value_map.detach(),
                "responses": {key: value.detach() for key, value in responses.items()},
                "descending_indices": vdrs_indices.detach(),
                "ascending_indices": reverse_indices.detach(),
            }
        )

    def compute_dino_kl_loss(self, temperature=None):
        if self.af is None or self.df is None:
            return self.A_logs.new_zeros(())

        temperature = temperature or self.mvgl_temperature
        batch, _, height, width = self.af.shape
        dino = self.df.detach().norm(dim=1, keepdim=True)
        dino = F.interpolate(
            dino, size=(height, width), mode="bilinear", align_corners=False
        )
        predicted = F.softmax(
            self.af.view(batch, -1) / temperature, dim=-1
        )
        guided = F.softmax(dino.view(batch, -1) / temperature, dim=-1)

        ratios = (0.25, 0.5, 0.75, 1.0)
        total = predicted.new_zeros(())
        flat_dino = dino.view(batch, -1)
        token_count = height * width
        for index, ratio in enumerate(ratios):
            topk = flat_dino.topk(max(1, int(token_count * ratio)), dim=1).indices
            mask = torch.zeros_like(flat_dino).scatter_(1, topk, 1.0)
            p = predicted * mask
            q = guided * mask
            p = p / (p.sum(dim=1, keepdim=True) + 1e-8)
            q = q / (q.sum(dim=1, keepdim=True) + 1e-8)
            total = total + (0.5**index) * F.kl_div(
                (p + 1e-6).log(), q, reduction="batchmean"
            )
        return total

    def forward(self, x, dino_feat=None):
        profiler = self._latency_profiler
        block_name = self._latency_block_name
        vdrs_start = profiler.start() if profiler is not None else None

        batch, _, height, width = x.shape
        if profiler is not None:
            profiler.register_block(
                block_name, self.d_model, height, width
            )
        xz = self.in_proj(x.permute(0, 2, 3, 1))
        x, z = xz.chunk(2, dim=-1)
        x = self.act(self.conv2d(x.permute(0, 3, 1, 2).contiguous()))

        value_start = profiler.start() if profiler is not None else None
        offset_x, offset_y = self.offset(x)
        value_map = deformable_sampling_frequency(
            x, offset_x, offset_y, self.num_offset_groups
        )
        if profiler is not None:
            profiler.end("value_prediction_ms", value_start, block_name)
        if self.training and dino_feat is not None:
            self.af = value_map
            self.df = dino_feat
        else:
            self.af = None
            self.df = None

        scan_input = x
        x, indices = reorder_by_value(
            x,
            value_map,
            self.num_offset_groups,
            profiler=profiler,
            block_name=block_name,
        )
        if self._state_propagation_debug_callback is not None:
            # Run analysis first and the production scan last.  This ordering
            # prevents repeated fused-kernel calls from retaining/overwriting
            # any temporary storage before the official result is consumed.
            with torch.no_grad():
                self._debug_recurrent_responses(
                    scan_input.detach(),
                    value_map.detach(),
                    indices.detach(),
                )
        scan_start = profiler.start() if profiler is not None else None
        x = self.forward_core(x)
        if profiler is not None:
            profiler.end("selective_scan_ms", scan_start, block_name)
        x = restore_from_value_order(
            x,
            indices,
            self.num_offset_groups,
            profiler=profiler,
            block_name=block_name,
        )
        x = self.out_norm(x.permute(0, 2, 3, 1))
        x = self.out_proj(x * F.silu(z))
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.permute(0, 3, 1, 2)
        if profiler is not None:
            profiler.end("vdrs_total_ms", vdrs_start, block_name)
        return x


class STE_GroupNorm(nn.Module):
    def __init__(self, dim, reduction_ratio=8):
        super().__init__()
        inner_dim = max(16, dim // reduction_ratio)
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.GroupNorm(4, dim),
            nn.GELU(),
            nn.Conv2d(dim, inner_dim, 1),
            nn.GroupNorm(4, inner_dim),
            nn.GELU(),
            nn.Conv2d(inner_dim, dim, 1),
            nn.GroupNorm(4, dim),
        )

    def forward(self, x):
        return self.proj(x) + x


class MambaConvMixer(nn.Module):
    def __init__(
        self,
        dim,
        hw,
        d_state,
        kernel_size,
        num_dynamic_kernels,
        num_offset_groups,
        reduction_ratio=4,
        value_kernel_size=1,
    ):
        super().__init__()
        if dim % 2:
            raise ValueError(f"dim ({dim}) must be even")
        branch_dim = dim // 2
        self.global_unit = ValueDrivenSS2D(
            d_model=branch_dim,
            d_state=d_state,
            expand=1,
            num_offset_groups=num_offset_groups,
            value_kernel_size=value_kernel_size,
        )
        self.local_unit = DynamicConv2d(
            branch_dim,
            hw,
            kernel_shape=kernel_size,
            reduction_ratio=reduction_ratio,
            num_dynamic_kernels=num_dynamic_kernels,
        )
        self.STE = STE_GroupNorm(dim)

    def forward(self, x, dino_feat=None):
        global_feature, local_feature = x.chunk(2, dim=1)
        global_feature = self.global_unit(global_feature, dino_feat)
        local_feature = self.local_unit(local_feature)
        return self.STE(torch.cat((global_feature, local_feature), dim=1))


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(0.0)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = (keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)).floor()
    return x.div(keep_prob) * mask


class SEModule(nn.Module):
    def __init__(self, channels, se_ratio=8):
        super().__init__()
        hidden = max(1, channels // se_ratio)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        scale = torch.sigmoid(self.fc2(F.silu(self.fc1(self.avg(x)))))
        return x * scale


class LatentSandwich(nn.Module):
    def __init__(
        self,
        channels,
        reduce_ratio=0.25,
        kernel_size=5,
        dilation=1,
        se_ratio=8,
        residual_scale=0.8,
        drop_path_prob=0.05,
    ):
        super().__init__()
        reduced = max(1, int(channels * reduce_ratio))
        padding = ((kernel_size - 1) // 2) * dilation
        self.pw1 = nn.Conv2d(channels, reduced, 1)
        self.pw2 = nn.Conv2d(reduced, channels, 1)
        self.dw = nn.Conv2d(
            reduced,
            reduced,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=reduced,
        )
        self.act = nn.GELU()
        self.se = SEModule(reduced, se_ratio) if se_ratio else None
        self.residual_scale = residual_scale
        self.drop_path_prob = drop_path_prob

        nn.init.kaiming_normal_(self.pw1.weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(self.pw1.bias)
        nn.init.kaiming_normal_(self.pw2.weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(self.pw2.bias)
        nn.init.kaiming_normal_(self.dw.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.dw.bias)

    def forward(self, x):
        residual = x
        x = self.act(self.pw1(x))
        x = self.act(self.dw(x))
        if self.se is not None:
            x = self.se(x)
        x = drop_path(self.pw2(x), self.drop_path_prob, self.training)
        return x + self.residual_scale * residual


class MCMBlock(nn.Module):
    def __init__(
        self,
        dim,
        hw,
        kernel_shape,
        num_dynamic_kernels,
        num_offset_groups,
        d_state,
        mlp_ratio=4.0,
        latent=False,
        value_kernel_size=1,
    ):
        super().__init__()
        self.mixer = MambaConvMixer(
            dim,
            hw,
            d_state=d_state,
            kernel_size=kernel_shape,
            num_dynamic_kernels=num_dynamic_kernels,
            num_offset_groups=num_offset_groups,
            value_kernel_size=value_kernel_size,
        )
        if latent:
            self.mlp = LatentSandwich(dim)
            self._mlp_nhwc = False
        else:
            self.mlp = MLP(dim, int(dim * mlp_ratio))
            self._mlp_nhwc = True
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.skip_scale1 = nn.Parameter(torch.ones(dim))
        self.skip_scale2 = nn.Parameter(torch.ones(dim))

    def forward(self, x, dino_feat=None):
        normed = self.norm1(x.permute(0, 2, 3, 1))
        normed = normed.permute(0, 3, 1, 2).contiguous()
        mid = (
            x * self.skip_scale1.view(1, -1, 1, 1)
            + self.mixer(normed, dino_feat)
        )
        normed = self.norm2(mid.permute(0, 2, 3, 1))
        if self._mlp_nhwc:
            mlp_out = self.mlp(normed).permute(0, 3, 1, 2)
        else:
            mlp_out = self.mlp(normed.permute(0, 3, 1, 2).contiguous())
        return mid * self.skip_scale2.view(1, -1, 1, 1) + mlp_out


class VDSNet(nn.Module):
    """Paper VDSNet model.

    Args:
        num_dynamic_kernels: Number of learned kernel bases mixed by the local
            DynamicConv2d branch at each feature level.
        num_offset_groups: Number of independently offset and value-sorted
            channel groups in the global VDRS branch at each feature level.
        dino_model_path: Optional local DINOv2 directory.  When supplied, DINO
            features are extracted only in training mode for MVGL.  Inference
            never executes DINOv2.
    """

    def __init__(
        self,
        in_channels=3,
        dim=32,
        hw=256,
        out_channels=3,
        num_blocks=(4, 6, 6, 8),
        kernel_shape=(7, 7, 7, 7),
        num_dynamic_kernels=(2, 2, 3, 4),
        num_offset_groups=(1, 1, 1, 1),
        based_state=4,
        num_refinements=4,
        num_heads=(1, 2, 4),
        value_kernel_size=1,
        dino_model_path=None,
        dino_layers=(3, 3, 4, 5),
    ):
        super().__init__()
        if not (
            len(num_blocks)
            == len(kernel_shape)
            == len(num_dynamic_kernels)
            == len(num_offset_groups)
            == 4
        ):
            raise ValueError(
                "num_blocks, kernel_shape, num_dynamic_kernels and "
                "num_offset_groups must have 4 entries"
            )

        dims = [dim * (2**index) for index in range(4)]
        resolutions = [hw // (2**index) for index in range(4)]
        d_state = based_state * 4

        def blocks(level, count, latent=False):
            return nn.ModuleList(
                [
                    MCMBlock(
                        dims[level],
                        resolutions[level],
                        kernel_shape[level],
                        num_dynamic_kernels[level],
                        num_offset_groups[level],
                        d_state,
                        latent=latent,
                        value_kernel_size=value_kernel_size,
                    )
                    for _ in range(count)
                ]
            )

        self.patchEmbed = OverlapPatchEmbed(in_channels, dim)
        self.encoder_level1 = blocks(0, num_blocks[0])
        self.down1_2 = DownSample(dims[0])
        self.encoder_level2 = blocks(1, num_blocks[1])
        self.down2_3 = DownSample(dims[1])
        self.encoder_level3 = blocks(2, num_blocks[2])
        self.down3_4 = DownSample(dims[2])
        self.latent = blocks(3, num_blocks[3], latent=True)

        self.HiLo3 = HiLo_Conv(dims[2], num_heads[2])
        self.up4_3 = UpSample(dims[3])
        self.decoder_level3 = blocks(2, num_blocks[2])
        self.HiLo2 = HiLo_Conv(dims[1], num_heads[1])
        self.up3_2 = UpSample(dims[2])
        self.decoder_level2 = blocks(1, num_blocks[1])
        self.HiLo1 = HiLo_Conv(dims[0], num_heads[0])
        self.up2_1 = UpSample(dims[1])
        self.decoder_level1 = blocks(0, num_blocks[0])
        self.refinement = blocks(0, num_refinements)
        self.layScale = LayScale(in_channels)
        self.head = nn.Conv2d(dim, out_channels, 1)

        self.dino_layers = tuple(dino_layers)
        self.dino_model = None
        self.dino_processor = None
        self._latency_profiler = None
        if dino_model_path is not None:
            self._load_dino(dino_model_path)

    def _load_dino(self, model_path):
        from transformers import AutoImageProcessor, AutoModel

        model_path = str(Path(model_path).expanduser())
        self.dino_model = AutoModel.from_pretrained(
            model_path,
            output_hidden_states=True,
            trust_remote_code=True,
            use_safetensors=True,
            local_files_only=True,
        )
        self.dino_processor = AutoImageProcessor.from_pretrained(
            model_path, local_files_only=True
        )
        self.dino_model.eval()
        self.dino_model.requires_grad_(False)

    @torch.no_grad()
    def extract_dino_features(self, images):
        if self.dino_model is None or self.dino_processor is None:
            return None
        self.dino_model.eval()
        from torchvision.transforms.functional import to_pil_image

        uint8_images = torch.clamp(images * 255, 0, 255).to(torch.uint8)
        pil_images = [to_pil_image(image.cpu()) for image in uint8_images]
        inputs = self.dino_processor(
            images=pil_images, return_tensors="pt"
        ).to(images.device)
        hidden_states = self.dino_model(**inputs).hidden_states
        features = []
        for layer_index in self.dino_layers:
            tokens = hidden_states[layer_index][:, 1:, :]
            side = int(tokens.shape[1] ** 0.5)
            feature = tokens.permute(0, 2, 1).reshape(
                tokens.shape[0], tokens.shape[2], side, side
            )
            features.append(feature)
        return features

    @staticmethod
    def _run(blocks, x, dino_feat):
        for block in blocks:
            x = block(x, dino_feat)
        return x

    def set_mvgl_temperature(self, temperature):
        for module in self.modules():
            if isinstance(module, ValueDrivenSS2D):
                module.mvgl_temperature = float(temperature)

    def enable_latency_profiling(self):
        """Enable per-forward CUDA event collection for all VDRS blocks."""
        if self._latency_profiler is None:
            self._latency_profiler = CudaLatencyProfiler()
        for name, module in self.named_modules():
            if isinstance(module, ValueDrivenSS2D):
                module._latency_profiler = self._latency_profiler
                module._latency_block_name = name
        self._latency_profiler.reset()

    def disable_latency_profiling(self):
        """Disable profiling without changing model parameters or computation."""
        for module in self.modules():
            if isinstance(module, ValueDrivenSS2D):
                module._latency_profiler = None
                module._latency_block_name = None
        self._latency_profiler = None

    def reset_latency_profile(self):
        if self._latency_profiler is None:
            raise RuntimeError("latency profiling is not enabled")
        self._latency_profiler.reset()

    def get_latency_profile(self, synchronize=True):
        if self._latency_profiler is None:
            raise RuntimeError("latency profiling is not enabled")
        return self._latency_profiler.summary(synchronize=synchronize)

    def get_retained_latency_event_count(self):
        if self._latency_profiler is None:
            return 0
        return self._latency_profiler.retained_event_count()

    def vdrs_block_count(self):
        return sum(
            isinstance(module, ValueDrivenSS2D) for module in self.modules()
        )

    def enable_state_propagation_debug(
        self, callback: Callable, module_names: Optional[Sequence[str]] = None
    ):
        """Enable recurrent-only scan-order analysis on every VDRS module."""
        if not callable(callback):
            raise TypeError("state-propagation debug callback must be callable")
        selected = None if module_names is None else set(module_names)
        names = []
        for name, module in self.named_modules():
            if isinstance(module, ValueDrivenSS2D):
                if selected is None or name in selected:
                    module._state_propagation_debug_callback = callback
                    module._state_propagation_debug_name = name
                    names.append(name)
                else:
                    module._state_propagation_debug_callback = None
                    module._state_propagation_debug_name = None
        if selected is not None and selected != set(names):
            missing = sorted(selected - set(names))
            raise ValueError(f"unknown VDRS debug module names: {missing}")
        return names

    def disable_state_propagation_debug(self):
        """Disable analysis callbacks and release any callback-owned state."""
        for module in self.modules():
            if isinstance(module, ValueDrivenSS2D):
                module._state_propagation_debug_callback = None
                module._state_propagation_debug_name = None

    def forward(self, x, dino_features: Optional[Sequence[torch.Tensor]] = None):
        profiler = self._latency_profiler
        model_start = profiler.start() if profiler is not None else None

        raw = x
        if self.training and dino_features is None:
            dino_features = self.extract_dino_features(x)
        if dino_features is None:
            dino_features = (None, None, None, None)
        if len(dino_features) != 4:
            raise ValueError("dino_features must contain four stage features")

        x = self.patchEmbed(x)
        x = self._run(self.encoder_level1, x, dino_features[0])
        skip1 = x
        x = self._run(self.encoder_level2, self.down1_2(x), dino_features[1])
        skip2 = x
        x = self._run(self.encoder_level3, self.down2_3(x), dino_features[2])
        skip3 = x
        x = self._run(self.latent, self.down3_4(x), dino_features[3])

        weight = torch.sigmoid(self.HiLo3(x, skip3))
        x = weight * skip3 + (1 - weight) * self.up4_3(x)
        x = self._run(self.decoder_level3, x, dino_features[2])

        weight = torch.sigmoid(self.HiLo2(x, skip2))
        x = weight * skip2 + (1 - weight) * self.up3_2(x)
        x = self._run(self.decoder_level2, x, dino_features[1])

        weight = torch.sigmoid(self.HiLo1(x, skip1))
        x = weight * skip1 + (1 - weight) * self.up2_1(x)
        x = self._run(self.decoder_level1, x, dino_features[0])
        x = self._run(self.refinement, x, dino_features[0])
        x = self.head(x) + self.layScale(raw)
        if profiler is not None:
            profiler.end("model_total_ms", model_start)
        return x


class HaarDWT2d(nn.Module):
    """Parameter-free Haar transform used by the real RD_3_small path."""

    def __init__(self, pad_mode="reflect"):
        super().__init__()
        self.pad_mode = pad_mode

    def forward(self, x):
        height, width = x.shape[-2:]
        pad_h, pad_w = height % 2, width % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=self.pad_mode)
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]
        bands = (
            (x00 + x01 + x10 + x11) * 0.5,
            (-x00 - x01 + x10 + x11) * 0.5,
            (-x00 + x01 - x10 + x11) * 0.5,
            (x00 - x01 - x10 + x11) * 0.5,
        )
        return torch.cat(bands, dim=1), (height, width)


class HaarIWT2d(nn.Module):
    def forward(self, x, output_size):
        ll, lh, hl, hh = torch.chunk(x, chunks=4, dim=1)
        x00 = (ll - lh - hl + hh) * 0.5
        x01 = (ll - lh + hl - hh) * 0.5
        x10 = (ll + lh - hl - hh) * 0.5
        x11 = (ll + lh + hl + hh) * 0.5
        batch, channels, height, width = x00.shape
        output = x.new_empty(batch, channels, height * 2, width * 2)
        output[:, :, 0::2, 0::2] = x00
        output[:, :, 0::2, 1::2] = x01
        output[:, :, 1::2, 0::2] = x10
        output[:, :, 1::2, 1::2] = x11
        return output[:, :, : output_size[0], : output_size[1]]


class VDSNetS(VDSNet):
    """Wavelet-based lightweight VDSNet-S architecture."""

    def __init__(
        self,
        in_channels=3,
        dim=24,
        hw=128,
        out_channels=3,
        num_blocks=(1, 2, 2, 3),
        kernel_shape=(7, 7, 7, 7),
        num_dynamic_kernels=(2, 2, 3, 4),
        num_offset_groups=(1, 1, 1, 1),
        based_state=4,
        num_refinements=4,
        num_heads=(1, 2, 4),
        value_kernel_size=1,
        dino_model_path=None,
        dino_layers=(3, 3, 4, 5),
    ):
        nn.Module.__init__(self)
        if not (
            len(num_blocks)
            == len(kernel_shape)
            == len(num_dynamic_kernels)
            == len(num_offset_groups)
            == 4
        ):
            raise ValueError(
                "num_blocks, kernel_shape, num_dynamic_kernels and "
                "num_offset_groups must have 4 entries"
            )

        dims = [dim * (2**index) for index in range(4)]
        resolutions = [hw // (2**index) for index in range(4)]
        d_state = based_state * 4

        def blocks(level, count, latent=False):
            return nn.ModuleList(
                [
                    MCMBlock(
                        dims[level],
                        resolutions[level],
                        kernel_shape[level],
                        num_dynamic_kernels[level],
                        num_offset_groups[level],
                        d_state,
                        latent=latent,
                        value_kernel_size=value_kernel_size,
                    )
                    for _ in range(count)
                ]
            )

        wavelet_channels = 4 * in_channels
        self.dwt = HaarDWT2d(pad_mode="reflect")
        self.iwt = HaarIWT2d()
        self.patchEmbed = OverlapPatchEmbed(wavelet_channels, dim)
        self.encoder_level1 = blocks(0, num_blocks[0])
        self.down1_2 = DownSample(dims[0])
        self.encoder_level2 = blocks(1, num_blocks[1])
        self.down2_3 = DownSample(dims[1])
        self.encoder_level3 = blocks(2, num_blocks[2])
        self.down3_4 = DownSample(dims[2])
        self.latent = blocks(3, num_blocks[3], latent=True)

        self.HiLo3 = HiLo_Conv(dims[2], num_heads[2])
        self.up4_3 = UpSample(dims[3])
        self.decoder_level3 = blocks(2, num_blocks[2])
        self.HiLo2 = HiLo_Conv(dims[1], num_heads[1])
        self.up3_2 = UpSample(dims[2])
        self.decoder_level2 = blocks(1, num_blocks[1])
        self.HiLo1 = HiLo_Conv(dims[0], num_heads[0])
        self.up2_1 = UpSample(dims[1])
        self.decoder_level1 = blocks(0, num_blocks[0])
        self.refinement = blocks(0, num_refinements)
        self.layScale = LayScale(wavelet_channels)
        self.head = nn.Conv2d(dim, 4 * out_channels, 1)

        self.dino_layers = tuple(dino_layers)
        self.dino_model = None
        self.dino_processor = None
        self._latency_profiler = None
        if dino_model_path is not None:
            self._load_dino(dino_model_path)

    def forward(self, x, dino_features: Optional[Sequence[torch.Tensor]] = None):
        profiler = self._latency_profiler
        model_start = profiler.start() if profiler is not None else None

        if self.training and dino_features is None:
            dino_features = self.extract_dino_features(x)
        if dino_features is None:
            dino_features = (None, None, None, None)
        if len(dino_features) != 4:
            raise ValueError("dino_features must contain four stage features")

        x, output_size = self.dwt(x)
        raw = x
        x = self.patchEmbed(x)
        x = self._run(self.encoder_level1, x, dino_features[0])
        skip1 = x
        x = self._run(self.encoder_level2, self.down1_2(x), dino_features[1])
        skip2 = x
        x = self._run(self.encoder_level3, self.down2_3(x), dino_features[2])
        skip3 = x
        x = self._run(self.latent, self.down3_4(x), dino_features[3])

        weight = torch.sigmoid(self.HiLo3(x, skip3))
        x = weight * skip3 + (1 - weight) * self.up4_3(x)
        x = self._run(self.decoder_level3, x, dino_features[2])
        weight = torch.sigmoid(self.HiLo2(x, skip2))
        x = weight * skip2 + (1 - weight) * self.up3_2(x)
        x = self._run(self.decoder_level2, x, dino_features[1])
        weight = torch.sigmoid(self.HiLo1(x, skip1))
        x = weight * skip1 + (1 - weight) * self.up2_1(x)
        x = self._run(self.decoder_level1, x, dino_features[0])
        x = self._run(self.refinement, x, dino_features[0])
        x = self.iwt(self.head(x) + self.layScale(raw), output_size)
        if profiler is not None:
            profiler.end("model_total_ms", model_start)
        return x


__all__ = ["VDSNet", "VDSNetS"]
