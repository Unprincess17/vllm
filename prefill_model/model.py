"""
Prefill latency estimator using a simple roofline model.

This module estimates the time to prefill (encode) a prompt of length S tokens
through a decoder-only Transformer, accounting for both compute and memory.

Model assumptions (tunable via parameters):
- FLOPs per layer per sequence (batch B, seq S, hidden d):
    MACs ≈ 12 * B * S * d^2   [QKV+O + MLP(4d)]
          + 2 * B * S^2 * d   [attention QK^T and P*V]
  FLOPs are 2 × MACs.
- KV cache write traffic: 2 * B * S * d elements per layer (K and V)
- Weight read traffic per layer: ~12 * d^2 elements (one read per sequence)
- Activation and miscellaneous traffic are approximated with coefficients.

These are approximate and intended for comparative reasoning and quick sizing,
not hardware-vendor-accurate timing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ModelConfig:
    """Transformer model parameters for prefill estimation.

    Attributes:
        num_layers: Number of Transformer layers L.
        hidden_size: Model hidden size d.
        num_attention_heads: Number of attention heads h (used for sanity, not required by formula).
        sequence_length: Prompt tokens per sequence S.
        batch_size: Number of sequences processed concurrently B.
        dtype_bytes: Bytes per scalar (e.g., 2 for fp16/bf16, 1 for fp8).
        expansion_ratio: Feedforward expansion ratio (default 4.0).
        use_flash_attention: If True, model reduced memory traffic for attention.
        attention_memory_factor: Tunable factor for attention memory traffic (dimensionless).
        activation_bytes_factor: Multiplier for estimated activation read/write bytes per layer.
        kv_cache_write_multiplier: Multiplier for KV cache write traffic (dimensionless).
        weight_read_multiplier: Multiplier for weight read traffic (dimensionless).
    """

    num_layers: int
    hidden_size: int
    num_attention_heads: int
    sequence_length: int
    batch_size: int = 1
    dtype_bytes: int = 2
    expansion_ratio: float = 4.0
    use_flash_attention: bool = True
    attention_memory_factor: float = 1.0
    activation_bytes_factor: float = 3.0
    kv_cache_write_multiplier: float = 1.0
    weight_read_multiplier: float = 1.0


@dataclass
class HardwareConfig:
    """Hardware parameters for roofline estimation.

    Attributes:
        peak_tflops: Peak math throughput in TFLOPs for the chosen dtype (tensor cores).
        memory_bandwidth_gbps: Sustained HBM bandwidth in GB/s.
        compute_utilization: Realized fraction of peak FLOPs (0-1).
        bandwidth_utilization: Realized fraction of peak bandwidth (0-1).
        kernel_overhead_us: Fixed per-layer kernel overhead (microseconds), amortized per sequence.
    """

    peak_tflops: float
    memory_bandwidth_gbps: float
    compute_utilization: float = 0.6
    bandwidth_utilization: float = 0.8
    kernel_overhead_us: float = 20.0


@dataclass
class EstimateResult:
    total_flops: float
    total_bytes: float
    compute_time_s: float
    memory_time_s: float
    overhead_time_s: float
    predicted_time_s: float
    breakdown: Dict[str, float]


def _compute_flops(model: ModelConfig) -> float:
    """Estimate forward FLOPs for prefill.

    Returns FLOPs, not MACs.
    """
    L = float(model.num_layers)
    d = float(model.hidden_size)
    S = float(model.sequence_length)
    B = float(model.batch_size)
    r = float(model.expansion_ratio)

    # MACs per layer: Projections(=4 S d^2) + MLP(=2 * S * d * r d = 2 r S d^2) + Attention(=2 S^2 d)
    # With r=4, MLP MACs = 8 S d^2. Generally, 2 r S d^2.
    macs_dense = (4.0 + 2.0 * r) * B * S * d * d
    macs_attn = 2.0 * B * S * S * d
    macs_total = L * (macs_dense + macs_attn)
    flops_total = 2.0 * macs_total
    return flops_total


def _estimate_memory_bytes(model: ModelConfig) -> Dict[str, float]:
    """Estimate dominant memory traffic in bytes for prefill.

    This is a heuristic. Factors allow tuning to match measurements.
    """
    L = float(model.num_layers)
    d = float(model.hidden_size)
    S = float(model.sequence_length)
    B = float(model.batch_size)
    bytes_per_elem = float(model.dtype_bytes)
    r = float(model.expansion_ratio)

    # Weights read once per layer per sequence (assumes good tiling over S and B)
    # Elements per layer: projections (4 d^2) + MLP (2 r d^2)
    weight_elems_per_layer = (4.0 + 2.0 * r) * d * d
    bytes_weights = (
        L * weight_elems_per_layer * bytes_per_elem * model.weight_read_multiplier
    )

    # KV cache writes: per layer, per token: 2 * d elements
    bytes_kv_write = (
        L * B * S * (2.0 * d) * bytes_per_elem * model.kv_cache_write_multiplier
    )

    # Attention memory traffic: heuristic
    if model.use_flash_attention:
        # FlashAttention reduces HBM traffic to O(B S d) with a small constant
        bytes_attn = (
            L
            * B
            * S
            * d
            * bytes_per_elem
            * 2.0
            * model.attention_memory_factor
        )
    else:
        # Naive attention can incur O(B S^2 d) traffic; constant kept small due to tiling
        bytes_attn = (
            L
            * B
            * (S * S)
            * d
            * bytes_per_elem
            * 0.125
            * model.attention_memory_factor
        )

    # Activation and miscellaneous reads/writes per layer: O(B S d)
    bytes_activation = (
        L * B * S * d * bytes_per_elem * model.activation_bytes_factor
    )

    return {
        "weights": bytes_weights,
        "kv_write": bytes_kv_write,
        "attention": bytes_attn,
        "activations": bytes_activation,
    }


def estimate_prefill_time(
    model: ModelConfig,
    hardware: HardwareConfig,
) -> EstimateResult:
    """Estimate prefill latency using a roofline model.

    Returns an EstimateResult with breakdowns and the predicted time in seconds.
    """
    flops_total = _compute_flops(model)
    mem_breakdown = _estimate_memory_bytes(model)
    bytes_total = sum(mem_breakdown.values())

    # Compute-side time
    peak_flops_per_s = hardware.peak_tflops * 1e12 * max(hardware.compute_utilization, 1e-6)
    compute_time_s = flops_total / peak_flops_per_s

    # Memory-side time
    peak_bytes_per_s = (
        hardware.memory_bandwidth_gbps * 1e9 * max(hardware.bandwidth_utilization, 1e-6)
    )
    memory_time_s = bytes_total / peak_bytes_per_s

    # Overhead: kernel launches per layer (amortized per sequence)
    overhead_time_s = (hardware.kernel_overhead_us * 1e-6) * model.num_layers

    predicted_time_s = max(compute_time_s, memory_time_s) + overhead_time_s

    breakdown = {
        "flops_total": flops_total,
        "bytes_total": bytes_total,
        "bytes_weights": mem_breakdown["weights"],
        "bytes_kv_write": mem_breakdown["kv_write"],
        "bytes_attention": mem_breakdown["attention"],
        "bytes_activations": mem_breakdown["activations"],
        "compute_time_s": compute_time_s,
        "memory_time_s": memory_time_s,
        "overhead_time_s": overhead_time_s,
    }

    return EstimateResult(
        total_flops=flops_total,
        total_bytes=bytes_total,
        compute_time_s=compute_time_s,
        memory_time_s=memory_time_s,
        overhead_time_s=overhead_time_s,
        predicted_time_s=predicted_time_s,
        breakdown=breakdown,
    )


def format_estimate(result: EstimateResult) -> str:
    """Return a human-readable summary string for the estimate."""
    gb = result.total_bytes / 1e9
    tflops = result.total_flops / 1e12
    lines = []
    lines.append(f"Predicted prefill time: {result.predicted_time_s * 1e3:.2f} ms")
    lines.append("Breakdown:")
    lines.append(f"  Total FLOPs: {tflops:.3f} TFLOPs")
    lines.append(f"  Total bytes: {gb:.3f} GB")
    lines.append(f"  Compute time: {result.compute_time_s * 1e3:.2f} ms")
    lines.append(f"  Memory time:  {result.memory_time_s * 1e3:.2f} ms")
    lines.append(f"  Overheads:    {result.overhead_time_s * 1e3:.2f} ms")
    lines.append("  Memory bytes:")
    lines.append(
        f"    weights:     {result.breakdown['bytes_weights'] / 1e9:.3f} GB"
    )
    lines.append(
        f"    kv_write:    {result.breakdown['bytes_kv_write'] / 1e9:.3f} GB"
    )
    lines.append(
        f"    attention:   {result.breakdown['bytes_attention'] / 1e9:.3f} GB"
    )
    lines.append(
        f"    activations: {result.breakdown['bytes_activations'] / 1e9:.3f} GB"
    )
    return "\n".join(lines)

