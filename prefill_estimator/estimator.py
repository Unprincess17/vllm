from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict


@dataclass
class ModelConfig:
    """Transformer model configuration relevant for prefill FLOPs.

    Notes:
    - Uses a simplified FLOPs model per layer for prefill (full-sequence attention):
      FLOPs_per_layer ≈ 12 * n * d_model^2 + 2 * n^2 * d_model
      where n is sequence length per sample.
    - MLP assumes a 4x expansion by default (typical for many LLMs). You can
      override via ffn_multiple.
    - For MoE models, approximate the effective MLP cost using active_experts and
      expert_multiple if desired.
    """

    num_layers: int
    d_model: int
    num_heads: int
    # Feed-forward multiple relative to d_model (e.g., 4 for many LLMs)
    ffn_multiple: float = 4.0
    # If MoE: number of total experts and how many active per token
    num_experts: Optional[int] = None
    active_experts: Optional[int] = None
    expert_multiple: Optional[float] = None  # expert hidden expansion relative to d_model

    def effective_ffn_multiple(self) -> float:
        """Return the effective FFN multiple used in FLOPs for the MLP path.

        For dense models, this is just ffn_multiple.
        For MoE models, we approximate effective multiple as:
          (active_experts / num_experts) * expert_multiple * num_experts
        which simplifies to active_experts * expert_multiple.
        """
        if self.num_experts and self.active_experts and self.expert_multiple:
            return float(self.active_experts) * float(self.expert_multiple)
        return float(self.ffn_multiple)


@dataclass
class HardwareConfig:
    """Hardware performance characteristics.

    - gpu_flops_tflops: Theoretical GPU peak FLOPs (TFLOPS) for the precision used.
    - compute_utilization: Fraction [0,1] of peak achieved (kernel efficiency, overheads).
    - mem_bandwidth_GBps: Sustained device memory bandwidth in GB/s.
    - write_efficiency: Fraction [0,1] of bandwidth achieved for KV writes.
    - bytes_per_elem: Bytes per element used for most compute tensors (e.g., 2 for fp16/bf16).
    - kv_bytes_per_elem: Bytes per element for KV cache writes (often fp16/bf16 -> 2). If None, defaults to bytes_per_elem.
    """

    gpu_flops_tflops: float
    compute_utilization: float = 0.35
    mem_bandwidth_GBps: float = 900.0
    write_efficiency: float = 0.8
    bytes_per_elem: int = 2
    kv_bytes_per_elem: Optional[int] = None

    def effective_flops_per_s(self) -> float:
        return self.gpu_flops_tflops * 1e12 * max(min(self.compute_utilization, 1.0), 0.0)

    def effective_write_Bps(self) -> float:
        return self.mem_bandwidth_GBps * 1e9 * max(min(self.write_efficiency, 1.0), 0.0)

    def kv_bpe(self) -> int:
        return self.kv_bytes_per_elem if self.kv_bytes_per_elem is not None else self.bytes_per_elem


@dataclass
class WorkloadConfig:
    """Describes the prompt prefill workload.

    You can specify either:
      - batch_size and avg_seq_len, or
      - an explicit list of sequence lengths via seq_lens which overrides batch_size/avg_seq_len.
    """

    batch_size: int = 1
    avg_seq_len: int = 1024
    seq_lens: Optional[List[int]] = None

    def expanded_seq_lens(self) -> List[int]:
        if self.seq_lens is not None and len(self.seq_lens) > 0:
            return list(self.seq_lens)
        return [int(self.avg_seq_len)] * int(self.batch_size)


def _flops_prefill_for_sequence_length(n: int, model: ModelConfig) -> float:
    """Approximate forward FLOPs for prefill for a single sequence of length n.

    Uses simplified per-layer formula:
      FLOPs_layer ≈ proj_mlp_term + attn_term
      proj_mlp_term ≈ 4 (QKV) + 1 (O) + 2*ffn + ffn_down ~ 12 d_model^2 per token (typical 4x MLP)
        We scale the MLP component by effective_ffn_multiple / 4 to account for non-4x.
      attn_term ≈ 2 * n^2 * d_model (QK^T + Attn*V)

    Total per layer for sequence n:
      12 * n * d_model^2 * (effective_ffn_multiple / 4) + 2 * n^2 * d_model
    Then multiply by num_layers.
    """
    d = float(model.d_model)
    L = float(model.num_layers)
    eff_ffn_mult = model.effective_ffn_multiple()

    proj_mlp = 12.0 * float(n) * (d ** 2) * (eff_ffn_mult / 4.0)
    attn = 2.0 * (float(n) ** 2) * d
    return L * (proj_mlp + attn)


def _kv_bytes_written_for_sequence_length(n: int, model: ModelConfig, hw: HardwareConfig) -> int:
    """Total KV cache bytes written during prefill for a single sequence of length n.

    For each token and each layer, write K and V of size d_model elements each.
    Total elements written = n * num_layers * 2 * d_model
    Multiply by bytes per element for KV.
    """
    elements = int(n) * int(model.num_layers) * 2 * int(model.d_model)
    return elements * int(hw.kv_bpe())


def estimate_prefill_time(model: ModelConfig, hw: HardwareConfig, workload: WorkloadConfig) -> Dict[str, float]:
    """Estimate prefill performance and return a metrics dictionary.

    Returns keys:
      - total_flops
      - total_kv_bytes
      - compute_time_s
      - memory_time_s
      - prefill_time_s (bottleneck)
      - tokens (total)
      - tokens_per_s_prefill
    """
    seqs = workload.expanded_seq_lens()

    total_flops = 0.0
    total_kv_bytes = 0
    total_tokens = 0

    for n in seqs:
        total_flops += _flops_prefill_for_sequence_length(n, model)
        total_kv_bytes += _kv_bytes_written_for_sequence_length(n, model, hw)
        total_tokens += int(n)

    compute_time_s = total_flops / max(hw.effective_flops_per_s(), 1e-9)
    memory_time_s = total_kv_bytes / max(hw.effective_write_Bps(), 1e-9)
    prefill_time_s = max(compute_time_s, memory_time_s)

    tokens_per_s = float(total_tokens) / prefill_time_s if prefill_time_s > 0 else 0.0

    return {
        "total_flops": total_flops,
        "total_kv_bytes": float(total_kv_bytes),
        "compute_time_s": compute_time_s,
        "memory_time_s": memory_time_s,
        "prefill_time_s": prefill_time_s,
        "tokens": float(total_tokens),
        "tokens_per_s_prefill": tokens_per_s,
    }


def pretty_print_results(results: Dict[str, float]) -> str:
    def human_bytes(x: float) -> str:
        units = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while x >= 1024.0 and i < len(units) - 1:
            x /= 1024.0
            i += 1
        return f"{x:.2f} {units[i]}"

    def human_flops(x: float) -> str:
        units = ["FLOPs", "KFLOPs", "MFLOPs", "GFLOPs", "TFLOPs", "PFLOPs", "EFLOPs"]
        i = 0
        while x >= 1000.0 and i < len(units) - 1:
            x /= 1000.0
            i += 1
        return f"{x:.2f} {units[i]}"

    lines = []
    lines.append(f"Total FLOPs: {human_flops(results['total_flops'])}")
    lines.append(f"KV bytes written: {human_bytes(results['total_kv_bytes'])}")
    lines.append(f"Compute time: {results['compute_time_s']:.4f} s")
    lines.append(f"Memory time: {results['memory_time_s']:.4f} s")
    lines.append(f"Prefill time (bottleneck): {results['prefill_time_s']:.4f} s")
    lines.append(f"Total tokens: {int(results['tokens'])}")
    lines.append(f"Prefill throughput: {results['tokens_per_s_prefill']:.2f} tok/s")
    return "\n".join(lines)

