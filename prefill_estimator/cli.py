from __future__ import annotations

import argparse
import json
from typing import Optional, List

from .estimator import (
    ModelConfig,
    HardwareConfig,
    WorkloadConfig,
    estimate_prefill_time,
    pretty_print_results,
)


def presets_model(name: str) -> ModelConfig:
    key = name.lower()
    if key in {"llama2-7b", "llama-2-7b", "llama2_7b"}:
        return ModelConfig(num_layers=32, d_model=4096, num_heads=32, ffn_multiple=4.0)
    if key in {"llama2-13b", "llama-2-13b", "llama2_13b"}:
        return ModelConfig(num_layers=40, d_model=5120, num_heads=40, ffn_multiple=4.0)
    if key in {"llama2-70b", "llama-2-70b", "llama2_70b"}:
        return ModelConfig(num_layers=80, d_model=8192, num_heads=64, ffn_multiple=4.0)
    if key in {"mixtral-8x7b", "mixtral", "mixtral_8x7b"}:
        # Approximate MoE config: 32 layers, 8 experts, top-2 routing, ~4x per expert
        return ModelConfig(
            num_layers=32,
            d_model=4096,
            num_heads=32,
            num_experts=8,
            active_experts=2,
            expert_multiple=4.0,
        )
    raise ValueError(f"Unknown model preset: {name}")


def presets_hardware(name: str) -> HardwareConfig:
    key = name.lower()
    if key in {"a100-80g", "a100", "nvidia-a100"}:
        # Approximate FP16/BF16 theoretical 312 TFLOPS, ~2 TB/s HBM2e
        return HardwareConfig(gpu_flops_tflops=312.0, compute_utilization=0.35, mem_bandwidth_GBps=2039.0, write_efficiency=0.8, bytes_per_elem=2)
    if key in {"h100-sxm", "h100", "nvidia-h100"}:
        # Provide conservative placeholder; users should override --gpu-tflops based on their SKU
        return HardwareConfig(gpu_flops_tflops=1000.0, compute_utilization=0.35, mem_bandwidth_GBps=3350.0, write_efficiency=0.8, bytes_per_elem=2)
    return HardwareConfig(gpu_flops_tflops=100.0)


def parse_seq_lens(arg: Optional[str]) -> Optional[List[int]]:
    if not arg:
        return None
    return [int(x.strip()) for x in arg.split(",") if x.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Estimate LLM prompt prefill time using a FLOPs + KV-write model.")

    m = p.add_argument_group("Model")
    m.add_argument("--model-preset", type=str, default=None, help="Model preset name (e.g., llama2-7b, llama2-13b, llama2-70b, mixtral-8x7b)")
    m.add_argument("--layers", type=int, default=None, help="Number of transformer layers")
    m.add_argument("--d-model", type=int, default=None, help="Model hidden size (d_model)")
    m.add_argument("--heads", type=int, default=None, help="Number of attention heads")
    m.add_argument("--ffn-multiple", type=float, default=None, help="FFN expansion multiple (default 4.0)")
    m.add_argument("--num-experts", type=int, default=None, help="Total experts (MoE)")
    m.add_argument("--active-experts", type=int, default=None, help="Active experts per token (MoE)")
    m.add_argument("--expert-multiple", type=float, default=None, help="Expert FFN expansion multiple relative to d_model (MoE)")

    h = p.add_argument_group("Hardware")
    h.add_argument("--hw-preset", type=str, default=None, help="Hardware preset (e.g., a100-80g, h100-sxm)")
    h.add_argument("--gpu-tflops", type=float, default=None, help="GPU theoretical TFLOPS for precision used (e.g., FP16)")
    h.add_argument("--util", type=float, default=None, help="Compute utilization fraction [0,1] (default 0.35)")
    h.add_argument("--bandwidth", type=float, default=None, help="Device memory bandwidth GB/s (sustained)")
    h.add_argument("--write-eff", type=float, default=None, help="Write efficiency for KV cache [0,1] (default 0.8)")
    h.add_argument("--bytes-per-elem", type=int, default=None, help="Bytes per tensor element for compute (2 for fp16/bf16)")
    h.add_argument("--kv-bytes-per-elem", type=int, default=None, help="Bytes per KV element (defaults to bytes-per-elem)")

    w = p.add_argument_group("Workload")
    w.add_argument("--batch-size", type=int, default=1, help="Batch size")
    w.add_argument("--avg-seq-len", type=int, default=1024, help="Average prompt token length")
    w.add_argument("--seq-lens", type=str, default=None, help="Comma-separated list of per-sequence lengths, overrides avg/batch")

    o = p.add_argument_group("Output")
    o.add_argument("--json", action="store_true", help="Emit JSON instead of text")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    p = build_arg_parser()
    args = p.parse_args(argv)

    # Model config
    if args.model_preset:
        model = presets_model(args.model_preset)
    else:
        if args.layers is None or args.d_model is None or args.heads is None:
            p.error("Either --model-preset or all of --layers/--d-model/--heads must be provided")
        model = ModelConfig(
            num_layers=args.layers,
            d_model=args.d_model,
            num_heads=args.heads,
            ffn_multiple=args.ffn_multiple if args.ffn_multiple is not None else 4.0,
            num_experts=args.num_experts,
            active_experts=args.active_experts,
            expert_multiple=args.expert_multiple,
        )

    # Hardware config
    if args.hw_preset:
        hw = presets_hardware(args.hw_preset)
    else:
        if args.gpu_tflops is None:
            p.error("Either --hw-preset or --gpu-tflops must be provided")
        hw = HardwareConfig(gpu_flops_tflops=args.gpu_tflops)

    if args.util is not None:
        hw.compute_utilization = args.util
    if args.bandwidth is not None:
        hw.mem_bandwidth_GBps = args.bandwidth
    if args.write_eff is not None:
        hw.write_efficiency = args.write_eff
    if args.bytes_per_elem is not None:
        hw.bytes_per_elem = args.bytes_per_elem
    if args.kv_bytes_per_elem is not None:
        hw.kv_bytes_per_elem = args.kv_bytes_per_elem

    # Workload
    seq_lens = parse_seq_lens(args.seq_lens)
    workload = WorkloadConfig(batch_size=args.batch_size, avg_seq_len=args.avg_seq_len, seq_lens=seq_lens)

    results = estimate_prefill_time(model, hw, workload)
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(pretty_print_results(results))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

