from __future__ import annotations

import argparse
from .model import ModelConfig, HardwareConfig, estimate_prefill_time, format_estimate


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prefill latency estimator")
    # Model args
    p.add_argument("--layers", type=int, required=True, help="Number of transformer layers")
    p.add_argument("--hidden", type=int, required=True, help="Hidden size d")
    p.add_argument("--heads", type=int, required=True, help="Attention heads h")
    p.add_argument("--seq", type=int, required=True, help="Prompt token length S")
    p.add_argument("--batch", type=int, default=1, help="Batch size B")
    p.add_argument("--dtype-bytes", type=int, default=2, help="Bytes per element (2=fp16/bf16)")
    p.add_argument("--ffn-ratio", type=float, default=4.0, help="FFN expansion ratio r")
    p.add_argument("--flash-attn", action="store_true", help="Assume FlashAttention-like kernels")
    p.add_argument("--no-flash-attn", dest="flash_attn", action="store_false")
    p.set_defaults(flash_attn=True)
    p.add_argument("--attn-mem-factor", type=float, default=1.0, help="Attention memory factor")
    p.add_argument("--act-bytes-factor", type=float, default=3.0, help="Activation bytes factor")
    p.add_argument("--kv-write-mul", type=float, default=1.0, help="KV write multiplier")
    p.add_argument("--weight-read-mul", type=float, default=1.0, help="Weight read multiplier")

    # Hardware args
    p.add_argument("--tflops", type=float, required=True, help="Peak TFLOPs for dtype")
    p.add_argument("--mem-gbps", type=float, required=True, help="HBM bandwidth GB/s")
    p.add_argument("--util-compute", type=float, default=0.6, help="Compute utilization 0-1")
    p.add_argument("--util-bw", type=float, default=0.8, help="Bandwidth utilization 0-1")
    p.add_argument("--overhead-us", type=float, default=20.0, help="Per-layer overhead in us")

    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)

    model = ModelConfig(
        num_layers=args.layers,
        hidden_size=args.hidden,
        num_attention_heads=args.heads,
        sequence_length=args.seq,
        batch_size=args.batch,
        dtype_bytes=args.dtype_bytes,
        expansion_ratio=args.ffn_ratio,
        use_flash_attention=args.flash_attn,
        attention_memory_factor=args.attn_mem_factor,
        activation_bytes_factor=args.act_bytes_factor,
        kv_cache_write_multiplier=args.kv_write_mul,
        weight_read_multiplier=args.weight_read_mul,
    )

    hw = HardwareConfig(
        peak_tflops=args.tflops,
        memory_bandwidth_gbps=args.mem_gbps,
        compute_utilization=args.util_compute,
        bandwidth_utilization=args.util_bw,
        kernel_overhead_us=args.overhead_us,
    )

    result = estimate_prefill_time(model, hw)
    print(format_estimate(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

