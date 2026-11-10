# Dusk vs Apocrita — full-scale SAE sizing (measured, 2026-10-27)

For alex's compute-location call. All dusk numbers measured on this host; H200/A100
numbers are FLOP-model estimates from published peak specs with an assumed 40% MFU.

## Measured on dusk (Radeon 890M, gfx1150, ROCm 2.10)

| config | throughput (measured) | wall for 1 epoch of one layer (9.995M tok) |
|---|---|---|
| 32k latents, k=32, bf16, batch 1024 | 318.4 tok/s (6.14M tok / 5.36 h, pilot) | 8.7 h |
| 64k latents, k=64, bf16, batch 1024 | ~160 tok/s (sizing smoke, RUNPLAN) | 17.4 h |

- 64k bf16 torch alloc 27.4 GiB of 48G GTT — fits only on a quiet host (no bulk
  transfer; page-cache growth caused a 10× reclaim-crawl once).
- 64k fp32 = 54 GiB = OOM. Never on dusk.
- Loader is the co-bottleneck (CPU batch prep + prefetch), so measured tok/s is
  the honest number — pure-FLOP estimates understate dusk wall time.

## FLOP model (for cross-hardware scaling)

64k rung: fwd ≈ 2·(2·d_in·n_latents) = 7.25 GFLOP/token; fwd+bwd ≈ 21.7 GFLOP/token.
Full program (64 layers × 1 epoch × 64k): ≈ 13.9 EFLOP.

| host | sustained (assumed) | 1 layer, 64k, 1 epoch | 64k × 200k steps (1 layer) | 64-layer × 1 epoch |
|---|---|---|---|---|
| dusk 890M | 5.13 TF/s fp16 (measured) | 17.4 h (measured rate) | ~12 days | ~46 days |
| Apocrita H200 | 989 TF/s bf16 peak × 40% MFU | ~0.15 h GPU | ~3.1 h GPU | ~0.4 day GPU |
| A100-class | 312 TF/s bf16 peak × 40% MFU | ~0.5 h GPU | ~9 h GPU (spans 2 jobs) | ~1.3 day GPU |

Caveats on the Apocrita column: assumes the loader keeps up (per-layer parquet is
3.8G — node-local staging before the job, absolute paths, watch inodes) and 8h
Slurm cap (a 200k-step rung at ~3 h leaves 2.5× headroom; --requeue covers preemption).
H200 wall will be loader/IO-bound at these tiny compute times, not GPU-bound —
budget ~1 h/layer realistically.

## Recommendation (decision-class — alex decides)

- **dusk**: right size for layer pilots, censuses, atlas analysis. Free and
  always-on, but the 64-layer 64k program is ~6–7 weeks serial, and 64k runs
  require a quiet host.
- **Apocrita H200**: the 64-layer × 64k program is a one-day sbatch array; a
  200k-step rung fits a single 8h job with margin. This was the original sh-hr9
  target and the numbers say it is the only sane venue for full scale.
- Hybrid that costs nothing: keep pilot/census work on dusk, move the 64k rung
  and the 64-layer program to Apocrita when alex gives the word (via env_apoc
  job specs, not hand-rolled ssh).
