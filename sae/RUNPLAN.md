# SAE run plan — dusk pilot (sh-hr9), 2026-10-26

## What trains on what

- **Data**: `pilot/fineweb-10m-sparse/sparse/layer=LL/` — per token × layer the
  top-64 MLP neurons by |act|/scale, columns `(token_idx, neuron, gate, up)`,
  fp16, 64 rows/token, 9,995,501 tokens/layer. Reconstructed input vector:
  `x[neuron] = silu(gate)·up`, normalized by the atlas `neuron_scale`
  (d_in = 27,648). This is the breadth tier only — fineweb-10m-sparse has no
  dense resid/deep vectors, so this SAE models the *sparse-tier reconstruction*
  of the MLP activation, not the raw activation.
- **Model**: eai-sparsify 1.3.3 `Sae`, TopK (activation="topk"),
  normalize_decoder=True, AdamW(3e-4, β=(0.9,0.95)), fvu loss.
- **Hardware**: dusk Radeon 890M (gfx1150 via HSA_OVERRIDE 11.5.1), ROCm torch
  2.10.0+rocm7.13, 5.05 TFLOP/s fp16 sustained, 48 GiB visible GTT (shared with
  46 G system RAM — qwen :8091 left running per inf_amd note).

## Sizing (measured, not guessed)

| config | params | torch alloc | fits? |
|---|---|---|---|
| 64k latents (n_latents=65536, k=64), bf16 | 3,624M | 27.4 GiB | only when rsync idle; crawled once page cache filled |
| 32k latents (n_latents=32768, k=32), bf16 | ~1,812M | ~14 GiB | comfortable; pilot size |
| 64k fp32 (original plan) | 3,624M | 54 GiB | OOM on 48G — H200/Apocrita only |

- Throughput measured: ~160 tok/s at batch 1024/64k (CPU-side batch prep was
  the gap vs the ~460 tok/s GPU ceiling; prefetch added).

## Launch (long pilot run, after rsync completes)

```sh
ssh dusk
cd ~/marin-sae
export LD_LIBRARY_PATH=/nix/store/78x9i5x1wpqw4kq0h39b8f35abcv156h-zlib-1.3.2/lib
export CC=/nix/store/3d1c302vw7kc8a5vknhmn34c0pd7zm6m-gcc-wrapper-15.3.0/bin/gcc
~/torch-rocm-env ~/marin-sae/train_sae.py \
  --layer-parquet ~/marin-sae/layer00-tokensorted.parquet \
  --out-dir ~/marin-sae/runs/L00-pilot --layer-label L00 \
  --n-latents 32768 --k 32 --batch-tokens 1024 --steps 6000 \
  --log-every 50 --save-every 1000
```

~6.1M tokens (0.6 epoch of layer 00). Smoke evidence: fvu 0.980 → 0.963 over
the first 20 steps; GPU busy 100%.

## NixOS gotchas on dusk (solved, do not rediscover)

- `duckdb`/`numpy` C-libs need libstdc++ AND libz: always launch via
  `~/torch-rocm-env`, plus `LD_LIBRARY_PATH` must include
  `/nix/store/78x9i5x1wpqw4kq0h39b8f35abcv156h-zlib-1.3.2/lib` (the wrapper's
  zlib glob can resolve to a `.drv`).
- Triton JIT needs a C compiler: `CC=/nix/store/3d1c302vw7kc8a5vknhmn34c0pd7zm6m-gcc-wrapper-15.3.0/bin/gcc`.
- `Sae.forward` returns `ForwardOutput(sae_out, latent_acts, latent_indices)`.
- pyarrow in this venv: use `.to_numpy()`, not `.to_pydict()`.

## Dusk vs Apocrita (for the 64-layer program)

- **dusk**: free, always-on, no queue; ~5 TFLOP/s → 64k-latent × 200k-step
  program ≈ weeks. Right size for layer pilots + atlas analysis.
- **Apocrita H200 (via env_apoc wW:p7, sbatch array, 8h cap)**: the original
  sh-hr9 target; a 64k/200k-step run is an 8h-class job there, 64 layers =
  one sbatch array. alex weighs dusk vs Apocrita — decision-class, not ours.

## Transfer state

- rsync dawn→dusk `fineweb-10m-sparse`: **COMPLETE 2026-10-27** — 253G, 82 files,
  both sides identical (size-only dry-run diff empty). The "384G" figure was
  `data/pilot/` total on dawn (fineweb-10m-sparse 253G + stage-a-100k 31G +
  fineweb-1m 101G); only fineweb-10m-sparse was in scope for dusk.
- Layer-00 pilot consumed the first 393,216,000 file rows (6,144,000 tokens) of
  `layer00-tokensorted.parquet`; top-up slice `layer00-topup.parquet` (1.3G,
  3,851,502 tokens, cut at token_idx >= 6147001) covers the unseen remainder —
  pilot + top-up = exactly one epoch of layer 00.
- Globus SDK path: blocked on one human consent click; recipe in
  `docs/globus-consent-recipe.md` (do NOT retry unattended).

## Queue state (2026-10-27)

- Census + top-up chain: `runs/L00-extend` (warm-start from sae_L00_final,
  32k/bf16, batch 768 or 1024 depending on free RAM vs the llama servers),
  log `~/marin-sae/topup_chain.log`; census JSONs `census_L00_final.json`,
  `census_L00_extend.json` on dusk.
- 64k rung: STAGED as `~/marin-sae/run_L00_64k.sh` (refuses to run without
  ALEX_GO=1). 1 epoch = 9762 steps ≈ 17.4 h at the measured ~160 tok/s.
- Sizing for the compute-location call: `sae/SIZING-dusk-vs-apocrita.md`.
