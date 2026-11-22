"""Torch-free conventions + data-path helpers for the marin-32b atlas SAE trainer.

Extracted from train_sae.py (sh-kas, 2026-11-22) so the input-convention logic is
importable and testable on CPU-only hosts (dawn has no torch).

BINDING RULES (METRICS.md):
- Every run MUST resolve a per-neuron scale vector; silent raw-acts fallback is the
  bug behind the 64k-sweep incident (2026-11-07). parse_layer_id raises instead of
  guessing; callers must hard-fail on error.
- The layer id is the digits immediately after the leading "L" of --layer-label
  ("L57", "L00w64", "L63_scaled" all parse; garbage raises).
"""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import re

D_MODEL_MLP = 27648  # marin-32b SwiGLU neurons per layer (act = silu(gate)*up)
ROWS_PER_TOKEN = 64  # atlas sparse tier: exactly 64 rows per token


def parse_layer_id(label):
    """Layer index from a --layer-label. Raises ValueError instead of guessing.

    "L57" -> 57, "L00w64" -> 0, "L63_scaled" -> 63, "L7" -> 7.
    Anything without digits immediately after the leading L raises.
    """
    if not isinstance(label, str):
        raise ValueError(f"layer label must be a string, got {label!r}")
    m = re.match(r"^L(\d+)", label)
    if not m:
        raise ValueError(
            f"cannot resolve layer from --layer-label {label!r}: expected "
            "digits immediately after the leading 'L' (e.g. L57, L00w64)"
        )
    return int(m.group(1))


def load_scale_vec(scale_parquet, layer_id, d_model=D_MODEL_MLP):
    """Per-neuron normalization scale for one layer; missing neurons get 1e-4."""
    t = pq.read_table(scale_parquet)
    lay = t.column("layer").to_numpy()
    neu = t.column("neuron").to_numpy()
    scl = t.column("scale").to_numpy()
    vec = np.full(d_model, 1e-4, dtype=np.float32)
    sel = lay == layer_id
    vec[neu[sel]] = np.maximum(scl[sel].astype(np.float32), 1e-4)
    return vec


def _dense_batch(rb, scales_by_neuron, d_model):
    """One RecordBatch (n % 64 == 0 rows, token-sorted) -> [B, d_model] dense acts."""
    n = rb.num_rows
    assert n % ROWS_PER_TOKEN == 0
    tok = rb.column("token_idx").to_numpy()
    neu = rb.column("neuron").to_numpy()
    gate = rb.column("gate").to_numpy().astype(np.float32)
    up = rb.column("up").to_numpy().astype(np.float32)
    B = n // ROWS_PER_TOKEN
    # Contiguity invariant (sh-u2m): exactly ROWS_PER_TOKEN rows per token, token_idx
    # constant within each block and strictly increasing across blocks. A violation
    # means the parquet is not the token-sorted atlas format -> silent training-data
    # corruption if we proceeded.
    blocks = tok.reshape(B, ROWS_PER_TOKEN)
    if not (np.all(blocks == blocks[:, :1]) and np.all(np.diff(blocks[:, 0]) > 0)):
        raise ValueError(
            "sparse parquet violates the token-sorted 64-rows-per-token invariant "
            "(token_idx must be constant within each 64-row block and strictly "
            "increase across blocks) — refusing to build training batches"
        )
    x = np.zeros((B, d_model), dtype=np.float32)
    act = gate / (1.0 + np.exp(-gate)) * up
    if scales_by_neuron is not None:
        act = act / scales_by_neuron[neu]
    flat_b = np.repeat(np.arange(B, dtype=np.int64), ROWS_PER_TOKEN)
    x[flat_b, neu] = act
    return x


def batches(parquet_path, batch_tokens, scales_by_neuron=None, max_rows=None, d_model=D_MODEL_MLP):
    """Yield dense fp32 batches [B, d_model] (numpy arrays; caller wraps in torch).

    max_rows caps the training stream (rows beyond it, e.g. a val holdout, are skipped).
    Raises ValueError on contiguity violations (see _dense_batch).
    """
    pf = pq.ParquetFile(parquet_path)
    rows_per_batch = batch_tokens * ROWS_PER_TOKEN
    emitted = 0
    for rb in pf.iter_batches(batch_size=rows_per_batch, columns=["token_idx", "neuron", "gate", "up"]):
        n = rb.num_rows
        if max_rows is not None and emitted + n > max_rows:
            rb = rb.slice(0, max_rows - emitted)
            n = rb.num_rows
        if n == 0:
            return
        yield _dense_batch(rb, scales_by_neuron, d_model)
        emitted += n


def _tail_batches(pf, start_row, batch_tokens):
    """Yield pyarrow RecordBatches covering rows [start_row, end), aligned to 64."""
    md = pf.metadata
    spans = []
    pos = 0
    for g in range(md.num_row_groups):
        spans.append((pos, md.row_group(g).num_rows))
        pos += md.row_group(g).num_rows
    buf = None
    for gi, (gstart, gsize) in enumerate(spans):
        gend = gstart + gsize
        if gend <= start_row:
            continue
        take_start = max(start_row, gstart)
        tbl = pf.read_row_group(gi, columns=["token_idx", "neuron", "gate", "up"])
        tbl = tbl.slice(take_start - gstart, gend - take_start)
        if buf is not None:
            tbl = pa.concat_tables([buf, tbl])
        nb = tbl.num_rows // ROWS_PER_TOKEN * ROWS_PER_TOKEN
        if nb == 0:
            buf = tbl
            continue
        for i in range(0, nb, batch_tokens * ROWS_PER_TOKEN):
            yield tbl.slice(i, min(batch_tokens * ROWS_PER_TOKEN, nb - i)).combine_chunks()
        buf = tbl.slice(nb) if tbl.num_rows > nb else None


def val_batches(parquet_path, val_rows, batch_tokens, scales_by_neuron=None, cap_tokens=None,
                d_model=D_MODEL_MLP):
    """Validation batches from the trailing val_rows of a token-sorted parquet."""
    pf = pq.ParquetFile(parquet_path)
    total = pf.metadata.num_rows
    start = max(0, total - val_rows)
    emitted = 0
    for rb in _tail_batches(pf, start, batch_tokens):
        if rb.num_rows % ROWS_PER_TOKEN != 0:
            continue
        x = _dense_batch(rb, scales_by_neuron, d_model)
        yield x
        emitted += x.shape[0]
        if cap_tokens and emitted >= cap_tokens:
            return
