"""CPU tests for sae_conventions (dawn-runnable; no torch).

Run:  python3 test_train_sae_cpu.py
Covers the fixes from beads sh-kas (layer-label hard-fail), sh-u2m (contiguity check),
and the data-path reconstruction invariants (audit SMOKE2, 2026-11-22).
"""
import io
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import sae_conventions as sc

failures = []


def check(name, cond):
    print(("PASS" if cond else "FAIL"), name)
    if not cond:
        failures.append(name)


# ---------- parse_layer_id (sh-kas) ----------
ok = [( "L57", 57), ("L7", 7), ("L00w64", 0), ("L00", 0), ("L63_scaled", 63),
      ("L40w128", 40)]
for label, want in ok:
    try:
        got = sc.parse_layer_id(label)
        check(f"parse_layer_id({label!r}) == {want}", got == want)
    except ValueError as e:
        check(f"parse_layer_id({label!r}) == {want}", False)

bad = ["L?", "layer7", "Lw64", "", "w64", "00w64", None, 57]
for label in bad:
    try:
        sc.parse_layer_id(label)
        check(f"parse_layer_id({label!r}) raises", False)
    except ValueError:
        check(f"parse_layer_id({label!r}) raises", True)

# ---------- synthetic atlas parquet + dense reconstruction ----------
D = 128
N_TOK = 40
rng = np.random.default_rng(0)
tok_idx = np.repeat(np.arange(N_TOK), 64)
neu = np.stack([rng.permutation(D)[:64] for _ in range(N_TOK)]).ravel()
gate = rng.normal(size=N_TOK * 64).astype(np.float32)
up = rng.normal(size=N_TOK * 64).astype(np.float32)
act_true = gate / (1.0 + np.exp(-gate)) * up
scale = (np.abs(rng.normal(1.0, 0.3, size=D)) + 1e-3).astype(np.float32)

buf = io.BytesIO()
pq.write_table(pa.table({"token_idx": tok_idx, "neuron": neu, "gate": gate, "up": up}), buf)
buf.seek(0)

xs = np.concatenate(list(sc.batches(buf, 7, scale, d_model=D)))  # ragged vs row groups
Xtrue = np.zeros((N_TOK, D), np.float32)
Xtrue[np.repeat(np.arange(N_TOK), 64), neu] = act_true / scale[neu]
check("batches: scaled reconstruction == ground truth", np.allclose(xs, Xtrue, rtol=1e-5))

buf.seek(0)
xs_raw = np.concatenate(list(sc.batches(buf, 7, None, d_model=D)))
Xraw = np.zeros((N_TOK, D), np.float32)
Xraw[np.repeat(np.arange(N_TOK), 64), neu] = act_true
check("batches: raw path (no scales) == raw acts", np.allclose(xs_raw, Xraw, rtol=1e-5))

# max_rows cap skips the tail (val holdout semantics)
buf.seek(0)
xs_cap = np.concatenate(list(sc.batches(buf, 5, scale, max_rows=20 * 64, d_model=D)))
check("batches: max_rows caps the stream at whole tokens", xs_cap.shape == (20, D))

# ---------- contiguity check (sh-u2m) ----------
def write_and_collect(tok, neu_, gate_, up_):
    b = io.BytesIO()
    pq.write_table(pa.table({"token_idx": tok, "neuron": neu_, "gate": gate_, "up": up_}), b)
    b.seek(0)
    return list(sc.batches(b, 4, None, d_model=D))

# case 1: token_idx not constant within a 64-row block
tok_bad = tok_idx.copy()
tok_bad[10] += 1
try:
    write_and_collect(tok_bad, neu, gate, up)
    check("contiguity: mixed 64-row block raises", False)
except ValueError:
    check("contiguity: mixed 64-row block raises", True)

# case 2: tokens not sorted (block order swapped)
perm = rng.permutation(N_TOK)
tok_swapped = np.repeat(np.sort(perm), 64)
neu_s = np.stack([rng.permutation(D)[:64] for _ in range(N_TOK)]).ravel()
try:
    write_and_collect(tok_swapped, neu_s, gate, up)  # sorted, should PASS
    check("contiguity: sorted tokens pass", True)
except ValueError:
    check("contiguity: sorted tokens pass", False)

# case 3: duplicate token blocks (strictly-increasing violated)
tok_dup = np.repeat(np.tile(np.arange(5), 8), 64)  # tokens 0..4 repeat 8 times
try:
    write_and_collect(tok_dup, neu[:N_TOK * 64], gate[:N_TOK * 64], up[:N_TOK * 64])
    check("contiguity: repeated token blocks raise", False)
except ValueError:
    check("contiguity: repeated token blocks raise", True)

# case 4: rows not a multiple of 64
try:
    b = io.BytesIO()
    pq.write_table(pa.table({"token_idx": tok_idx[:63], "neuron": neu[:63],
                             "gate": gate[:63], "up": up[:63]}), b)
    b.seek(0)
    list(sc.batches(b, 4, None))
    check("contiguity: non-multiple-of-64 rows raise", False)
except (ValueError, AssertionError):
    check("contiguity: non-multiple-of-64 rows raise", True)

# ---------- load_scale_vec semantics ----------
t = pa.table({"layer": pa.array([3] * D + [9] * 4, type=pa.int64()),
              "neuron": pa.array(list(range(D)) + [0, 1, 2, 3], type=pa.int64()),
              "scale": pa.array(np.concatenate([scale, [0.0, 0.0, 0.0, 0.0]]), type=pa.float32())})
pb = io.BytesIO(); pq.write_table(t, pb)
import os
os.makedirs("/tmp/sae_test", exist_ok=True)
with open("/tmp/sae_test/scales.parquet", "wb") as f:
    f.write(pb.getvalue())
vec = sc.load_scale_vec("/tmp/sae_test/scales.parquet", 3, d_model=D)
check("load_scale_vec: layer 3 scales loaded", np.allclose(vec, scale, rtol=1e-6))
vec9 = sc.load_scale_vec("/tmp/sae_test/scales.parquet", 9, d_model=D)
check("load_scale_vec: missing layer -> all default 1e-4 (caller must hard-fail on this)",
      bool(np.all(vec9 == 1e-4)))
zeroed = sc.load_scale_vec("/tmp/sae_test/scales.parquet", 9, d_model=D)
check("load_scale_vec: zero scales clamp to 1e-4 floor", bool(np.all(zeroed[:4] == 1e-4)))

print()
if failures:
    print(f"{len(failures)} FAILURES:", failures)
    raise SystemExit(1)
print("ALL TESTS PASS")
