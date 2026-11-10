#!/usr/bin/env python3
"""Sanity: token_idx contiguity of the token-sorted layer parquet (top-up slice prep)."""
import sys
import pyarrow.parquet as pq

path = sys.argv[1]
pf = pq.ParquetFile(path)
md = pf.metadata
print("rows", md.num_rows, "row_groups", md.num_row_groups, "cols", md.num_columns)
mn, mx = None, None
n_below = 0
CUT = 6_144_000
for rb in pf.iter_batches(batch_size=1 << 22, columns=["token_idx"]):
    t = rb.column("token_idx").to_numpy()
    if mn is None:
        mn = int(t[0])
    mx = int(t[-1])
    n_below += int((t < CUT).sum())
print("min", mn, "max", mx, "tokens(rows//64)", md.num_rows // 64)
print("rows with token_idx <", CUT, "=", n_below, "(expect", CUT * 64, "if contiguous)")
