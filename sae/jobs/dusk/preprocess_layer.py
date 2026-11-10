#!/usr/bin/env python3
"""Re-sort one atlas sparse/layer=LL parquet by token_idx for streaming SAE batches.

Input is sorted by (neuron, token_idx); SAE training wants contiguous tokens.
Output: one parquet, sorted by token_idx, row groups sized for batch reads.
Usage: preprocess_layer.py <in.parquet> <out.parquet> [tmp_dir]
"""
import sys
import duckdb

src, dst = sys.argv[1], sys.argv[2]
tmp = sys.argv[3] if len(sys.argv) > 3 else "/home/dusk/tmp"
con = duckdb.connect()
con.execute(f"SET memory_limit='10GB'")
con.execute(f"SET temp_directory='{tmp}'")
con.execute(
    f"""COPY (
        SELECT token_idx, neuron, gate, up
        FROM read_parquet('{src}')
        ORDER BY token_idx
    ) TO '{dst}' (FORMAT PARQUET, ROW_GROUP_SIZE 131072)"""
)
n = con.execute(f"SELECT count(*) FROM read_parquet('{dst}')").fetchone()[0]
print(f"rows={n} tokens={n//64} (rows%64={n%64})")
assert n % 64 == 0, "expected exactly 64 rows per token"
