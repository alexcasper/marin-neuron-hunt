#!/usr/bin/env python3
"""Slice the token-sorted layer-00 parquet at a file-row boundary (duckdb, no sort:
file is already token-sorted, we only keep the tail). Usage: slice_topup.py <cut_rows>"""
import sys
import duckdb

path = "/home/dusk/marin-sae/layer00-tokensorted.parquet"
cut = int(sys.argv[1])
con = duckdb.connect()
con.execute("SET memory_limit='10GB'")
con.execute("SET temp_directory='/home/dusk/marin-sae/tmp'")
# boundary token_idx = last token_idx among the first <cut> rows (streamed, no sort)
row = con.execute(
    f"SELECT max(token_idx) FROM (SELECT token_idx FROM read_parquet('{path}') LIMIT {cut})"
).fetchone()
assert row is not None and row[0] is not None, "empty parquet?"
tcut = row[0]
n_tail = con.execute(
    f"SELECT count(*) FROM read_parquet('{path}') WHERE token_idx >= ?", [tcut]
).fetchone()[0]
print(f"cut token_idx >= {tcut}; tail rows {n_tail} = {n_tail//64} tokens", flush=True)
con.execute(
    f"""COPY (SELECT * FROM read_parquet('{path}') WHERE token_idx >= {tcut})
        TO '/home/dusk/marin-sae/layer00-topup.parquet'
        (FORMAT PARQUET, ROW_GROUP_SIZE 131072)"""
)
print("wrote layer00-topup.parquet", flush=True)
