import pyarrow.parquet as pq, json
t = pq.read_table("/data/scratch/apx186/marin32b/tokenlayer/layer=07/class_summary.parquet")
d = t.to_pydict()
print("cols:", t.column_names, "rows:", t.num_rows)
ps = d["pos_mass_share"]; ns = d["neg_abs_mass_share"]
print("pos shares sum:", round(sum(ps), 6), "| neg shares sum:", round(sum(ns), 6))
for r in range(t.num_rows):
    print(f"  {d['class'][r]:14s} n={d['n_tokens'][r]:>8} pos_share={ps[r]:.4f}")
