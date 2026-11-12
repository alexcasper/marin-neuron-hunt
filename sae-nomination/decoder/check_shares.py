import glob, pyarrow.parquet as pq
bad = 0
print(f"{'layer':>5} {'rows':>10} {'pos_sum':>8} {'neg_sum':>8}")
for fp in sorted(glob.glob("/data/scratch/apx186/marin32b/tokenlayer/layer=*/class_summary.parquet")):
    LL = fp.split("layer=")[1][:2]
    t = pq.read_table(fp).to_pydict()
    ps = sum(t["pos_mass_share"]); ns = sum(t["neg_abs_mass_share"])
    flag = "  <-- DEVIATES" if abs(ps - 1.0) > 0.01 or abs(ns - 1.0) > 0.01 else ""
    if flag: bad += 1
    print(f"{LL:>5} {len(t['class']):>10} {ps:>8.4f} {ns:>8.4f}{flag}")
print(f"BAD={bad}/64")
