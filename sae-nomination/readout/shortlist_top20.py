import json, pyarrow.parquet as pq
PQ = "/data/scratch/apx186/marin32b/sae-64k/layer=57/nominate_enc/attribution_enc.parquet"
PS = "/data/scratch/apx186/marin32b/atlas/pilot/fineweb-10m-sparse/summaries/pole_stats.parquet"
t = pq.read_table(PQ).to_pydict()
rows = [dict(latent=t["latent"][i], neuron=t["top1_neuron"][i], s1=t["top1_share"][i],
             eff=t["eff_neurons"][i], fired=t["fired_tokens"][i])
        for i in range(t.num_rows) if t["top1_share"][i] >= 0.5]
rows.sort(key=lambda r: -r["s1"])
rows = rows[:20]
ps = pq.read_table(PS, columns=None).to_pydict()
print("pole_stats cols:", list(ps.keys()), "rows:", len(ps[list(ps.keys())[0]]), flush=True)
layer = ps.get("layer"); neu = ps.get("neuron")
idx = {}
for i in range(len(neu)):
    if layer is None or layer[i] == 57:
        idx[neu[i]] = i
out = []
print(f"{'latent':>7} {'neuron':>7} {'top1_sh':>8} {'eff':>7} {'fired':>7} | {'pos_q99':>8} {'neg_q99.99':>10} {'frac_pos':>8}")
for r in rows:
    i = idx.get(r["neuron"])
    pq99 = ps["pos_q99"][i] if i is not None and "pos_q99" in ps else None
    nq = ps.get("neg_q9999", ps.get("neg_q99_99"))
    nq9999 = nq[i] if (i is not None and nq) else None
    fp = ps["frac_pos"][i] if i is not None and "frac_pos" in ps else None
    print(f"{r['latent']:>7} {r['neuron']:>7} {r['s1']:>8.4f} {r['eff']:>7.1f} {r['fired']:>7} | "
          f"{pq99 if pq99 is not None else float('nan'):>8.4f} {nq9999 if nq9999 is not None else float('nan'):>10.4f} {fp if fp is not None else float('nan'):>8.4f}", flush=True)
    out.append({**r, "pos_q99": pq99, "neg_q9999": nq9999, "frac_pos": fp})
json.dump({"layer": 57, "n_top1_ge_0.5_total": None, "shortlist": out},
          open("/data/scratch/apx186/marin32b/sae-64k/layer=57/nominate_enc/shortlist_top20.json", "w"), indent=1)
print("SHORTLIST WRITTEN", flush=True)
