import numpy as np, pyarrow.parquet as pq, pyarrow.compute as pc
S="/data/scratch/apx186/marin32b"
EXCL={2462,25368,5873,10940,17606,3872,26033,14976,26751,26059,10275}
t=pq.read_table(f"{S}/sae-64k/layer=57/nominate_enc/attribution_enc.parquet").to_pydict()
tbl=pq.read_table(f"{S}/atlas/pilot/fineweb-10m-sparse/summaries/pole_stats.parquet")
tbl=tbl.filter(pc.equal(tbl.column("layer"),57)).to_pydict()
ps={tbl["neuron"][i]:i for i in range(len(tbl["neuron"]))}
band=[(t["latent"][i],t["top1_neuron"][i],t["top1_share"][i],t["eff_neurons"][i],t["fired_tokens"][i])
      for i in range(len(t["latent"])) if 0.30<=t["top1_share"][i]<0.50]
print(f"BAND_TOTAL {len(band)}")
med=float(np.median([tbl["pos_q99"][i] for i in ps.values()]))
kept=[]
for la,ne,s1,ef,fi in sorted(band,key=lambda r:-r[2]):
    if ne in EXCL: continue
    i=ps.get(ne)
    fp=tbl["frac_pos"][i]; pq99=tbl["pos_q99"][i]; nq=tbl["neg_q9999"][i]
    if fp>0.99 or pq99>100*med: continue
    kept.append((la,ne,s1,ef,fi,pq99,nq,fp))
print(f"KEPT {len(kept)} (after excl)")
print(f"{'latent':>7} {'neuron':>7} {'top1_sh':>8} {'eff':>6} {'fired':>7} {'pos_q99':>8} {'neg_q99.99':>10} {'frac_pos':>8}")
for la,ne,s1,ef,fi,pq99,nq,fp in kept[:25]:
    print(f"{la:>7} N{ne:<6} {s1:>8.4f} {ef:>6.1f} {fi:>7} {pq99:>8.3f} {nq:>10.3f} {fp:>8.3f}")
