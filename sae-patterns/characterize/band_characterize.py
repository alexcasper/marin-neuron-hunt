#!/usr/bin/env python3
"""Band characterization: L57 neurons with >=6 fired tokens from the 0.3-0.5
attribution band. Same recipe as characterize_nominees.py: both-pole exemplars
(top-5, +-16 tok context), logit lens push/pull, pole_stats, plus the
nominating latents from attribution_enc.parquet."""
import json
import pyarrow.parquet as pq
import pyarrow.compute as pc

S = "/data/scratch/apx186/marin32b"
SUM = f"{S}/atlas/pilot/fineweb-10m-sparse/summaries"
NOMD = f"{S}/sae-64k/layer=57/nominate_enc"
OUT = f"{NOMD}/band-characterization.md"
NEURONS = [15656, 5847, 10507, 6946, 12238, 8538, 492]
WINDOW = 16

# nominating latents for these neurons
t = pq.read_table(f"{NOMD}/attribution_enc.parquet").to_pydict()
noms = {}
for i in range(len(t["latent"])):
    n = t["top1_neuron"][i]
    if n in NEURONS and t["top1_share"][i] >= 0.30:
        noms.setdefault(n, []).append(
            dict(latent=t["latent"][i], s1=t["top1_share"][i],
                 eff=t["eff_neurons"][i], fired=t["fired_tokens"][i]))
for n in noms:
    noms[n].sort(key=lambda r: -r["s1"])

# pole_stats
_tbl = pq.read_table(f"{SUM}/pole_stats.parquet")
ps = _tbl.filter(pc.equal(_tbl.column("layer"), 57)).to_pydict()
psmap = {ps["neuron"][i]: i for i in range(len(ps["neuron"]))}

# exemplars top-5 per pole
neu_set = set(NEURONS)
pf = pq.ParquetFile(f"{SUM}/exemplars.parquet")
ex = {}
for rg in range(pf.metadata.num_row_groups):
    d = pf.read_row_group(rg, columns=["layer", "neuron", "pole", "rank", "token_idx", "act"]).to_pydict()
    for i in range(len(d["neuron"])):
        if d["layer"][i] == 57 and d["neuron"][i] in neu_set and d["rank"][i] < 5:
            ex.setdefault((d["neuron"][i], d["pole"][i]), []).append(
                (d["rank"][i], d["token_idx"][i], d["act"][i]))

# tokens + windows
tf = pq.read_table(f"{SUM}/tokens.parquet").to_pydict()
id2str, tokdoc, tokpos = {}, {}, {}
for i in range(len(tf["token_idx"])):
    ti = tf["token_idx"][i]
    id2str.setdefault(tf["token_id"][i], tf["token_str"][i])
    tokdoc[ti] = tf["doc_id"][i]
    tokpos[ti] = tf["pos"][i]
need_tokens = {ti for (n, p) in ex for (_, ti, _) in ex[(n, p)]}
need_docs = {tokdoc[ti] for ti in need_tokens if ti in tokdoc}
doc_map = {}
for i in range(len(tf["token_idx"])):
    ti = tf["token_idx"][i]
    d = tf["doc_id"][i]
    if d in need_docs:
        doc_map.setdefault(d, []).append((tf["pos"][i], ti))

def context(ti):
    d = tokdoc.get(ti)
    if d is None or d not in doc_map:
        return ""
    p = tokpos[ti]
    out = []
    for (pp, t2) in sorted(doc_map[d]):
        if abs(pp - p) > WINDOW:
            continue
        s = id2str.get(tf["token_id"][t2], "?")
        out.append(f"[[{s}]]" if t2 == ti else s)
    return " ".join(out).replace("\n", " ")[:600]

def dec(ids, vals, k=10):
    return ", ".join(f"{id2str.get(int(i), f'<tid {int(i)}>')} ({v:.2f})"
                     for i, v in list(zip(ids, vals))[:k])

ll = pq.read_table(f"{SUM}/logit_lens.parquet")
ll = ll.filter(pc.equal(ll.column("layer"), 57)).to_pydict()
llmap = {ll["neuron"][i]: i for i in range(len(ll["neuron"])) if ll["neuron"][i] in neu_set}

L = []
L.append("# L57 band characterization — encoder-attribution neurons with >=6 fired tokens\n")
L.append("Neurons from the 0.3-0.5 attribution band (>=0.5 set and already-reported neurons excluded). "
         "fired_tokens aggregated across nominating latents. CAVEAT: correlational evidence "
         "(what the neuron fires on in Fineweb-10m-sparse); hunt verification needs interventions.\n")
for n in NEURONS:
    i = psmap.get(n)
    L.append(f"## L57-N{n}\n")
    L.append("```yaml")
    lats = ", ".join(f"{{latent: {r['latent']}, top1_share: {r['s1']:.4f}, eff: {r['eff']:.1f}, fired: {r['fired']}}}"
                     for r in noms.get(n, []))
    L.append(f"nominating_latents: [{lats}]")
    L.append(f"fired_tokens_total: {sum(r['fired'] for r in noms.get(n, []))}")
    if i is not None:
        L.append(f"pole_stats: {{pos_q99: {ps['pos_q99'][i]:.4f}, neg_q9999: {ps['neg_q9999'][i]:.4f}, "
                 f"frac_pos: {ps['frac_pos'][i]:.4f}, max: {ps['max'][i]:.4f}, min: {ps['min'][i]:.4f}}}")
    for pole in sorted({pp for (nn, pp) in ex if nn == n}):
        sign = "+" if pole > 0 else "-"
        L.append(f"exemplars_{sign}:")
        for (rk, ti, av) in sorted(ex[(n, pole)]):
            L.append(f"  - {{rank: {rk}, token_idx: {ti}, act: {av:.4f}, tok: {json.dumps(id2str.get(tf['token_id'][ti], '?'))}, ctx: {json.dumps(context(ti))}}}")
    li = llmap.get(n)
    if li is not None:
        L.append(f"logit_lens_push_plus: {dec(ll['top_ids'][li], ll['top_vals'][li])}")
        L.append(f"logit_lens_push_minus: {dec(ll['bottom_ids'][li], ll['bottom_vals'][li])}")
    L.append("```")
    L.append("")
open(OUT, "w").write("\n".join(L))
print(f"BAND CHAR DONE: {len(NEURONS)} neurons -> {OUT}", flush=True)
