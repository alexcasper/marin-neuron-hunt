#!/usr/bin/env python3
"""Characterize the 40 SAE-nominated L57 neurons (alex follow-on 2026-11-08).

For each latent with attribution top1_share >= 0.5:
  (1) attribution: latent, top1_neuron, top1_share, eff_neurons, fired_tokens
  (2) pattern-pass label from interp/patterns.jsonl (family, top_form, span)
  (3) neuron's own atlas exemplars (top-5 per pole) with +-16 tok context
  (4) logit_lens top/bottom pushed tokens per pole
  (+) pole_stats: pos_q99, neg_q9999, frac_pos
Writes characterization.md. CAVEAT baked into doc: labels describe what the
latent DETECTS / neuron fires on (correlational); hunt-format verification
needs interventions.
"""
import json
import pyarrow.parquet as pq
import pyarrow.compute as pc

S = "/data/scratch/apx186/marin32b"
SUM = f"{S}/atlas/pilot/fineweb-10m-sparse/summaries"
L57 = f"{S}/sae-64k/layer=57"
NOMD = f"{L57}/nominate_enc"
OUT = f"{NOMD}/characterization.md"
NEURONS_N = 40
WINDOW = 16

# ---- (1) nominations ----
t = pq.read_table(f"{NOMD}/attribution_enc.parquet").to_pydict()
NROWS = len(t["latent"])
noms = [dict(latent=t["latent"][i], neuron=t["top1_neuron"][i], s1=t["top1_share"][i],
             eff=t["eff_neurons"][i], fired=t["fired_tokens"][i])
        for i in range(NROWS) if t["top1_share"][i] >= 0.5]
noms.sort(key=lambda r: -r["s1"])
noms = noms[:NEURONS_N]
neu_set = {r["neuron"] for r in noms}
lat_set = {r["latent"] for r in noms}
n_total_ge = sum(1 for i in range(NROWS) if t["top1_share"][i] >= 0.5)

# ---- (2) pattern labels ----
pat = {}
with open(f"{L57}/interp/patterns.jsonl") as f:
    for line in f:
        d = json.loads(line)
        if d.get("latent_id") in lat_set:
            hit = (d.get("hits") or [{}])[0]
            pat[d["latent_id"]] = {
                "family": d.get("family"), "top_form": d.get("top_form"),
                "form_share": d.get("form_share"), "fire_rate": d.get("fire_rate"),
                "span": (hit.get("span") or "")[:160],
            }

# ---- (+) pole_stats for L57 neurons ----
_tbl = pq.read_table(f"{SUM}/pole_stats.parquet")
ps = _tbl.filter(pc.equal(_tbl.column("layer"), 57)).to_pydict()
psmap = {ps["neuron"][i]: i for i in range(len(ps["neuron"]))}

# ---- (3) exemplars: top-5 per pole for our neurons ----
pf = pq.ParquetFile(f"{SUM}/exemplars.parquet")
ex = {}  # (neuron, pole_sign) -> [(rank, token_idx, act)]
for rg in range(pf.metadata.num_row_groups):
    d = pf.read_row_group(rg, columns=["layer", "neuron", "pole", "rank", "token_idx", "act"]).to_pydict()
    for i in range(len(d["neuron"])):
        if d["layer"][i] == 57 and d["neuron"][i] in neu_set and d["rank"][i] < 5:
            ex.setdefault((d["neuron"][i], d["pole"][i]), []).append(
                (d["rank"][i], d["token_idx"][i], d["act"][i]))

# ---- tokens for context windows ----
tf = pq.read_table(f"{SUM}/tokens.parquet").to_pydict()
id2str, tokdoc, tokpos = {}, {}, {}
for i in range(len(tf["token_idx"])):
    ti = tf["token_idx"][i]
    id2str.setdefault(tf["token_id"][i], tf["token_str"][i])
    tokdoc[ti] = tf["doc_id"][i]
    tokpos[ti] = tf["pos"][i]
# per-doc pos -> token_idx for windows
doc_map = {}
need_tokens = {ti for (n, p) in ex for (_, ti, _) in ex[(n, p)]}
need_docs = {tokdoc[ti] for ti in need_tokens if ti in tokdoc}
for i in range(len(tf["token_idx"])):
    ti = tf["token_idx"][i]
    if tf["doc_id"][i] in need_docs:
        doc_map.setdefault(tf["doc_id"][i], []).append((tf["pos"][i], ti))

def context(ti):
    """+-WINDOW token_str window around ti, centered token marked."""
    d = tokdoc.get(ti)
    if d is None or d not in doc_map:
        return ""
    p = tokpos[ti]
    toks = [s for (pp, t2) in sorted(doc_map[d]) if abs(pp - p) <= WINDOW
            for s in [id2str.get(tf["token_id"][t2], "?")]]
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

# ---- (4) logit lens for our neurons ----
_tbl = pq.read_table(f"{SUM}/logit_lens.parquet")
ll = _tbl.filter(pc.equal(_tbl.column("layer"), 57)).to_pydict()
llmap = {ll["neuron"][i]: i for i in range(len(ll["neuron"])) if ll["neuron"][i] in neu_set}

# ---- emit markdown ----
L = []
L.append("# L57 SAE-nominated neuron characterization (encoder attribution)\n")
L.append("Source: `sae-64k/layer=57/nominate_enc/attribution_enc.parquet` — latents with "
         f"attribution top1_share >= 0.5 (n={n_total_ge}); top {len(noms)} by share shown. "
         f"Attribution run: ckpt sae_L57w64_final_s21000, 200,704 atlas tokens, K=5 column-shuffle "
         "control (real p50 0.064 vs permuted p50 0.000).\n")
L.append("> **CAVEAT**: labels below describe what the latent DETECTS / what the neuron fires on "
         "(correlational evidence from Fineweb-10m-sparse). Hunt-format verification requires "
         "intervention experiments (activation patching / ablation), per the hunt README.\n")
L.append("## Top-20 summary table\n")
L.append("| latent | neuron | top1_share | eff_neurons | fired | family | top_form | pos_q99 | neg_q99.99 | frac_pos |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
for r in noms:
    i = psmap.get(r["neuron"])
    pq99 = ps["pos_q99"][i] if i is not None else float("nan")
    nq = ps["neg_q9999"][i] if i is not None else float("nan")
    fp = ps["frac_pos"][i] if i is not None else float("nan")
    p = pat.get(r["latent"], {})
    L.append(f"| {r['latent']} | N{r['neuron']} | {r['s1']:.3f} | {r['eff']:.1f} | {r['fired']} "
             f"| {p.get('family','-')} | {p.get('top_form','-')} | {pq99:.4f} | {nq:.4f} | {fp:.3f} |")
L.append("")
L.append("## Per-neuron detail\n")
for r in noms:
    n = r["neuron"]
    i = psmap.get(n)
    p = pat.get(r["latent"], {})
    L.append(f"### L57-N{n} <- latent {r['latent']}\n")
    L.append("```yaml")
    L.append(f"attribution: {{latent: {r['latent']}, top1_neuron: N{n}, top1_share: {r['s1']:.4f}, "
             f"eff_neurons: {r['eff']:.1f}, fired_tokens: {r['fired']}}}")
    L.append(f"pattern: {{family: {p.get('family')}, top_form: {p.get('top_form')!r}, "
             f"form_share: {p.get('form_share')}, fire_rate: {p.get('fire_rate')}}}")
    if p.get("span"):
        L.append(f"pattern_span: >-{chr(10)}  {p['span']}")
    if i is not None:
        L.append(f"pole_stats: {{pos_q99: {ps['pos_q99'][i]:.4f}, neg_q9999: {ps['neg_q9999'][i]:.4f}, "
                 f"frac_pos: {ps['frac_pos'][i]:.4f}, max: {ps['max'][i]:.4f}, min: {ps['min'][i]:.4f}}}")
    for pole in sorted({pp for (nn, pp) in ex if nn == n}):
        sign = "+" if pole > 0 else "-"
        L.append(f"exemplars_{sign}:")
        for (rk, ti, av) in sorted(ex[(n, pole)]):
            L.append(f"  - {{rank: {rk}, token_idx: {ti}, act: {av:.4f}, ctx: {json.dumps(context(ti))}}}")
    li = llmap.get(n)
    if li is not None:
        L.append(f"logit_lens_push_plus: {dec(ll['top_ids'][li], ll['top_vals'][li])}")
        L.append(f"logit_lens_push_minus: {dec(ll['bottom_ids'][li], ll['bottom_vals'][li])}")
    L.append("```")
    L.append("")
open(OUT, "w").write("\n".join(L))
print(f"CHAR DONE: {len(noms)} neurons (of {n_total_ge} with top1>=0.5) -> {OUT}", flush=True)
