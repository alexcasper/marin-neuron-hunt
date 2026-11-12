#!/usr/bin/env python3
"""Token selectivity per (layer, neuron, pole), pass 2: full sparse tier.

Same stats as pass 1 (token_selectivity_exemplars.py), but over EVERY token
where the neuron appears in the atlas sparse tier (each token's top-64
neurons) for fineweb-10m (10,000,384 tokens), not just the top ~31 exemplars.
Pole = sign of act = silu(gate) * up.

Columns (one row per layer, neuron, pole present):
  n_fire               tokens where the neuron is in the sparse tier with this sign
  n_distinct_tokens    distinct token ids among them
  n_distinct_docs      distinct documents among them
  top_token_id/_str    most frequent token id (ties: arbitrary)
  top_token_share      count share of that token, 0..1
  top_token_mass_share |act| mass share of that token, 0..1
  top_token_lift       top_token_share / that token's corpus frequency
                       (>> 1 = selective; ~1 = just a common token)
  eff_tokens           1 / sum_t p_t^2 over count shares

Writes one parquet per layer to OUT_DIR/layer=NN.parquet; finished layers are
skipped, so the run resumes after an interruption.

Usage: token_selectivity_full.py [--layers 0,57] [--chunks 8]
"""
import argparse, glob, os, time
import duckdb

ROOT = "/mnt/d/ops/sh/data/pilot/fineweb-10m-sparse"
SUM = f"{ROOT}/summaries"
OUT_DIR = "/mnt/d/ops/sh/data/derived/token_selectivity_full"
N_NEURONS = 27648

ap = argparse.ArgumentParser()
ap.add_argument("--layers", default=",".join(str(i) for i in range(64)))
ap.add_argument("--chunks", type=int, default=8, help="neuron ranges per layer (bounds memory)")
ap.add_argument("--neuron-max", type=int, default=N_NEURONS, help="smoke: only neurons < this")
a = ap.parse_args()
os.makedirs(OUT_DIR, exist_ok=True)

con = duckdb.connect()
con.execute("SET memory_limit='16GB'; SET threads=12; SET preserve_insertion_order=false")
con.execute(f"SET temp_directory='{OUT_DIR}/.duckdb_tmp'")
# token lookup + corpus frequency, materialised once
con.execute(f"""CREATE TABLE tok AS
  SELECT token_idx, token_id, doc_id FROM read_parquet('{SUM}/tokens.parquet')""")
con.execute("""CREATE TABLE freq AS
  SELECT token_id, count(*)::DOUBLE / (SELECT count(*) FROM tok) AS f FROM tok GROUP BY token_id""")
con.execute(f"""CREATE TABLE vocab AS
  SELECT token_id, any_value(token_str) AS token_str FROM read_parquet('{SUM}/tokens.parquet') GROUP BY token_id""")

for L in [int(x) for x in a.layers.split(",")]:
    LL = f"{L:02d}"
    out = f"{OUT_DIR}/layer={LL}.parquet"
    if os.path.exists(out) and a.neuron_max == N_NEURONS:
        print(f"layer {LL}: exists, skip", flush=True); continue
    src = sorted(glob.glob(f"{ROOT}/sparse/layer={LL}/*.parquet"))
    assert src, f"no sparse files for layer {LL}"
    srcs = "[" + ",".join(f"'{s}'" for s in src) + "]"
    t0 = time.time()
    con.execute("DROP TABLE IF EXISTS res")
    step = -(-a.neuron_max // a.chunks)
    for lo in range(0, a.neuron_max, step):
        hi = min(a.neuron_max, lo + step)
        sql = f"""
        WITH s AS (
          SELECT s.neuron, tok.token_id, tok.doc_id,
                 (s.gate::FLOAT / (1 + exp(-s.gate::FLOAT))) * s.up::FLOAT AS act
          FROM read_parquet({srcs}) s JOIN tok USING (token_idx)
          WHERE s.neuron >= {lo} AND s.neuron < {hi}
        ),
        s2 AS (SELECT neuron, token_id, doc_id, sign(act)::TINYINT AS pole, abs(act) AS a FROM s WHERE act <> 0),
        g AS (SELECT neuron, pole, token_id, count(*) AS c, sum(a) AS m FROM s2 GROUP BY ALL),
        g2 AS (SELECT *, sum(c) OVER w AS n, sum(m) OVER w AS mt FROM g
               WINDOW w AS (PARTITION BY neuron, pole)),
        agg AS (
          SELECT neuron, pole,
                 any_value(n)::BIGINT                   AS n_fire,
                 count(*)::INTEGER                      AS n_distinct_tokens,
                 arg_max(token_id, c)                   AS top_token_id,
                 (max(c) / any_value(n))::FLOAT         AS top_token_share,
                 (max_by(m, c) / any_value(mt))::FLOAT  AS top_token_mass_share,
                 (1.0 / sum((c / n) ^ 2))::FLOAT        AS eff_tokens
          FROM g2 GROUP BY ALL
        ),
        docs AS (SELECT neuron, pole, count(DISTINCT doc_id)::INTEGER AS n_distinct_docs FROM s2 GROUP BY ALL)
        SELECT {L}::UTINYINT AS layer, agg.neuron::USMALLINT AS neuron, agg.pole, n_fire,
               n_distinct_tokens, n_distinct_docs, top_token_id, vocab.token_str AS top_token_str,
               top_token_share, top_token_mass_share,
               (top_token_share / freq.f)::FLOAT AS top_token_lift, eff_tokens
        FROM agg JOIN docs USING (neuron, pole)
        LEFT JOIN vocab ON vocab.token_id = agg.top_token_id
        LEFT JOIN freq ON freq.token_id = agg.top_token_id
        """
        if lo == 0:
            con.execute(f"CREATE TABLE res AS {sql}")
        else:
            con.execute(f"INSERT INTO res {sql}")
        print(f"layer {LL}: neurons {lo}-{hi - 1} done ({time.time() - t0:.0f}s)", flush=True)
    target = out if a.neuron_max == N_NEURONS else f"{OUT_DIR}/smoke_layer={LL}_n{a.neuron_max}.parquet"
    tmp = target + ".tmp"
    con.execute(f"COPY (SELECT * FROM res ORDER BY neuron, pole) TO '{tmp}' (FORMAT PARQUET)")
    os.replace(tmp, target)
    n = con.execute("SELECT count(*), sum(n_fire) FROM res").fetchall()[0]
    print(f"layer {LL}: wrote {n[0]} rows, {n[1]} firings -> {target} in {time.time() - t0:.0f}s", flush=True)
