#!/usr/bin/env python3
"""Token selectivity per (layer, neuron, pole), pass 1: exemplar tail only.

Joins atlas exemplars.parquet (~31 top-ranked firing tokens per neuron per
pole) with tokens.parquet to measure how concentrated each pole's TOP TAIL is
on one token type. Measures the tail, not typical firing: pass 2
(full sparse tier) answers that.

Columns (one row per layer, neuron, pole):
  n_ex                 exemplars present for this pole
  n_distinct_tokens    distinct token ids among them
  n_distinct_docs      distinct documents among them (low = doc artifact risk)
  top_token_id/_str    most frequent token id among the exemplars (ties: arbitrary)
  top_token_share      count share of that token, 0..1
  top_token_mass_share |act| mass share of that token, 0..1
  eff_tokens           1 / sum_t p_t^2 over count shares (1 = single token type)

Usage: token_selectivity_exemplars.py [--layers 0,57] [--out PATH]
"""
import argparse, os, time
import duckdb

ROOT = "/mnt/d/ops/sh/data/pilot/fineweb-10m-sparse/summaries"
DEF_OUT = "/mnt/d/ops/sh/data/derived/token_selectivity_exemplars.parquet"

ap = argparse.ArgumentParser()
ap.add_argument("--layers", default=None, help="comma list; default all")
ap.add_argument("--out", default=DEF_OUT)
a = ap.parse_args()
os.makedirs(os.path.dirname(a.out), exist_ok=True)

con = duckdb.connect()
con.execute("SET memory_limit='16GB'; SET threads=12")
con.execute(f"SET temp_directory='{os.path.dirname(a.out)}/.duckdb_tmp'")
where = f"WHERE e.layer IN ({a.layers})" if a.layers else ""
t0 = time.time()
con.execute(f"""
COPY (
WITH e AS (
  SELECT e.layer, e.neuron, e.pole, t.token_id, t.doc_id, abs(e.act) AS a
  FROM read_parquet('{ROOT}/exemplars.parquet') e
  JOIN read_parquet('{ROOT}/tokens.parquet') t USING (token_idx)
  {where}
),
g AS (
  SELECT layer, neuron, pole, token_id, count(*) AS c, sum(a) AS m
  FROM e GROUP BY ALL
),
g2 AS (
  SELECT *, sum(c) OVER w AS n, sum(m) OVER w AS mt
  FROM g WINDOW w AS (PARTITION BY layer, neuron, pole)
),
agg AS (
  SELECT layer, neuron, pole,
         any_value(n)::INTEGER                    AS n_ex,
         count(*)::INTEGER                        AS n_distinct_tokens,
         arg_max(token_id, c)                     AS top_token_id,
         (max(c) / any_value(n))::FLOAT           AS top_token_share,
         (max_by(m, c) / any_value(mt))::FLOAT    AS top_token_mass_share,
         (1.0 / sum((c / n) ^ 2))::FLOAT          AS eff_tokens
  FROM g2 GROUP BY ALL
),
docs AS (
  SELECT layer, neuron, pole, count(DISTINCT doc_id)::INTEGER AS n_distinct_docs
  FROM e GROUP BY ALL
),
vocab AS (
  SELECT token_id, any_value(token_str) AS token_str
  FROM read_parquet('{ROOT}/tokens.parquet') GROUP BY token_id
)
SELECT agg.layer, agg.neuron, agg.pole, n_ex, n_distinct_tokens, n_distinct_docs,
       top_token_id, vocab.token_str AS top_token_str,
       top_token_share, top_token_mass_share, eff_tokens
FROM agg JOIN docs USING (layer, neuron, pole)
LEFT JOIN vocab ON vocab.token_id = agg.top_token_id
ORDER BY layer, neuron, pole
) TO '{a.out}' (FORMAT PARQUET)
""")
n = con.execute(f"SELECT count(*) FROM read_parquet('{a.out}')").fetchone()[0]
print(f"wrote {n} rows -> {a.out} in {time.time() - t0:.1f}s", flush=True)
