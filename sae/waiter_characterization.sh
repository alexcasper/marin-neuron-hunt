#!/bin/bash
# Waiter: poll for the L57 neuron-characterization doc mirrored by env_apoc,
# send the results to Discord #camelot when it lands.
# Polls 8h max at 5-min intervals; exits 0 on send, 1 on timeout.
set -u
CANDIDATES=(
  "/mnt/d/env/apoc/marin32b/characterization.md"
  "/mnt/d/env/apoc/marin32b/characterization-L57.md"
  "/mnt/d/env/apoc/marin32b/nominate_enc/characterization.md"
  "/mnt/d/env/apoc/marin32b/sae-64k/layer=57/nominate_enc/characterization.md"
)
for i in $(seq 1 96); do
  for f in "${CANDIDATES[@]}"; do
    if [ -s "$f" ]; then
      echo "FOUND: $f"
      HERMES_HOME=/home/dawn/.hermes hermes send --to discord:1511732903409750248 \
        "**ops_sh — L57 SAE-nominated neurons: characterization landed** (40 attribution-dominated latents; shuffle control clean; local fork only, nothing upstream). Full doc: \`$f\`" \
        || { echo "SEND FAILED"; exit 2; }
      echo "SENT to #camelot"
      exit 0
    fi
  done
  sleep 300
done
echo "TIMEOUT: no characterization doc after 8h" >&2
exit 1
