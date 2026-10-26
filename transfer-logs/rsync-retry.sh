#!/bin/bash
# rsync dawn -> dusk with auto-retry (resume via --partial; restarts until exit 0)
LOG=/mnt/d/ops/sh/transfer-logs/rsync-fineweb-10m.log
attempt=0
while true; do
  attempt=$((attempt+1))
  echo "=== attempt $attempt start $(date -Is) ===" >> "$LOG"
  rsync -a --info=progress2 --partial --checksum \
    /mnt/d/ops/sh/data/pilot/fineweb-10m-sparse/ \
    dusk:~/marin-atlas/fineweb-10m-sparse/ >> "$LOG" 2>&1
  rc=$?
  echo "=== attempt $attempt exit=$rc $(date -Is) ===" >> "$LOG"
  [ $rc -eq 0 ] && break
  if [ $attempt -ge 20 ]; then echo "GIVING_UP after $attempt attempts" >> "$LOG"; break; fi
  sleep 15
done
