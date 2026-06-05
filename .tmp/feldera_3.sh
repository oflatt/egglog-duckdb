#!/bin/bash
set -u
BIN="$1"
TIMEOUT="${2:-90}"
export EGGLOG_TEST_FELDERA=1
OUT=/tmp/egglog-feldera/.tmp/feldera_3_results.txt
: > "$OUT"
for name in rectangle_feldera repro_should_saturate_feldera naturals_feldera; do
  ( "$BIN" --exact "$name" --nocapture --test-threads=1 ) >/tmp/egglog-feldera/.tmp/one3.log 2>&1 &
  pid=$!
  waited=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 1; waited=$((waited+1))
    if [ "$waited" -ge "$TIMEOUT" ]; then
      kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
      echo "TIMEOUT($TIMEOUT s)  $name" >> "$OUT"; pid=""; break
    fi
  done
  if [ -n "${pid:-}" ]; then
    wait "$pid"; rc=$?
    if [ "$rc" -eq 0 ]; then echo "PASS  $name (${waited}s)" >> "$OUT"
    else echo "FAIL  $name :: $(grep -aoE 'panicked at[^\n]*' /tmp/egglog-feldera/.tmp/one3.log | head -1 | cut -c1-160)" >> "$OUT"; fi
  fi
done
echo "=== DONE ===" >> "$OUT"
