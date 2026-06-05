#!/bin/bash
set -u
BIN="$1"
TIMEOUT="${2:-25}"
export EGGLOG_TEST_FELDERA=1
OUT=/tmp/egglog-feldera/.tmp/feldera_results2.txt
: > "$OUT"
NAMES=$("$BIN" --list 2>/dev/null | sed -n 's/: test$//p' | grep '_feldera$')
run_one() {
  local name="$1"
  ( "$BIN" --exact "$name" --nocapture --test-threads=1 ) >/tmp/egglog-feldera/.tmp/one2.log 2>&1 &
  local pid=$!
  local waited=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 1; waited=$((waited+1))
    if [ "$waited" -ge "$TIMEOUT" ]; then
      kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
      echo "TIMEOUT  $name" >> "$OUT"; return
    fi
  done
  wait "$pid"; local rc=$?
  if [ "$rc" -eq 0 ]; then echo "PASS     $name" >> "$OUT"
  else
    reason=$(grep -aoE "(panicked at|feldera\)?:|not supported[^\"]*|unbound|snapshot)[^\n]*" /tmp/egglog-feldera/.tmp/one2.log | head -1 | cut -c1-160)
    echo "FAIL     $name :: ${reason}" >> "$OUT"
  fi
}
for n in $NAMES; do run_one "$n"; done
echo "=== SUMMARY ===" >> "$OUT"
echo "PASS:    $(grep -c '^PASS' "$OUT")" >> "$OUT"
echo "FAIL:    $(grep -c '^FAIL' "$OUT")" >> "$OUT"
echo "TIMEOUT: $(grep -c '^TIMEOUT' "$OUT")" >> "$OUT"
