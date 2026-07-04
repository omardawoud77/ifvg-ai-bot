#!/bin/zsh
# Daily shadow-log backup: pulls shadow_log_*.csv from the Railway container
# (divine-balance / worker) and merges new rows into a local archive.
# Safe to run any time; dedupes by full line, keeps header once.
# Archive lives in phase2/data/ which is git-ignored — never committed.

set -u
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

REPO_DIR="$HOME/trading-indicator"
ARCHIVE_DIR="$REPO_DIR/ai-model/phase2/data/shadow_archive"
LOG="$ARCHIVE_DIR/pull.log"
# Shadow logs moved to the volume (/data) as of commit e777f1a; check both
# locations so the script works across old/new deploys.
REMOTE_PATHS=("/data" "/app/crypto")

mkdir -p "$ARCHIVE_DIR"
echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] pull start" >> "$LOG"

cd "$REPO_DIR" || { echo "  repo dir missing" >> "$LOG"; exit 1; }

for sym in btcusdt ethusdt solusdt; do
  tmp="$(mktemp)"
  got=""
  for rp in "${REMOTE_PATHS[@]}"; do
    # keep only the CSV header + ISO-timestamp rows — railway ssh can return
    # a gateway JSON blob mid-deploy, which must never reach the archive
    railway ssh -- cat "$rp/shadow_log_${sym}.csv" 2>/dev/null \
      | grep -E '^(ts_utc,|20[0-9]{2}-)' > "$tmp"
    if [ -s "$tmp" ]; then
      got="$rp"
      break
    fi
  done
  if [ -z "$got" ]; then
    echo "  $sym: no remote file found (bot restarted recently?)" >> "$LOG"
    rm -f "$tmp"
    continue
  fi
  archive="$ARCHIVE_DIR/shadow_log_${sym}.csv"
  if [ ! -f "$archive" ]; then
    cp "$tmp" "$archive"
    echo "  $sym: new archive, $(wc -l < "$archive" | tr -d ' ') rows (from $got)" >> "$LOG"
  else
    before=$(wc -l < "$archive" | tr -d ' ')
    # append rows not already present (header dedupes naturally)
    grep -Fxv -f "$archive" "$tmp" >> "$archive" || true
    after=$(wc -l < "$archive" | tr -d ' ')
    echo "  $sym: +$((after - before)) new rows (total $after, from $got)" >> "$LOG"
  fi
  rm -f "$tmp"
done
echo "  done" >> "$LOG"
