#!/usr/bin/env bash
# Robust resumable upload of 02_Sample.zip via sftp reput.
# Tolerant to WiFi drops: each iteration trims 1 MB tail safety,
# verifies size hasn't shrunk below the known-good 13 GB,
# then sftp reputs. Stops when remote size == expected.

set -uo pipefail

LOCAL="c:/Users/li_k1/M_thesis/02_Sample.zip"
REMOTE_HOST="li_k1@login002.merlin7.psi.ch"
REMOTE_PATH="/data/user/li_k1/M_thesis/02_Sample.zip"
EXPECTED=36351384126    # 34 GB exactly
GOOD_FLOOR=13000000000  # 13 GB — verified clean prefix
SAFETY_TRIM=1048576     # 1 MB safety margin per attempt

attempt=0
while true; do
  attempt=$((attempt + 1))
  R=$(ssh -o ConnectTimeout=15 "${REMOTE_HOST}" "stat -c%s '${REMOTE_PATH}'" 2>/dev/null || echo ERR)
  if [ "$R" = "$EXPECTED" ]; then
    echo "[$(date)] remote size matches expected ($EXPECTED). DONE."
    exit 0
  fi
  if [ "$R" = "ERR" ]; then
    echo "[$(date)] cannot read remote size (network?). Sleeping 15s."
    sleep 15
    continue
  fi
  # Trim 1 MB safety, but never below the known-good 13 GB floor.
  TRIM_TO=$((R - SAFETY_TRIM))
  if [ "$TRIM_TO" -lt "$GOOD_FLOOR" ]; then TRIM_TO="$GOOD_FLOOR"; fi
  echo "[$(date)] attempt $attempt: remote=$R, trimming to $TRIM_TO, reputing..."
  ssh -o ConnectTimeout=15 "${REMOTE_HOST}" "truncate -s ${TRIM_TO} '${REMOTE_PATH}'" \
    || { echo "  truncate failed, sleeping 15s"; sleep 15; continue; }
  sftp -o ServerAliveInterval=10 -o ServerAliveCountMax=2 -o ConnectTimeout=15 \
       -b - "${REMOTE_HOST}" <<EOF
cd /data/user/li_k1/M_thesis/
reput "${LOCAL}" 02_Sample.zip
EOF
  ex=$?
  echo "[$(date)] attempt $attempt: sftp exit=$ex"
  if [ "$ex" -ne 0 ]; then
    sleep 30
  fi
done
