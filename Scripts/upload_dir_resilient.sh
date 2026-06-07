#!/usr/bin/env bash
# Resilient directory upload via batched sftp.
# Re-runnable: on each iteration, compares local vs remote inventory by size,
# only uploads files that are missing or wrong-size on the remote.
# Tolerant to WiFi drops: a dropped sftp session ends the current iteration
# and the loop simply re-queries inventory and retries with the smaller batch.

set -uo pipefail

LOCAL_DIR="c:/Users/li_k1/M_thesis/02_Sample_raw"
REMOTE_HOST="li_k1@login002.merlin7.psi.ch"
REMOTE_DIR="/data/user/li_k1/M_thesis/02_Sample_raw"
PATTERN="*.tiff"

ssh -o ConnectTimeout=15 "${REMOTE_HOST}" "mkdir -p '${REMOTE_DIR}'"

attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "[$(date)] === attempt $attempt: building inventory ==="

  # Local inventory: name<TAB>size
  local_inv=$(mktemp)
  for f in "${LOCAL_DIR}"/${PATTERN}; do
    name=$(basename "$f")
    size=$(stat -c%s "$f")
    printf '%s\t%s\n' "$name" "$size"
  done | sort > "${local_inv}"
  total_local=$(wc -l < "${local_inv}")
  echo "  local: ${total_local} files"

  # Remote inventory
  remote_inv=$(mktemp)
  ssh -o ConnectTimeout=15 "${REMOTE_HOST}" \
    "cd '${REMOTE_DIR}' 2>/dev/null && stat -c '%n	%s' ${PATTERN} 2>/dev/null" \
    | sort > "${remote_inv}"
  total_remote=$(wc -l < "${remote_inv}")
  echo "  remote: ${total_remote} files"

  # Files missing or wrong-size: present in local but not in remote_inv (matched by name+size)
  missing=$(mktemp)
  comm -23 "${local_inv}" "${remote_inv}" | cut -f1 > "${missing}"
  n_missing=$(wc -l < "${missing}")
  rm -f "${local_inv}" "${remote_inv}"

  if [ "${n_missing}" -eq 0 ]; then
    echo "[$(date)] === all ${total_local} files present and correct on remote. DONE ==="
    rm -f "${missing}"
    exit 0
  fi
  echo "  to upload: ${n_missing}"

  # Build sftp batch file
  batch=$(mktemp)
  echo "cd '${REMOTE_DIR}'" > "${batch}"
  while IFS= read -r name; do
    echo "put '${LOCAL_DIR}/${name}' '${name}'" >> "${batch}"
  done < "${missing}"
  rm -f "${missing}"

  echo "[$(date)] running sftp batch of ${n_missing} files..."
  sftp -o ServerAliveInterval=10 -o ServerAliveCountMax=2 -o ConnectTimeout=15 \
       -b "${batch}" "${REMOTE_HOST}"
  ex=$?
  rm -f "${batch}"
  echo "[$(date)] sftp exit=${ex} after attempt $attempt"

  if [ "${ex}" -ne 0 ]; then
    sleep 15
  fi
done
