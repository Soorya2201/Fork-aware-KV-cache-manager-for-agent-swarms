#!/bin/bash
# Copy this run's results and job logs into the private data repo and push them.
# Runs as the last job of run_all.sh; can also be run by hand on a login node:
#   bash slurm/publish.sh
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/carc.env"
cd "$PROJ"
STAMP=$(date +%Y%m%d_%H%M)
DEST="$PRIVATE/carc_runs/$STAMP"
mkdir -p "$DEST/logs"
rsync -a --exclude 'private/' results/ "$DEST/results/"
cp logs/*.out logs/*.csv "$DEST/logs/" 2>/dev/null || true
for f in logs/sglang_*.log; do      # server logs can be large: keep the start and the end
  [ -f "$f" ] && { head -200 "$f"; echo "... [trimmed] ..."; tail -300 "$f"; } > "$DEST/logs/$(basename "$f")"
done
squeue -u "$USER" -h -o "%i %j %T" > "$DEST/logs/queue_at_publish.txt" 2>/dev/null || true
sacct -u "$USER" -S "$(date -d '-3 days' +%F)" -o JobID,JobName%20,State,Elapsed,ExitCode -n \
  > "$DEST/logs/sacct.txt" 2>/dev/null || true
cd "$PRIVATE"
git add -A
git -c user.name="${GIT_NAME:-Soorya2201}" -c user.email="${GIT_EMAIL:-Soorya2201@users.noreply.github.com}" \
  commit -q -m "CARC run $STAMP" || echo "nothing new to commit"
git pull -q --rebase || true
git push -q && echo "pushed carc_runs/$STAMP"
