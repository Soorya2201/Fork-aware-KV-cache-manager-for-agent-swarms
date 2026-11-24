#!/bin/bash
# Submit the whole CARC pipeline in order, then push results to the private data repo.
# Run on a login node from the repo root:   bash slurm/run_all.sh
#
#   fc_setup ──┬── fc_prefill ──┐
#              ├── fc_replay ───┼── fc_policies ──┐
#              ├── fc_kernel    │                 │
#              ├── fc_compress  │                 ├── fc_publish (always runs last)
#              └── fc_trace7b ──┴─────────────────┘
#
# SKIP="trace7b kernel" leaves jobs out. Watch with: squeue -u $USER
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
ENV=slurm/carc.env
mkdir -p logs

if [ ! -f "$ENV" ]; then
  acct=$(sacctmgr -nP show assoc where user="$USER" format=account 2>/dev/null | sort -u | head -1)
  gres=$(sinfo -p gpu -h -o "%G" 2>/dev/null | tr ',' '\n')
  gpu=""
  for g in a100 a40 l40s h100; do echo "$gres" | grep -q "gpu:$g" && { gpu=$g; break; }; done
  [ -n "$acct" ] || { echo "Could not detect your SLURM account; run 'myaccount' and set ACCOUNT in $ENV"; }
  [ -n "$gpu" ] || { echo "No A100/A40/L40S/H100 found in 'sinfo -p gpu'; set GPU in $ENV"; gpu=a100; }
  sed -e "s|^ACCOUNT=.*|ACCOUNT=${acct:-CHANGE_ME}|" -e "s|^GPU=.*|GPU=$gpu|" \
      -e "s|^PROJ=.*|PROJ=$ROOT|" slurm/carc.env.example > "$ENV"
  echo "wrote $ENV (account=${acct:-?}, gpu=$gpu). Check it, then run this script again."
  exit 0
fi
source "$ENV"
grep -q "CHANGE_ME" "$ENV" && { echo "Set ACCOUNT in $ENV first (see 'myaccount')."; exit 1; }

[ -d "$PRIVATE/.git" ] || { echo "Private data repo missing at $PRIVATE. Clone it first:"
  echo "  git clone git@github-forkcache:Soorya2201/forkcache-data.git $PRIVATE"; exit 1; }
git -C "$PRIVATE" pull -q || echo "warning: could not pull the data repo"
[ -d "$ZAGE_DIR" ] || echo "warning: ZAGE_DIR=$ZAGE_DIR not found; fc_trace7b will fail"

sub() {  # sub <script> [sbatch args...]  -> prints job id
  local s=$1; shift
  sbatch --parsable --kill-on-invalid-dep=yes --account="$ACCOUNT" --partition="$PARTITION" --gres="gpu:$GPU:1" "$@" "slurm/$s"
}
skip() { [[ " ${SKIP:-} " == *" $1 "* ]]; }

setup=$(sub setup.sbatch); echo "fc_setup     $setup"
after="--dependency=afterok:$setup"
ids=()
skip prefill  || { prefill=$(sub prefill_curve.sbatch "$after"); ids+=("$prefill"); echo "fc_prefill   $prefill"; }
skip replay   || { replay=$(sub replay.sbatch "$after");         ids+=("$replay");  echo "fc_replay    $replay"; }
skip kernel   || { k=$(sub kernel.sbatch "$after");              ids+=("$k");       echo "fc_kernel    $k"; }
skip compress || { c=$(sub compression.sbatch "$after");         ids+=("$c");       echo "fc_compress  $c"; }
skip trace7b  || { t=$(sub trace_7b.sbatch "$after");            ids+=("$t");       echo "fc_trace7b   $t"; }
if ! skip policies; then
  dep="afterok:$setup"; [ -n "${prefill:-}" ] && dep="afterok:$setup,afterany:$prefill"   # needs setup; runs even if prefill fails
  p=$(sub replay_policies.sbatch "--dependency=$dep"); ids+=("$p"); echo "fc_policies  $p"
fi
all=$(IFS=:; echo "${ids[*]:-$setup}")
pub=$(sbatch --parsable --account="$ACCOUNT" --partition="$CPU_PARTITION" \
      --dependency="afterany:$setup:$all" slurm/publish.sbatch)
echo "fc_publish   $pub   (pushes results to the private repo when everything above ends)"
echo "watch: squeue -u $USER     logs: ls logs/"
