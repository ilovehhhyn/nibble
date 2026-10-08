#!/bin/bash
#SBATCH --job-name=nibble-transfer15
#SBATCH --account=griffith
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gpfs/ARORA/hh9077/nibble/work/slurm-%j.out
set -euo pipefail
source "$NIBBLE_PROJECT_ROOT/work/env.sh"
MODEL="$NIBBLE_PROJECT_ROOT/../hf_cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
CKPT=$($NIBBLE_PYTHON - "$NIBBLE_PROJECT_ROOT/state/comparison-$NIBBLE_REFERENCE_JOB/campaign.json" <<'CONFIG'
import json,sys
from pathlib import Path
c=json.loads(Path(sys.argv[1]).read_text())
assert c['status']=='completed'
print(Path(c['checkpoint']).parent)
CONFIG
)
export COUNT=64
bash "$NIBBLE_CODE_ROOT/tools/transfer/login_replay.sh" "$CKPT" "$MODEL" 15 "$NIBBLE_PROJECT_ROOT/../areal-kv/results/calibration-v0.jsonl" "$NIBBLE_PROJECT_ROOT/state/transfer-$SLURM_JOB_ID"
