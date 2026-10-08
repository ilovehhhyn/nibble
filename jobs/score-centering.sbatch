#!/bin/bash
#SBATCH --job-name=nibble-sc-cost
#SBATCH --account=griffith
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/gpfs/ARORA/hh9077/nibble/work/slurm-%j.out
set -euo pipefail
source "$NIBBLE_PROJECT_ROOT/work/env.sh"
export PYTHONPATH="$NIBBLE_CODE_ROOT/src:$NIBBLE_PROJECT_ROOT/work/deps/areal-040cbfd5-nibble-v4:$NIBBLE_PROJECT_ROOT/../cado/tplane/src"
OUT="$NIBBLE_PROJECT_ROOT/state/score-centering-$SLURM_JOB_ID.json"
MODEL="$NIBBLE_PROJECT_ROOT/../hf_cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
"$NIBBLE_PYTHON" -m nibble.score_centering_benchmark "$MODEL" "$OUT" --prefixes "$NIBBLE_PROJECT_ROOT/../areal-kv/results/calibration-v0.jsonl"
