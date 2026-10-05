#!/bin/bash
#SBATCH --job-name=nibble-compare
#SBATCH --account=arora
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/ARORA/hh9077/nibble/work/slurm-%j.out
set -euo pipefail
: "${NIBBLE_PROJECT_ROOT:?submit with NIBBLE_PROJECT_ROOT set}"
source "$NIBBLE_PROJECT_ROOT/work/env.sh"
export NIBBLE_PROJECT_ROOT
export HF_DATASETS_CACHE="$NIBBLE_PROJECT_ROOT/../hf_cache/datasets"
export NIBBLE_SPLITS="$NIBBLE_PROJECT_ROOT/work/data/splits.json"
export NIBBLE_CALIBRATION="$NIBBLE_PROJECT_ROOT/work/data/calibration-qwen25-1p5b.jsonl"
export NIBBLE_CACHE_ROOT="$NIBBLE_PROJECT_ROOT/work/cache/areal"
export PYTHONPATH="${NIBBLE_CODE_ROOT:-$NIBBLE_PROJECT_ROOT}/src:$NIBBLE_PROJECT_ROOT/work/deps/areal-040cbfd5:$NIBBLE_PROJECT_ROOT/../cado/tplane/src"
export NIBBLE_RUNNER_CONFIG="$NIBBLE_PROJECT_ROOT/work/runner-$SLURM_JOB_ID.json"
export PATH="$(dirname "$NIBBLE_PYTHON"):$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
"$NIBBLE_PYTHON" - <<'CONFIG'
import json,os
from pathlib import Path
root=Path(os.environ['NIBBLE_PROJECT_ROOT'])
config=json.loads((root/'work/runner.json').read_text())
visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
if len(visible)!=4 or not all(x.isdigit() for x in visible):
    raise RuntimeError('expected four numeric CUDA_VISIBLE_DEVICES from Slurm; refusing to guess allocation')
config['gpu_ids']=visible
config['areal_root']=str(root/'work/deps/areal-040cbfd5')
config['models']['qwen25-1.5b']=str(root.parent/'hf_cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306')
config['datasets']['gsm8k-benchmark']='nibble.native:benchmark_dataset'
config['rewards']['gsm8k-strict']='nibble.native:strict_gsm8k_reward'
Path(os.environ['NIBBLE_RUNNER_CONFIG']).write_text(json.dumps(config,indent=2)+'\n')
CONFIG
"$NIBBLE_PYTHON" "${NIBBLE_CODE_ROOT:-$NIBBLE_PROJECT_ROOT}/tools/apply_areal_patch.py" "$NIBBLE_PROJECT_ROOT/work/deps/areal-040cbfd5" --verify
CAMPAIGN="$NIBBLE_PROJECT_ROOT/state/campaign-$SLURM_JOB_ID"
trap 'test ! -f "$CAMPAIGN/campaign.json" || "$NIBBLE_PYTHON" -m nibble.campaign "$CAMPAIGN" --summarize' EXIT
cd "$NIBBLE_PROJECT_ROOT"
"$NIBBLE_PYTHON" -m nibble.campaign "$CAMPAIGN" --max-steps "${NIBBLE_MAX_STEPS:-15}"
