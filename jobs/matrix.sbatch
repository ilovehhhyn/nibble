#!/bin/bash
#SBATCH --job-name=nibble-matrix
#SBATCH --account=griffith
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --constraint=gpu80
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/gpfs/ARORA/hh9077/nibble/work/slurm-%j.out
set -euo pipefail
: "${NIBBLE_PROJECT_ROOT:?}" "${NIBBLE_CODE_ROOT:?}" "${NIBBLE_ARM:?}" "${NIBBLE_ALGORITHM:?}"
source "$NIBBLE_PROJECT_ROOT/work/env.sh"
export HF_DATASETS_CACHE="$NIBBLE_PROJECT_ROOT/../hf_cache/datasets"
export PYTHONPATH="$NIBBLE_CODE_ROOT/src:$NIBBLE_PROJECT_ROOT/work/deps/areal-040cbfd5-nibble-v5:$NIBBLE_PROJECT_ROOT/../cado/tplane/src"
"$NIBBLE_PYTHON" -m nibble.cuda_build "$NIBBLE_PROJECT_ROOT" "$NIBBLE_PROJECT_ROOT/work/cuda-build-$SLURM_JOB_ID.sh"
source "$NIBBLE_PROJECT_ROOT/work/cuda-build-$SLURM_JOB_ID.sh"
export NIBBLE_RUNNER_CONFIG="$NIBBLE_PROJECT_ROOT/work/runner-$SLURM_JOB_ID.json"
export PATH="$(dirname "$NIBBLE_PYTHON"):$PATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
"$NIBBLE_PYTHON" - <<'CONFIG'
import json,os
from pathlib import Path
root=Path(os.environ['NIBBLE_PROJECT_ROOT'])
config=json.loads((root/'work/runner.json').read_text())
visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
expected=4 if os.environ['NIBBLE_ARM'] in ('api-map','api-map-sc') else 3
if len(visible)!=expected or not all(x.isdigit() for x in visible):
    raise RuntimeError('unexpected Slurm GPU identifiers')
config['gpu_ids']=visible
config['areal_root']=str(root/'work/deps/areal-040cbfd5-nibble-v5')
config['models']['qwen25-1.5b']=str(root.parent/'hf_cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306')
Path(os.environ['NIBBLE_RUNNER_CONFIG']).write_text(json.dumps(config,indent=2)+'\n')
CONFIG
"$NIBBLE_PYTHON" "$NIBBLE_CODE_ROOT/tools/apply_areal_patch.py" "$NIBBLE_PROJECT_ROOT/work/deps/areal-040cbfd5-nibble-v5" --verify
cd "$NIBBLE_PROJECT_ROOT"
"$NIBBLE_PYTHON" -m nibble.fork_campaign "$NIBBLE_PROJECT_ROOT/state/comparison-$SLURM_JOB_ID" --arm "$NIBBLE_ARM" --algorithm "$NIBBLE_ALGORITHM"
