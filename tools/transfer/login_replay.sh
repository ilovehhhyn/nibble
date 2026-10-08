#!/bin/bash
# Replay every policy transition of one AReaL run in the real server: fit the map for v_k -> v_{k+1}, then compare transfer with re-prefill.
# Usage: login_replay.sh <checkpoint default dir> <base model dir> <number of updates> <calibration prefixes> <out dir>
set -euo pipefail
source "$NIBBLE_PROJECT_ROOT/work/env.sh"
export PYTHONPATH="$NIBBLE_CODE_ROOT/src:$NIBBLE_PROJECT_ROOT/work/deps/areal-040cbfd5-nibble-v4:$NIBBLE_PROJECT_ROOT/../cado/tplane/src"
export HF_DATASETS_CACHE="$NIBBLE_PROJECT_ROOT/../hf_cache/datasets"
export PROJECT_ROOT="$NIBBLE_PROJECT_ROOT"
export OMP_NUM_THREADS=4 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?}"
CKPT=$1; BASE=$2; UPDATES=$3; CAL=$4; OUT=$5
COUNT=${COUNT:-64}; ABORT_AFTER=${ABORT_AFTER:-0.5}; SERVER_MEM_FRACTION=${SERVER_MEM_FRACTION:-0.45}; PORT=30133
T=$NIBBLE_CODE_ROOT/tools; R="$NIBBLE_PYTHON"
cd "$NIBBLE_PROJECT_ROOT/work/deps/areal-040cbfd5-nibble-v4"
mkdir -p $OUT; REG=$OUT/registry; mkdir -p $REG
ckpt_of() { if [ "$1" = 0 ]; then echo $BASE; else ls -d $CKPT/epoch*globalstep$(($1-1)) | head -1; fi; }
echo "replay start $(date -Is) updates $UPDATES"
for k in $(seq 0 $((UPDATES-1))); do
  [ -f $REG/v$k-v$((k+1))/mapper.json ] || $R $T/fit_mapper_hf.py --source $(ckpt_of $k) --target $(ckpt_of $((k+1))) --prefixes $CAL --limit 128 \
    --max-tokens 1024 --batch-size 16 --lambda 0.01 --stride 4 --selection same_layer --out $REG/v$k-v$((k+1)) --provenance "{\"transition\":\"v$k-v$((k+1))\"}" > $OUT/fit-v$k.log 2>&1
  grep -E "extract_caches|fit_total|load_models|hash_checkpoints" $REG/v$k-v$((k+1))/fit_report.json | tr -d "\n"; echo " <- fit v$k-v$((k+1))"
done
mkdir -p $OUT/dumps
SERVER=""; SERVER_STARTED=0; SERVER_GENERATION=0
start_server() {
  SERVER_GENERATION=$((SERVER_GENERATION+1))
  SGLANG_KV_RETAIN_DUMP_DIR=$OUT/dumps "$NIBBLE_PYTHON" -m sglang.launch_server --model-path $BASE --tokenizer-path $BASE --trust-remote-code --tp-size 1 --random-seed 1 \
    --skip-tokenizer-init --disable-radix-cache --attention-backend flashinfer --context-length 32768 --mem-fraction-static $SERVER_MEM_FRACTION \
    --chunked-prefill-size -1 --dtype bfloat16 --log-level warning --host 127.0.0.1 --port $PORT \
    --kv-retain-on-abort --kv-migration-mode map --kv-max-migration-depth -1 --kv-mapper-module areal.experimental.kvmap.sglang_bridge \
    --kv-mapper-registry $REG --kv-event-log $OUT/kv_events > $OUT/server-$SERVER_GENERATION.log 2>&1 &
  SERVER=$!
  for i in $(seq 1 90); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && break; sleep 5; done
  curl -sf http://127.0.0.1:$PORT/health >/dev/null || { echo "server did not start"; tail -30 $OUT/server-$SERVER_GENERATION.log; exit 1; }
  SERVER_STARTED=$(date +%s)
  # Warm up every new server on holdout prompts so one-time compilation and the first map load are not charged to either treatment.
  REPLAY_COUNT=8 REPLAY_MAX_NEW=128 replay 0 holdout 0 concurrent $OUT/warmup-$SERVER_GENERATION-concurrent || { tail -20 $OUT/warmup-$SERVER_GENERATION-concurrent.log; exit 1; }
  REPLAY_COUNT=4 REPLAY_MAX_NEW=128 replay 0 holdout 8 sequential $OUT/warmup-$SERVER_GENERATION-sequential || { tail -20 $OUT/warmup-$SERVER_GENERATION-sequential.log; exit 1; }
}
stop_server() { [ -n "$SERVER" ] && { kill $SERVER 2>/dev/null || true; wait $SERVER 2>/dev/null || true; sleep 3; }; SERVER=""; }
# A login node kills processes after about ten minutes of heavy CPU use, and the server's scheduler spins; keep each instance young.
fresh_server() { if [ -z "$SERVER" ] || [ $(( $(date +%s) - SERVER_STARTED )) -gt ${SERVER_MAX_AGE_S:-300} ]; then stop_server; start_server; fi; }
trap stop_server EXIT
replay() { # k, split, offset, send, outdir, [extra server_replay arguments]
  local k=$1 split=$2 offset=$3 send=$4 out=$5; shift 5
  $R $T/transfer/server_replay.py --url http://127.0.0.1:$PORT --source $(ckpt_of $k) --target $(ckpt_of $((k+1))) --source-version $k --target-version $((k+1)) \
    --splits $PROJECT_ROOT/work/data/fork-splits.json --split $split --offset $offset --count ${REPLAY_COUNT:-$COUNT} --abort-after-s ${REPLAY_ABORT:-$ABORT_AFTER} --max-new-tokens ${REPLAY_MAX_NEW:-1024} --send $send --out-dir $out "$@" > $out.log 2>&1
}
for k in $(seq 0 $((UPDATES-1))); do
  send=concurrent; [ $((k % 2)) = 1 ] && send=sequential
  fresh_server
  replay $k holdout $(((k+2)*COUNT)) $send $OUT/v$k-v$((k+1)) || { echo "transition $k failed"; tail -15 $OUT/v$k-v$((k+1)).log; continue; }
  $R -c "import json;s=json.load(open('$OUT/v$k-v$((k+1))/summary.json'));print('v$k->v$((k+1))',s['send'],'interrupted',s['interrupted'],'used',s['transfer_used_cache'],'acc T/B',round(s.get('transfer_accuracy',-1),3),round(s.get('baseline_accuracy',-1),3),'ttft p50 T/B',round(s.get('transfer_ttft_ms',{}).get('p50',-1),1),round(s.get('baseline_ttft_ms',{}).get('p50',-1),1),'equal',round(s.get('continuation_equal_rate',-1),3))"
done
# Long-prefix latency sweep on the last transition: rollouts ignore EOS so prefixes grow with the abort time (synthetic; latency only).
LAST=$((UPDATES-1))
for spec in $(for seconds in ${LONG_SECONDS:-2 5 10 17}; do echo $seconds:sequential $seconds:concurrent; done); do
  seconds=${spec%%:*}; send=${spec##*:}
  fresh_server
  REPLAY_COUNT=8 REPLAY_ABORT=$seconds REPLAY_MAX_NEW=8192 replay $LAST holdout $(((UPDATES+3)*COUNT)) $send $OUT/long-${seconds}s-$send --ignore-eos --resume-new-tokens 8 \
    || { echo "long sweep $spec failed"; tail -8 $OUT/long-${seconds}s-$send.log; continue; }
  $R -c "import json;s=json.load(open('$OUT/long-${seconds}s-$send/summary.json'));print('long',$seconds,'s','$send','interrupted',s['interrupted'],'used',s['transfer_used_cache'],'prefix p50',s.get('prefix_tokens',{}).get('p50'),'ttft p50 transfer/baseline',round(s.get('transfer_ttft_ms',{}).get('p50',-1),1),round(s.get('baseline_ttft_ms',{}).get('p50',-1),1))"
done
stop_server
$R $T/transfer/aggregate_replay.py $OUT
# The map is fitted on Hugging Face caches and applied to the server pool: check the two agree for the base checkpoint.
$R $T/transfer/check_pool_layout.py --dump-dir $OUT/dumps --checkpoint $BASE --version 0 || echo "layout check could not run"
$R -c "
import glob, json, statistics
events = [json.loads(line) for path in glob.glob('$OUT/kv_events*.jsonl') for line in open(path)]
batches = [e for e in events if e['event'] == 'translate_batch']
print('translate batches:', len(batches))
for e in sorted(batches, key=lambda e: e['tokens']):
    print('  entries', e['entries'], 'tokens', e['tokens'], 'translate_ms', round(e['translate_ms'], 2), 'ms per 1k tokens', round(1000 * e['translate_ms'] / e['tokens'], 3))
"
echo "LOGIN_REPLAY_DONE $(date -Is) out $OUT"
