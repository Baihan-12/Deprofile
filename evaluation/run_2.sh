#!/bin/bash
# 示例：多次调用 direct_api.py。需事先 export OPENAI_API_KEY。

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_JSON="$REPO_ROOT/data/main_select_profiles_2.json"
QA_JSON="$REPO_ROOT/evaluation/QA.json"

: "${OPENAI_API_KEY:?请设置 OPENAI_API_KEY}"
BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
MODEL_NAME="${MODEL_NAME:-deepseek-v3}"

cd "$SCRIPT_DIR"

run_direct() {
  local baseline="$1"
  local run_name="$2"
  local model="${3:-$MODEL_NAME}"
  python direct_api.py \
    --data_path "$DATA_JSON" \
    --questions_path "$QA_JSON" \
    --model "$model" \
    --base_url "$BASE_URL" \
    --baseline "$baseline" \
    --run_name "$run_name" \
    --language zh \
    --num_workers 16 \
    --skip_existing
}

run_direct G0 baseqa
run_direct G1 test_run
run_direct G2-1 test_run
run_direct G2-2 test_run
run_direct G3-1 test_run
run_direct G4-1 test_run
run_direct G6-1 test_run "gpt-4.1"
run_direct G6-2 test_run
