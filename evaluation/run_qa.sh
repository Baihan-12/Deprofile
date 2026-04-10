#!/bin/bash
set -e
export KMP_DUPLICATE_LIB_OK=TRUE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_JSON="$REPO_ROOT/data/main_select_profiles_2.json"

BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
API_KEY="${OPENAI_API_KEY:?请设置环境变量 OPENAI_API_KEY}"
NUM_WORKERS="${NUM_WORKERS:-5}"
MAX_TOKENS="${MAX_TOKENS:-2048}"

cd "$SCRIPT_DIR"

run_deepseek() {
  RUN_NAME="$1"
  QUESTIONS_PATH="$2"
  MODEL="$3"
  echo "=== Qwen run: ${RUN_NAME} (${QUESTIONS_PATH}) ==="
  for BASELINE in G1.5 G2.5; do
    python api.py \
      --data_path "${DATA_JSON}" \
      --questions_path "${QUESTIONS_PATH}" \
      --base_url "${BASE_URL}" \
      --api_key "${API_KEY}" \
      --model "${MODEL}" \
      --baseline "${BASELINE}" \
      --run_name "${RUN_NAME}" \
      --language zh \
      --num_workers "${NUM_WORKERS}" \
      --max_tokens "${MAX_TOKENS}" \
      --skip_existing
  done
}

run_deepseek "basic_info" "./questions/QA_basic_info.json" "gpt-4o-mini"
run_deepseek "syp1" "./questions/QA_symp_1.json" "gpt-4o-mini"
run_deepseek "syp2" "./questions/QA_symp_2.json" "gpt-4o-mini"
run_deepseek "syp3" "./questions/QA_symp_3.json" "gpt-4o-mini"
run_deepseek "timeline" "./questions/QA_timeline.json" "gpt-4o-mini"
