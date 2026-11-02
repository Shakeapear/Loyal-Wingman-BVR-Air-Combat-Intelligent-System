#!/usr/bin/env bash
# Linux 服务器训练入口（P0/P1 打击任务）。
#
# 用法：
#   bash 多任务智能体/scripts/train_strike.sh
#   POLICY=mlp TIMESTEPS=1000000 N_ENVS=8 OUT_DIR=models/p0_strike_1m \
#       bash 多任务智能体/scripts/train_strike.sh
#   OVERWRITE=1 ...   # 允许在已有输出目录上从头重跑（默认拒绝，防止误覆盖）
#
# 所有路径都从脚本位置推导，不依赖当前工作目录；需要覆盖时用环境变量。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$LIB_DIR/.." && pwd)"

cd "$LIB_DIR"

export PYTHONPATH="$LIB_DIR${PYTHONPATH:+:$PYTHONPATH}"
export BVR_JSBSIM_ROOT="${BVR_JSBSIM_ROOT:-$REPO_DIR/JSBSim}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

PYTHON="${PYTHON:-python}"
POLICY="${POLICY:-hybrid}"
TIMESTEPS="${TIMESTEPS:-1000000}"
N_ENVS="${N_ENVS:-8}"
N_STEPS="${N_STEPS:-256}"
BATCH_SIZE="${BATCH_SIZE:-64}"
N_EPOCHS="${N_EPOCHS:-10}"
SEED="${SEED:-0}"
SCENE="${SCENE:-bvr80-120}"
VEC_ENV="${VEC_ENV:-subproc}"
DEVICE="${DEVICE:-cpu}"
EVAL_SEEDS="${EVAL_SEEDS:-1-20}"
OUT_DIR="${OUT_DIR:-${BVR_MODEL_DIR:-$LIB_DIR/models}/${POLICY}_strike_${TIMESTEPS}}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
OVERWRITE="${OVERWRITE:-}"

echo "[预检] python=$PYTHON policy=$POLICY timesteps=$TIMESTEPS n_envs=$N_ENVS"
echo "[输出] $OUT_DIR"
"$PYTHON" -X utf8 validation/preflight_server.py --run-smoke

# EXTRA_ARGS 保持字符串以便按空格拆分；不要在其中放含空格的路径。
# shellcheck disable=SC2086
"$PYTHON" -X utf8 -m agents.ppo_agent \
  --task strike \
  --policy "$POLICY" \
  --timesteps "$TIMESTEPS" \
  --n-envs "$N_ENVS" \
  --n-steps "$N_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --n-epochs "$N_EPOCHS" \
  --seed "$SEED" \
  --scene "$SCENE" \
  --vec-env "$VEC_ENV" \
  --device "$DEVICE" \
  --out-dir "$OUT_DIR" \
  --eval-freq-episodes "${EVAL_FREQ_EPISODES:-500}" \
  --checkpoint-episodes "${CHECKPOINT_EPISODES:-1000}" \
  --eval-seeds "$EVAL_SEEDS" \
  --tensorboard \
  --tb-log "$OUT_DIR/tensorboard" \
  ${FINAL_EVAL:+--final-eval} \
  ${OVERWRITE:+--overwrite} \
  $EXTRA_ARGS
