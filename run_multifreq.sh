#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

PYTHON_BIN=${PYTHON:-python}

DATA_DIR="$SCRIPT_DIR/data/testcases"
RESULTS_DIR="$SCRIPT_DIR/results_multifreq"
TMP_DIR="$SCRIPT_DIR/.tmp"
MEAS_FILE="$TMP_DIR/multifreq_measurement.npy"

mkdir -p "$RESULTS_DIR" "$TMP_DIR"

CASES=("epsilon_austria" "epsilon_twinCircle")
FREQ="3,4,5,"
EXP_PREFIX="simul"
METHODS=("fd-isp" "pdtot-isp")

for case in "${CASES[@]}"; do
  "$PYTHON_BIN" -m deepcsi.generate_measurement \
    --expname "measurement" \
    --basedir "$TMP_DIR" \
    --params_path "$DATA_DIR/${case}.npy" \
    --output "$MEAS_FILE" \
    --freq "$FREQ" \
    --grid_num 96 \
    --N_inc 16 \
    --N_rec 32


  for method in "${METHODS[@]}"; do
    EXP_NAME="${EXP_PREFIX}_${case}_${method}"
    "$PYTHON_BIN" -m deepcsi.main \
      --expname "$EXP_NAME" \
      --basedir "$RESULTS_DIR" \
      --params_path "$DATA_DIR/${case}.npy" \
      --recdata_path "$MEAS_FILE" \
      --method "$method" \
      --freq "$FREQ" \
      --grid_num 64 \
      --N_inc 16 \
      --N_rec 32 \
      --max_iter 1000 \
      --netdepth 16 \
      --netwidth 256 \
      --multires 20 \
      --lrate 5e-2 \
      --params_lrate 1e-1 \
      --regularizer_weight 0.1 \
      --regularizer_decay 0.5 \
      --i_print 100 \
      --i_testset 100 \
      --J_network multi-mlp \
      --result_file "$RESULTS_DIR/result.csv"
  done
done
