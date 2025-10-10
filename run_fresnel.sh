#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

PYTHON_BIN=${PYTHON:-python}

DATA_DIR="$SCRIPT_DIR/data/Fresnel"
RESULTS_DIR="$SCRIPT_DIR/results_fresnel"

mkdir -p "$RESULTS_DIR"

FREQ="3,4,5,"
METHODS=("fd-isp")
CASES=("FoamDielExtTM" "FoamDielIntTM" "FoamTwinDielTM")

for case in "${CASES[@]}"; do
  REC_PATH="$DATA_DIR/${case}.exp"
  for method in "${METHODS[@]}"; do
    EXP_NAME="fresnel_${case}_${method}"
    "$PYTHON_BIN" -m deepcsi.main \
      --expname "$EXP_NAME" \
      --basedir "$RESULTS_DIR" \
      --params_path "$DATA_DIR/${case}.npy" \
      --recdata_path "$REC_PATH" \
      --method "$method" \
      --freq "$FREQ" \
      --L_doi 0.16775 \
      --R_t 1.67 \
      --R_r 1.67 \
      --grid_num 64 \
      --N_inc 8 \
      --max_iter 1000 \
      --netdepth 16 \
      --netwidth 256 \
      --multires 20 \
      --lrate 5e-2 \
      --lrate_decay 2 \
      --params_lrate 1e-1 \
      --regularizer_weight 0.1 \
      --regularizer_decay 0.5 \
      --i_print 100 \
      --i_testset 100 \
      --J_network multi-mlp \
      --result_file "$RESULTS_DIR/result.csv"
  done
done
