#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

PY="${PY:-$HOME/miniconda3/envs/mini_ml/bin/python}"
take_output() { sed -n 's/^CSV output: //p'; }

echo "== 1/5 round 2D data =="
DATA=$(./build/lab2_gen --dim 2 --samples 80 --sigma 0.4 --elong 1 --min -2 --max 2 --seed 42 | take_output)
echo "data: $DATA"

echo "== 2/5 train sgd / cg / newton =="
SGD_RUN=$(./build/lab2_logistic --data "$DATA/samples.csv" --optimizer sgd --epochs 300 --learning-rate 0.1 --weight-decay 0.01 --save-every 1 --seed 42 | take_output)
CG_RUN=$(./build/lab2_logistic --data "$DATA/samples.csv" --optimizer cg --epochs 60 --weight-decay 0.01 --save-every 1 --seed 42 | take_output)
NEWTON_RUN=$(./build/lab2_logistic --data "$DATA/samples.csv" --optimizer newton --epochs 30 --weight-decay 0.01 --save-every 1 --seed 42 | take_output)
echo "sgd: $SGD_RUN"
echo "cg: $CG_RUN"
echo "newton: $NEWTON_RUN"

echo "== 3/5 2D boundary / sgd loss / sigmoid surface / optimizer compare =="
rm -f lab/lab2/output/optimizer_compare_loss.png lab/lab2/output/optimizer_compare_boundary.png
"$PY" lab/lab2/plot2d.py --data-dir "$DATA" --run-dir "$SGD_RUN"
"$PY" lab/lab2/plot3d.py --data-dir "$DATA" --run-dir "$SGD_RUN"
"$PY" lab/lab2/compare_optimizers.py --data-dir "$DATA" --run-dir "$SGD_RUN" "$CG_RUN" "$NEWTON_RUN"

echo "== 4/5 3D data + decision plane =="
DATA3=$(./build/lab2_gen --dim 3 --samples 80 --sigma 0.4 --elong 2 --min -2 --max 2 --seed 11 | take_output)
RUN3=$(./build/lab2_logistic --data "$DATA3/samples.csv" --optimizer sgd --epochs 300 --learning-rate 0.1 --weight-decay 0.01 --save-every 1 --seed 11 | take_output)
echo "data: $DATA3 run: $RUN3"
"$PY" lab/lab2/plot3d.py --data-dir "$DATA3" --run-dir "$RUN3"

echo "== 5/5 naive bayes comparison (elongated 2D) =="
DATA_NB=$(./build/lab2_gen --dim 2 --samples 60 --sigma 0.3 --elong 5 --min -1 --max 1 --seed 7 | take_output)
LR_RUN=$(./build/lab2_logistic --data "$DATA_NB/samples.csv" --optimizer sgd --epochs 300 --learning-rate 0.1 --weight-decay 0.01 --save-every 1 --seed 7 | take_output)
NB_RUN=$(./build/lab2_nb --data "$DATA_NB/samples.csv" --seed 7 | take_output)
rm -f lab/lab2/output/nb_vs_lr_boundary.png
"$PY" lab/lab2/nb_compare.py --data-dir "$DATA_NB" --lr-dir "$LR_RUN" --nb-dir "$NB_RUN"

echo "all figures done"
