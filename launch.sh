#!/usr/bin/env bash
set -euo pipefail

optimizers=('sgd' 'sgd_with_momentum')

for optimizer in ${optimizers[@]}; do
  python train.py --optimizer $optimizer --save_dir "out/$optimizer"
done
