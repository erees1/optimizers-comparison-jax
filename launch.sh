#!/usr/bin/env bash
set -euo pipefail

# Add cuda to path
export PATH=/usr/local/cuda-11.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

optimizers=('adam' 'rmsprop_momentum' 'rmsprop' 'sgd' 'sgd_momentum')
datasets=('mnist' 'cifar')
models=('fcn' 'cnn')
epochs=40


for ds in ${datasets[@]}; do
  for model in ${models[@]}; do
    for optimizer in ${optimizers[@]}; do
      python train.py --optimizer $optimizer --ds $ds --epochs $epochs  --save_dir "out/$ds/$model/$optimizer" --model $model
    done
  done
done
