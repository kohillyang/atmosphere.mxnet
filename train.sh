#!/usr/bin/env bash
/home/kohill/anaconda3/bin/python3 trainval.py \
  --model resnet101_v1d --mode hybrid \
  --lr 0.0001 --lr-mode cosine --num-epochs 120 --batch-size 16 --num-gpus 4 -j 16 \
  --dtype float32 \
  --last-gamma --no-wd \
  --save-dir params_resnet50_v1_best \
  --logging-file resnet50_v1_best.log \
  --use-pretrained \
  --input-size 336