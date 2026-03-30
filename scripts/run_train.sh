#!/bin/bash

python train.py --config='./config/cityscale/final_sam.yml' \
    --n_gpus 4  \
    # --dev_run \
    # --resume --ckpt=./DeH4R/02rofa1m/checkpoints/epoch=14-step=37500.ckpt \
    # --run_id=02rofa1m
        

    