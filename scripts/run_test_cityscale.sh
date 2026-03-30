# !/bin/bash

CUDA_VISIBLE_DEVICES=1 python infer.py --config=./config/cityscale/final_sam2.yml \
    --ckpt=pretrained_pth/cityscale_sam2.pth