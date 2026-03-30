# !/bin/bash

python infer.py --config=./config/globalscale/final_sam.yml \
    --ckpt=pretrained_pth/globalscale_sam.pth \
    --OOD=$1    # 是否进行OOD推理