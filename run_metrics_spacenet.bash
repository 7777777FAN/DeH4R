#!/bin/bash

dir="$1"
trace=$(yq eval '.INFER.TRACE' ./config/spaceet/spacenet_sam2.yml)

cd ./metrics_spacenet

./cal_pixelscore.bash $dir $trace
./cal_apls.bash $dir $trace
./cal_topo.bash $dir $trace


