#!/bin/bash

dir="$1"
trace=$(yq eval '.INFER.TRACE' ./config/cityscale/final_sam2.yml)

cd ./metrics_cityscale

./cal_pixelscore.bash $dir $trace 
./cal_apls.bash $dir $trace
./cal_topo.bash $dir $trace



