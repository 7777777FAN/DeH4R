#!/bin/bash

dir="$1"
OOD="$2"
trace=$(yq eval '.INFER.TRACE' ./config/globalscale/final_sam.yml)

cd ./metrics_globalscale

./cal_pixelscore.bash $dir $trace $OOD
./cal_apls.bash $dir $trace $OOD
./cal_topo.bash $dir $trace $OOD



