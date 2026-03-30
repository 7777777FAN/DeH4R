#!/bin/bash

dir=$1
trace=$2
OOD=$3

if [ -z "$OOD" ]; then
    # 如果 OOD 参数为空，调用时不带 --OOD 参数
    python ./topo/main.py -savedir $dir -trace $trace 
else
    # 如果 OOD 参数存在，带上 --OOD 参数
    python ./topo/main.py -savedir $dir -trace $trace --OOD $OOD
fi

python topo.py -savedir $dir 