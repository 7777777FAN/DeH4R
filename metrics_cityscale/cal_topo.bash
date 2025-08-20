#!/bin/bash

dir=$1
trace=$2
python ./topo/main.py -savedir $dir -trace $trace
python topo.py -savedir $dir 