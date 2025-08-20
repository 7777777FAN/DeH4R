#!/bin/bash

declare -a arr=( $(jq -r '.test[]' ../data/spacenet/data_split.json) )

# source directory
dir=$1
trace="$2"

data_dir='spacenet'
mkdir -p ../$dir/results/apls
echo $dir

suffix="no"
if [[ "$trace" == "True" ]]; then
    suffix="with"
    echo "The tracing algorithm is enabnled, so we compute the metrics with tracing parts added"
fi


# delete exsited temp dir to make sure store new conerted pickle file
temp_dir="./temp"
if [ -e "$temp_dir" ]; then
   rm -rf "$temp_dir"
fi


# now loop through the above array
for i in "${arr[@]}"   
do
    # gt_graph=${i}__gt_graph_dense_spacenet.p
    if test -f "../${dir}/decode_result/${i}_graph_${suffix}_tracing.p"; then
        echo "========================Processing $i======================"
        python ./apls/convert.py   "../data/${data_dir}/RGB_1.0_meter/${i}__gt_graph_dense.p"    "${i}_gt.json"
        python ./apls/convert.py   "../${dir}/decode_result/${i}_graph_${suffix}_tracing.p"      "${i}_prop.json"
        
        go run ./apls/main.go      "./temp/${i}_gt.json"     "./temp/${i}_prop.json"     ../$dir/results/apls/$i.txt     spacenet
    fi
done


python apls.py --dir $dir
