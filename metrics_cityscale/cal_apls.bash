#!/bin/bash

declare -a arr=(8 9 19 28 29 39 48 49 59 68 69 79 88 89 99 108 109 119 128 129 139 148 149 159 168 169 179)

# source directory, 相对路径
dir="$1"
trace="$2"
echo "预测结果保存路径：${dir}"
mkdir -p ../$dir/results/apls


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
# for i in "${arr[@]}"   
# do
#     if test -f "../${dir}/decode_result/${i}_graph_${suffix}_tracing.p"; then

#         echo "========================Processing $i======================"
#         python ./apls/convert.py    "../data/cityscale/20cities/region_${i}_graph_gt.pickle"    "${i}_gt.json"
#         python ./apls/convert.py    "../${dir}/decode_result/${i}_graph_${suffix}_tracing.p"    "${i}_prop.json"

#         go run ./apls/main.go       "./temp/${i}_gt.json"     "./temp/${i}_prop.json"     "../$dir/results/apls/$i.txt"
#     fi
# done


# 最大并发任务数
max_jobs=10  # 根据系统性能调整

# 动态任务分配
function run_task() {
    local i="$1"  # 接收任务参数
    echo "========================Processing $i======================"

    # 执行任务
    python ./apls/convert.py "../data/cityscale/20cities/region_${i}_graph_gt.pickle" "${i}_gt.json" || { echo "convert.py failed for $i"; exit 1; }
    python ./apls/convert.py "../${dir}/decode_result/${i}_graph_${suffix}_tracing.p" "${i}_prop.json" || { echo "convert.py failed for $i"; exit 1; }
    go run ./apls/main.go "./temp/${i}_gt.json" "./temp/${i}_prop.json" "../$dir/results/apls/$i.txt" || { echo "go run failed for $i"; exit 1; }

    echo "========================Finished $i========================"
}

# 循环任务并动态分配
for i in "${arr[@]}"; do
    if test -f "../${dir}/decode_result/${i}_graph_${suffix}_tracing.p"; then
        # 检查当前后台任务数，动态控制任务分配
        while [[ $(jobs -r | wc -l) -ge $max_jobs ]]; do
            sleep 1  # 等待有空闲任务槽
        done

        # 分配任务到后台执行
        run_task "$i" &
    fi
done

# 等待所有后台任务完成
wait

echo "All tasks completed. Now running final python script."


python apls.py --dir $dir
