#!/bin/sh

# 查找所有打开 /dev/nvidia* 的进程
lsof /dev/nvidia* | awk '{print $2}' | grep -v PID | sort -u | while read pid; do
    # 获取进程名
    process_name=$(ps -p $pid -o comm=)

    # 检查进程名是否为 python3.10
    if [ "$process_name" = "python3.10" ]; then
        # 杀死该进程
        echo "Killing process $pid ($process_name)"
        kill -9 $pid
    fi
done
