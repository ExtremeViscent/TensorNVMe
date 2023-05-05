#!/bin/bash

module load cuda-11.7.1-gcc-12.1.0-v745rvm

gdsio -D /data -d 1 -w 8 -s 1G -i 16M -x 0 -I 1 -T 30

# 0 - Storage->GPU (GDS)
# 1 - Storage->CPU
# 2 - Storage->CPU->GPU
# 3 - Storage->CPU->GPU_ASYNC
# 4 - Storage->PAGE_CACHE->CPU->GPU
# 5 - Storage->GPU_ASYNC
# 6 - Storage->GPU_BATCH

# WRITE XferType: GPU_BATCH Threads: 1 IoDepth: 8 DataSetSize: 524206080/8388608(KiB) IOSize: 16384(KiB) Throughput: 1.664916 GiB/sec, Avg_Latency: 9383.000000 usecs ops: 31995 total_time 300.268538 secs
# IoType: WRITE XferType: CPU_GPU Threads: 8 DataSetSize: 97058816/8388608(KiB) IOSize: 16384(KiB) Throughput: 3.067191 GiB/sec, Avg_Latency: 40780.800505 usecs ops: 5924 total_time 30.178267 secs
# IoType: WRITE XferType: GPU_BATCH Threads: 1 IoDepth: 8 DataSetSize: 92323840/8388608(KiB) IOSize: 16384(KiB) Throughput: 2.993768 GiB/sec, Avg_Latency: 5218.000000 usecs ops: 5635 total_time 29.410055 secs