#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""
import numpy as np
import os
import threading

# 定义常量
topk = 10
bs = 111
dim = 768
ntotal = 10000000
ntotal_pad = (ntotal + 16 - 1) // 16 * 16

# 定义文件路径
bin_dir = "samples/build"

test_quantify = True


# 使用内存映射文件读取数据
def load_memmap_data(file_path, dtype, shape):
    try:
        data = np.memmap(file_path, dtype=dtype, mode='r')
        if data.size != np.prod(shape):
            raise ValueError(f"File size does not match expected shape {shape}")
        return data.reshape(shape)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except ValueError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"Error: Failed to load data from {file_path}. {e}")
        return None

# 加载特征数据
try:
    features_ori = load_memmap_data(os.path.join(bin_dir, "features_ori.bin"), np.float32, (ntotal, dim))
    if features_ori is None:
        raise ValueError("Failed to load features_ori")

    querys_ori = load_memmap_data(os.path.join(bin_dir, "querys_ori.bin"), np.float32, (bs, dim))
    if querys_ori is None:
        raise ValueError("Failed to load querys_ori")

    extra_score_fp16 = load_memmap_data(os.path.join(bin_dir, "extra_score_fp16.bin"), np.float16, (bs, ntotal_pad))
    if extra_score_fp16 is None:
        raise ValueError("Failed to load extra_score_fp16")

    dist_result = load_memmap_data(os.path.join(bin_dir, "dist_result.bin"), np.float32, (bs, topk))
    if dist_result is None:
        raise ValueError("Failed to load dist_result")

    if test_quantify:
        scale = load_memmap_data(os.path.join(bin_dir, "scale.npy"), np.float32, (1, dim))
        if scale is None:
            raise ValueError("Failed to load scale")
        scale_reciprocal = 1.0 / scale
        scale_reciprocal_fp16 = scale_reciprocal.astype(np.float16)

except Exception as e:
    print(f"Error: Failed to load data. {e}")
    exit(1)

# 将feature转换为float16或者int8保存，和npu计算逻辑一致
try:
    if test_quantify:
        # 量化，fp32特征乘以scale，限制到-128~127之间，保存成int8
        features_int8 = (features_ori * scale).clip(-128, 127).astype(np.int8)
        # 反量化，将int8转成fp16，乘以scale的倒数，保存成fp16
        features_fp16 = features_int8.astype(np.float16) * scale_reciprocal_fp16
    else:
        features_fp16 = features_ori.astype(np.float16)

    features_fp16_transpose = features_fp16.T
    querys_ori_fp16 = querys_ori.astype(np.float16)
except Exception as e:
    print(f"Error: Failed to convert data to float16. {e}")
    exit(1)

# 初始化结果矩阵
result = np.zeros((bs, ntotal), dtype=np.float16)

# 定义线程锁，用于确保线程安全
lock = threading.Lock()


def compute_row(i):
    try:
        # 计算第i行与features.T的点积
        row_result = np.dot(querys_ori_fp16[i], features_fp16_transpose)
        # 计算和extra score的均值
        row_result = (row_result + extra_score_fp16[i, 0:ntotal]) / 2.0
        # 按照降序排列
        row_result = np.sort(-row_result)
        # 使用锁确保写入结果时的线程安全
        with lock:
            result[i] = -row_result
    except Exception as e:
        print(f"Error in thread {i}: {e}")
        raise

# 创建并启动线程
thread_num = bs
threads = []
for i in range(thread_num):
    thread = threading.Thread(target=compute_row, args=(i,))
    threads.append(thread)
    thread.start()

# 等待所有线程完成
for thread in threads:
    thread.join()

final_result = result.astype(np.float32)


# 比较结果
def compare_results(result1, result2, topk, atol=1e-3, rtol=1e-3):
    try:
        equal_close = np.allclose(result1[:, :topk], result2[:, :topk], atol=atol, rtol=rtol)
        return equal_close
    except Exception as e:
        print(f"Error in result comparison: {e}")
        return (False, False)

try:
    are_equal_close = compare_results(dist_result, final_result, topk)
    print("two mat equal?  ", are_equal_close)
except Exception as e:
    print(f"Error: Failed to compare results. {e}")
    exit(1)
