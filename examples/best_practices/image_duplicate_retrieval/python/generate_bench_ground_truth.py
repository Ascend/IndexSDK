#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------
This file is part of the IndexSDK project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

IndexSDK is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

# pylint: disable=no-value-for-parameter
import concurrent.futures
import os
from pathlib import Path

import faiss
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "tmp/sift100m-prepared"
BASE_SIZES = (1_000_000, 10_000_000, 30_000_000, 40_000_000, 70_000_000, 100_000_000)
DIMENSION = 128
QUERY_COUNT = 10_000
TOPK = 100
ADD_BLOCK = 100_000
QUERY_WORKERS = os.cpu_count()


def read_fvecs(path: Path) -> np.ndarray:
    record_type = np.dtype([("dimension", "<i4"), ("values", "<f4", (DIMENSION,))])
    return np.memmap(path, dtype=record_type, mode="r")["values"]


def ground_truth_path(count: int) -> Path:
    return DATA_DIR / f"sift_groundtruth_{count // 1_000_000}m.ivecs"


def write_ivecs(path: Path, labels: np.ndarray) -> None:
    records = np.empty((labels.shape[0], labels.shape[1] + 1), dtype="<i4")
    records[:, 0] = labels.shape[1]
    records[:, 1:] = labels
    records.tofile(path)


def merge_topk(
    previous_distances: np.ndarray,
    previous_labels: np.ndarray,
    new_distances: np.ndarray,
    new_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    distances = np.concatenate((previous_distances, new_distances), axis=1)
    labels = np.concatenate((previous_labels, new_labels), axis=1)
    selected = np.argpartition(distances, -TOPK, axis=1)[:, -TOPK:]
    selected_distances = np.take_along_axis(distances, selected, axis=1)
    # 向量相似度的 metric 使用 内积（IP）距离，越大越好
    order = np.argsort(-selected_distances, axis=1)
    selected = np.take_along_axis(selected, order, axis=1)
    return (
        np.take_along_axis(distances, selected, axis=1),
        np.take_along_axis(labels, selected, axis=1),
    )


def search_exact(index: faiss.IndexFlatIP, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    chunks = [chunk for chunk in np.array_split(queries, QUERY_WORKERS) if len(chunk)]

    def search_chunk(chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        faiss.omp_set_num_threads(1)
        return index.search(chunk, TOPK)

    with concurrent.futures.ThreadPoolExecutor(max_workers=QUERY_WORKERS) as pool:
        results = list(pool.map(search_chunk, chunks))
    return (
        np.concatenate([distances for distances, _ in results]),
        np.concatenate([labels for _, labels in results]),
    )


def main() -> None:
    base = read_fvecs(DATA_DIR / "sift_base.fvecs")
    queries = np.ascontiguousarray(read_fvecs(DATA_DIR / "sift_query.fvecs"))
    faiss.omp_set_num_threads(1)
    best_distances = np.empty((QUERY_COUNT, 0), dtype=np.float32)
    best_labels = np.empty((QUERY_COUNT, 0), dtype=np.int64)
    previous = 0
    for count in BASE_SIZES:
        index = faiss.IndexFlatIP(DIMENSION)
        for start in range(previous, count, ADD_BLOCK):
            block = np.ascontiguousarray(base[start : min(start + ADD_BLOCK, count)])
            index.add(block)
        distances, labels = search_exact(index, queries)
        labels += previous
        best_distances, best_labels = merge_topk(best_distances, best_labels, distances, labels)
        output = ground_truth_path(count)
        temporary = output.with_suffix(output.suffix + ".part")
        write_ivecs(temporary, best_labels)
        temporary.replace(output)
        previous = count


if __name__ == "__main__":
    main()
