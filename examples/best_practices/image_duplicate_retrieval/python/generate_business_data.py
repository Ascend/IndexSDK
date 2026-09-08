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

from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
BASE_PATH = PROJECT_DIR / "tmp/base.fvecs"
QUERY_PATH = PROJECT_DIR / "tmp/query.fvecs"
GROUND_TRUTH_PATH = PROJECT_DIR / "tmp/groundtruth.ivecs"
DIM = 128
NTOTAL = 100_000_000
NQUERY = 1_000_000
TOPK = 100
GROUP_SIZE = TOPK
GROUP_BLOCK = 1_000
CODE_BITS = 20
BIT_REPEAT = 6
DIRECTION_INDICES = np.linspace(0, DIM - 1, GROUP_SIZE, dtype=np.int64)
PERTURBATIONS = np.linspace(0.03, 0.30, GROUP_SIZE, dtype=np.float32)


def write_fvecs(output, vectors: np.ndarray) -> None:
    record_type = np.dtype([("dimension", "<i4"), ("values", "<f4", (vectors.shape[1],))])
    records = np.empty(vectors.shape[0], dtype=record_type)
    records["dimension"] = vectors.shape[1]
    records["values"] = vectors
    records.tofile(output)


def write_ivecs(output, labels: np.ndarray) -> None:
    records = np.empty((labels.shape[0], labels.shape[1] + 1), dtype="<i4")
    records[:, 0] = labels.shape[1]
    records[:, 1:] = labels
    records.tofile(output)


def make_centers(start: int, count: int) -> np.ndarray:
    group_ids = np.arange(start, start + count, dtype=np.uint32)
    bit_positions = np.arange(CODE_BITS, dtype=np.uint32)
    bits = ((group_ids[:, None] >> bit_positions) & 1).astype(np.float32)

    centers = np.ones((count, DIM), dtype=np.float32)
    centers[:, : CODE_BITS * BIT_REPEAT] = np.repeat(bits * 2.0 - 1.0, BIT_REPEAT, axis=1)
    centers /= np.sqrt(np.float32(DIM))
    return centers


def make_database_vectors(centers: np.ndarray) -> np.ndarray:
    # 20 bit 只用于编码不同组的中心；从全部 128 个坐标维度中均匀选取 100 个扰动方向。
    noise = -centers[:, DIRECTION_INDICES, None] * centers[:, None, :]
    indices = np.arange(GROUP_SIZE)
    noise[:, indices, DIRECTION_INDICES] += 1.0
    noise /= np.sqrt(np.float32(1.0 - 1.0 / DIM))

    perturbations = PERTURBATIONS[None, :, None]
    vectors = centers[:, None, :] + perturbations * noise
    vectors /= np.sqrt(np.float32(1.0) + perturbations**2)
    return vectors.reshape(-1, DIM)


def main() -> None:
    BASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if NTOTAL != NQUERY * GROUP_SIZE:
        raise ValueError("database size must equal query count times TopK")

    with (
        BASE_PATH.open("wb") as base_output,
        QUERY_PATH.open("wb") as query_output,
        GROUND_TRUTH_PATH.open("wb") as ground_truth_output,
    ):
        for start in range(0, NQUERY, GROUP_BLOCK):
            count = min(GROUP_BLOCK, NQUERY - start)
            centers = make_centers(start, count)
            write_fvecs(query_output, centers)
            write_fvecs(base_output, make_database_vectors(centers))

            labels = np.arange(start * GROUP_SIZE, (start + count) * GROUP_SIZE, dtype=np.int32)
            write_ivecs(ground_truth_output, labels.reshape(count, GROUP_SIZE))


if __name__ == "__main__":
    main()
