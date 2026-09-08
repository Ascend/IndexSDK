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

# pylint: disable=duplicate-code
import csv
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
TOPK = 100
QUERY_COUNT = 1_000_000
MIN_QPS = 1_000_000 / (24 * 60 * 60)


def read_ivecs(path: Path) -> np.ndarray:
    if path.stat().st_size != QUERY_COUNT * (TOPK + 1) * np.dtype("<i4").itemsize:
        raise ValueError(f"unexpected result shape: {path}")
    records = np.memmap(path, dtype="<i4", mode="r", shape=(QUERY_COUNT, TOPK + 1))
    if not np.all(records[:, 0] == TOPK):
        raise ValueError(f"unexpected TopK: {path}")
    return records[:, 1:]


def validate_ids(labels: np.ndarray, count: int, path: Path) -> None:
    for start in range(0, labels.shape[0], 4096):
        block = labels[start : start + 4096]
        if np.any(block < 0) or np.any(block >= count):
            raise ValueError(f"result ID out of range: {path}")
        if np.any(np.diff(np.sort(block, axis=1), axis=1) == 0):
            raise ValueError(f"duplicate result ID: {path}")


def recall_at_100(output: np.ndarray, ground_truth: np.ndarray) -> float:
    matches = 0
    for start in range(0, output.shape[0], 256):
        left = output[start : start + 256, :TOPK]
        right = ground_truth[start : start + 256, :TOPK]
        matches += int((left[:, :, None] == right[:, None, :]).any(axis=2).sum())
    return matches / (output.shape[0] * TOPK)


def main() -> None:
    results = PROJECT_DIR / "tmp/business-results"
    with (results / "performance.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 1 or (rows[0]["algorithm"], int(rows[0]["n"]), int(rows[0]["device_count"])) != (
        "flat",
        100_000_000,
        1,
    ):
        raise ValueError("unexpected 100M Demo plan")
    row = rows[0]
    output = read_ivecs(results / "output.ivecs")
    ground_truth = read_ivecs(PROJECT_DIR / "tmp/groundtruth.ivecs")
    count = int(row["n"])
    validate_ids(output, count, results / "output.ivecs")
    validate_ids(ground_truth, count, PROJECT_DIR / "tmp/groundtruth.ivecs")
    recall = recall_at_100(output, ground_truth)

    size = f"{count // 1_000_000}M"
    print(f"{'algorithm':<12}{'N':<8}{'devices':>9}{'QPS':>10}{'R@100':>12}")
    print(f"{row['algorithm']:<12}{size:<8}{int(row['device_count']):>9}{float(row['qps']):>10.1f}{recall:>12.4f}")
    if recall < 0.95 or float(row["qps"]) < MIN_QPS:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
