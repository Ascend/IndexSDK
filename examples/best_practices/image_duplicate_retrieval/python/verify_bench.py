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
PREPARED_DIR = PROJECT_DIR / "tmp/sift100m-prepared"
RESULT_DIR = PROJECT_DIR / "tmp/bench-results"
VARIANTS = ("flat", "int8flat", "ivfflat_single", "ivfflat_dual", "ivfrabitq")
BASE_SIZES = (1_000_000, 10_000_000, 30_000_000, 40_000_000, 70_000_000, 100_000_000)
QUERY_COUNT = 10_000
TOPK = 100
EXPECTED_RESULTS = (
    {(algorithm, count, 1) for algorithm in ("flat", "int8flat", "ivfrabitq") for count in BASE_SIZES}
    | {("ivfflat", count, 1) for count in BASE_SIZES[:-1]}
    | {("ivfflat", BASE_SIZES[-1], 2)}
)
HBM_PERCENTILES = (
    ("P70", "hbm_total_p70_bytes"),
    ("P80", "hbm_total_p80_bytes"),
    ("P90", "hbm_total_p90_bytes"),
    ("P99", "hbm_total_p99_bytes"),
    ("P100", "hbm_total_peak_bytes"),
)


def read_ivecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype="<i4")
    if raw.size != QUERY_COUNT * (TOPK + 1):
        raise ValueError(f"unexpected result shape: {path}")
    records = raw.reshape(QUERY_COUNT, TOPK + 1)
    if not np.all(records[:, 0] == TOPK):
        raise ValueError(f"unexpected TopK: {path}")
    return records[:, 1:]


def validate_ids(labels: np.ndarray, count: int, path: Path) -> None:
    if np.any(labels < 0) or np.any(labels >= count):
        raise ValueError(f"result ID out of range: {path}")
    if np.any(np.diff(np.sort(labels, axis=1), axis=1) == 0):
        raise ValueError(f"duplicate result ID: {path}")


def recall_at_100(output: np.ndarray, ground_truth: np.ndarray) -> float:
    matches = 0
    for start in range(0, output.shape[0], 256):
        left = output[start : start + 256, :TOPK]
        right = ground_truth[start : start + 256, :TOPK]
        matches += int((left[:, :, None] == right[:, None, :]).any(axis=2).sum())
    return matches / (output.shape[0] * TOPK)


def size_label(count: int) -> str:
    return f"{count // 1_000_000}M"


def ground_truth_path(count: int) -> Path:
    return PREPARED_DIR / f"sift_groundtruth_{count // 1_000_000}m.ivecs"


def output_path(algorithm: str, count: int, device_count: int) -> Path:
    return RESULT_DIR / f"{algorithm}_{device_count}card_{count // 1_000_000}m.ivecs"


def read_memory_rows() -> list[dict[str, str]]:
    rows = []
    for variant in VARIANTS:
        with (RESULT_DIR / f"memory_{variant}.csv").open(newline="") as source:
            rows.extend(csv.DictReader(source))
    return rows


def result_key(row: dict[str, str]) -> tuple[str, int, int]:
    return row["algorithm"], int(row["n"]), int(row["device_count"])


def validate_plan(rows: list[dict[str, str]]) -> None:
    if len(rows) != len(EXPECTED_RESULTS) or {result_key(row) for row in rows} != EXPECTED_RESULTS:
        raise ValueError("benchmark result plan is incomplete")


def print_performance() -> None:
    rows = []
    for variant in VARIANTS:
        with (RESULT_DIR / f"performance_{variant}.csv").open(newline="") as source:
            rows.extend(csv.DictReader(source))
    validate_plan(rows)

    print(f"{'algorithm':<12}{'N':<8}{'devices':>9}{'QPS':>12}{'R@100':>12}")
    previous = ""
    for row in rows:
        algorithm = row["algorithm"]
        count = int(row["n"])
        if previous and algorithm != previous:
            print()
        devices = int(row["device_count"])
        result_path = output_path(algorithm, count, devices)
        truth_path = ground_truth_path(count)
        output = read_ivecs(result_path)
        ground_truth = read_ivecs(truth_path)
        validate_ids(output, count, result_path)
        validate_ids(ground_truth, count, truth_path)
        recall = recall_at_100(output, ground_truth)
        qps = float(row["qps"])
        print(f"{algorithm:<12}{size_label(count):<8}{devices:>9}{qps:>12.1f}{recall:>12.4f}")
        previous = algorithm


def print_memory(rows: list[dict[str, str]]) -> None:
    print("\nTotal HBM delta GiB")
    header = f"{'algorithm':<12}{'N':<8}{'devices':>9}"
    header += "".join(f"{label:>10}" for label, _ in HBM_PERCENTILES)
    header += f"{'max-card':>12}"
    print(header)
    previous = ""
    for row in rows:
        algorithm = row["algorithm"]
        if previous and algorithm != previous:
            print()
        values = "".join(f"{int(row[field]) / 1024**3:>10.2f}" for _, field in HBM_PERCENTILES)
        max_card = int(row["hbm_max_device_peak_bytes"]) / 1024**3
        print(f"{algorithm:<12}{size_label(int(row['n'])):<8}{int(row['device_count']):>9}{values}{max_card:>12.2f}")
        previous = algorithm


def main() -> None:
    memory_rows = read_memory_rows()
    validate_plan(memory_rows)
    print_performance()
    print_memory(memory_rows)


if __name__ == "__main__":
    main()
