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

import struct
from pathlib import Path
from typing import Optional

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "tmp/sift100m-raw"
OUTPUT_DIR = PROJECT_DIR / "tmp/sift100m-prepared"
DIMENSION = 128
BASE_COUNT = 100_000_000
LEARN_COUNT = 100_000
INT8_SCALE = 256.0
BLOCK_SIZE = 65_536


def vector_count(path: Path) -> int:
    with path.open("rb") as source:
        _, dimension = struct.unpack("<II", source.read(8))
    if dimension == 0 or dimension != DIMENSION:
        raise ValueError(f"invalid u8bin file: {path}")
    count, remainder = divmod(path.stat().st_size - 8, dimension)
    if remainder:
        raise ValueError(f"invalid u8bin file: {path}")
    return count


def write_normalized(source_path: Path, count: int, fvecs_path: Path, i8bin_path: Optional[Path]) -> None:
    source = np.memmap(source_path, dtype=np.uint8, mode="r", offset=8, shape=(vector_count(source_path), DIMENSION))
    fvecs_tmp = fvecs_path.with_suffix(fvecs_path.suffix + ".part")
    i8bin_tmp = i8bin_path.with_suffix(i8bin_path.suffix + ".part") if i8bin_path is not None else None
    record_type = np.dtype([("dimension", "<i4"), ("values", "<f4", (DIMENSION,))])

    with fvecs_tmp.open("wb") as fvecs_output:
        if i8bin_tmp is not None:
            i8bin_output = i8bin_tmp.open("wb")
            i8bin_output.write(struct.pack("<II", count, DIMENSION))
        else:
            i8bin_output = None

        try:
            for start in range(0, count, BLOCK_SIZE):
                block = np.asarray(source[start : min(start + BLOCK_SIZE, count)], dtype=np.float32)
                norms = np.linalg.norm(block, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                block /= norms

                records = np.empty(block.shape[0], dtype=record_type)
                records["dimension"] = DIMENSION
                records["values"] = block
                records.tofile(fvecs_output)

                if i8bin_output is not None:
                    quantized = np.clip(np.rint(block * INT8_SCALE), -127, 127).astype(np.int8)
                    quantized.tofile(i8bin_output)
        finally:
            if i8bin_output is not None:
                i8bin_output.close()

    fvecs_tmp.replace(fvecs_path)
    if i8bin_path is not None and i8bin_tmp is not None:
        i8bin_tmp.replace(i8bin_path)


def main() -> None:
    base_source = SOURCE_DIR / "base.first100M.u8bin"
    query_source = SOURCE_DIR / "query.public.10K.u8bin"
    learn_source = SOURCE_DIR / "learn.first100K.u8bin"
    if vector_count(base_source) != BASE_COUNT:
        raise ValueError("SIFT100M base prefix is incomplete")

    query_count = vector_count(query_source)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_normalized(
        base_source,
        BASE_COUNT,
        OUTPUT_DIR / "sift_base.fvecs",
        OUTPUT_DIR / "sift_base.i8bin",
    )
    write_normalized(
        query_source,
        query_count,
        OUTPUT_DIR / "sift_query.fvecs",
        None,
    )
    write_normalized(learn_source, LEARN_COUNT, OUTPUT_DIR / "sift_learn.fvecs", None)


if __name__ == "__main__":
    main()
