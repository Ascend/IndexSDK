/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * IndexSDK is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */

#pragma once

#include <memory>
#include <unordered_map>

#include "MathUtils.h"

namespace diskann_pro {

struct GenPQPivotInput {
    float *trainData = nullptr;
    size_t numTrain = 0;
    uint32_t dim = 0;
    uint32_t numCenters = 256;
    uint32_t numPQChunks = 0;
    bool makeZeroMean = false;
    bool verbose = false;
};

struct GenPQPivotOutput {
    std::unique_ptr<float[]> fullPivotData;
    std::unique_ptr<uint32_t[]> chunkOffsets;
    std::unique_ptr<float[]> centroid;
};

void GeneratePQPivotsImpl(const GenPQPivotInput &input, GenPQPivotOutput &output);

void GeneratePQDataFromPivotsImpl(const GenPQPivotOutput &pivot, GenPQPivotInput &input, uint32_t *compressedData);

} // namespace diskann_pro
