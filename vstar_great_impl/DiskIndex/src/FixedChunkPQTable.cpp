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


#include "FixedChunkPQTable.h"

namespace diskann_pro {

namespace {
    constexpr size_t CENTROID_NUM = 256; // number of centroids within each PQ subspace
    constexpr size_t BLOCK_SIZE = 5000000; // block size for reading/processing large files and matrices in blocks
    constexpr size_t INSTRUCTION_CONCURRENT = 4; // process 4 floats together with instruction set
    constexpr size_t MAX_KMEANS_REP = 12;
}

void GeneratePQPivotsImpl(const GenPQPivotInput &input, GenPQPivotOutput &output)
{
    const uint32_t MAX_DIM = 2000;
    const size_t MAX_NUM_TRAIN = 100000000;

    DISK_THROW_IF_NOT_MSG(input.dim >= 1 && input.dim <= MAX_DIM,
        "Error: dim must be in [1, 2000].\n");
    DISK_THROW_IF_NOT_MSG(input.numTrain >= 1 && input.numTrain <= MAX_NUM_TRAIN,
        "Error: numTrain must be in [1, 100000000].\n");
    DISK_THROW_IF_NOT_MSG(input.numPQChunks <= input.dim,
        "Error: number of chunks more than dimension.\n");

    // Calculate centroid and center the training data
    output.centroid = std::make_unique<float[]>(input.dim);

    for (uint64_t d = 0; d < input.dim; d++) {
        output.centroid[d] = 0;
    }
    if (input.makeZeroMean) {
      // If we use L2 distance, there is an option to
      // translate all vectors to make them centered and
      // then compute PQ. This needs to be set to false
      // when using PQ for MIPS as such translations dont
      // preserve inner products.
        for (uint64_t d = 0; d < input.dim; d++) {
            for (uint64_t p = 0; p < input.numTrain; p++) {
                output.centroid[d] += input.trainData[p * input.dim + d];
            }
            output.centroid[d] /= input.numTrain;
        }

        for (uint64_t d = 0; d < input.dim; d++) {
            for (uint64_t p = 0; p < input.numTrain; p++) {
                input.trainData[p * input.dim + d] -= output.centroid[d];
            }
        }
    }

    size_t lowVal = static_cast<size_t>(std::floor(static_cast<double>(input.dim) /
        static_cast<double>(input.numPQChunks)));
    size_t highVal = static_cast<size_t>(std::ceil(static_cast<double>(input.dim) /
        static_cast<double>(input.numPQChunks)));
    size_t maxNumHigh = input.dim - (lowVal * input.numPQChunks);
    size_t curNumHigh = 0;
    size_t curBinThreshold = highVal;

    std::vector<std::vector<uint32_t>> binToDims(input.numPQChunks);
    std::unordered_map<uint32_t, uint32_t> dimToBin;
    std::vector<float> binLoads(input.numPQChunks, 0);

    // Process dimensions not inserted by previous loop
    for (uint32_t d = 0; d < input.dim; d++) {
        if (dimToBin.find(d) != dimToBin.end()) {
            continue;
        }
        auto curBest = input.numPQChunks + 1;
        float curBestLoad = std::numeric_limits<float>::max();
        for (uint32_t b = 0; b < input.numPQChunks; b++) {
            if (binLoads[b] < curBestLoad && binToDims[b].size() < curBinThreshold) {
                curBest = b;
                curBestLoad = binLoads[b];
            }
        }
        binToDims[curBest].push_back(d);
        if (binToDims[curBest].size() == highVal) {
            curNumHigh++;
            if (curNumHigh == maxNumHigh) {
                curBinThreshold = lowVal;
            }
        }
    }

    output.chunkOffsets = std::make_unique<uint32_t[]>(input.numPQChunks + 1);
    output.chunkOffsets[0] = 0;
    for (uint32_t b = 1; b < input.numPQChunks; b++) {
        output.chunkOffsets[b] = output.chunkOffsets[b - 1] + static_cast<uint32_t>(binToDims[b - 1].size());
    }
    output.chunkOffsets[input.numPQChunks] = input.dim;

    output.fullPivotData.reset(new float[input.numCenters * input.dim]);

    for (size_t i = 0; i < input.numPQChunks; i++) {
        size_t curChunkSize = output.chunkOffsets[i + 1] - output.chunkOffsets[i];

        if (curChunkSize == 0) {
            continue;
        }
        std::unique_ptr<float[]> curPivotData = std::make_unique<float[]>(input.numCenters * curChunkSize);
        std::unique_ptr<float[]> curData = std::make_unique<float[]>(input.numTrain * curChunkSize);
        std::unique_ptr<uint32_t[]> closestCenter = std::make_unique<uint32_t[]>(input.numTrain);

#pragma omp parallel for schedule(static, 65536)
        for (int64_t j = 0; j < static_cast<int64_t>(input.numTrain); j++) {
            std::copy(
                input.trainData + j * input.dim + output.chunkOffsets[i],
                input.trainData + j * input.dim + output.chunkOffsets[i] + curChunkSize,
                curData.get() + j * curChunkSize
            );
        }

        kmeans::KmeansppSelectingPivots(curData.get(), input.numTrain, curChunkSize, curPivotData.get(),
            input.numCenters);

        kmeans::RunLloyds(curData.get(), input.numTrain, curChunkSize, curPivotData.get(), input.numCenters,
                          MAX_KMEANS_REP, nullptr, closestCenter.get());

        for (uint64_t j = 0; j < input.numCenters; j++) {
            std::copy(
                curPivotData.get() + j * curChunkSize,
                curPivotData.get() + j * curChunkSize + curChunkSize,
                output.fullPivotData.get() + j * input.dim + output.chunkOffsets[i]
            );
        }
    }
}

void GeneratePQDataFromPivotsImpl(const GenPQPivotOutput &pivot, GenPQPivotInput &input, uint32_t *compressedData)
{
    size_t curBlkSize = input.numTrain;

    // 因为我们要在原始数据上做减法，拷贝一份原始数据以保证原本数据不被更改
    std::vector<float> trainDataTmp(input.dim * curBlkSize);
    std::copy(input.trainData, input.trainData + trainDataTmp.size(), trainDataTmp.data());

    // 对初始化数据进行偏移（每个值去除每个维度的平均值）保证极值不影响kmeans计算
    for (size_t p = 0; p < curBlkSize; p++) {
        for (uint64_t d = 0; d < input.dim; d++) {
            trainDataTmp[p * input.dim + d] -= pivot.centroid[d];
        }
    }

    for (size_t i = 0; i <  input.numPQChunks; i++) {
        size_t curChunkSize = pivot.chunkOffsets[i + 1] - pivot.chunkOffsets[i];
        if (curChunkSize == 0) {
            continue;
        }
        std::unique_ptr<float[]> curPivotData = std::make_unique<float[]>(input.numCenters * curChunkSize);
        std::unique_ptr<float[]> curData = std::make_unique<float[]>(curBlkSize * curChunkSize);
        std::unique_ptr<uint32_t[]> closestCenter = std::make_unique<uint32_t[]>(curBlkSize);

#pragma omp parallel for schedule(static, 8192)
        for (int64_t j = 0; j < static_cast<int64_t>(curBlkSize); j++) {
            for (size_t k = 0; k < curChunkSize; k++) {
                curData[j * curChunkSize + k] = trainDataTmp[j * input.dim + pivot.chunkOffsets[i] + k];
            }
        }

#pragma omp parallel for schedule(static, 1)
        for (int64_t j = 0; j < static_cast<int64_t>(input.numCenters); j++) {
            std::copy(
                pivot.fullPivotData.get() + j * input.dim + pivot.chunkOffsets[i],
                pivot.fullPivotData.get() + j * input.dim + pivot.chunkOffsets[i] + curChunkSize,
                curPivotData.get() + j * curChunkSize
            );
        }

        math_utils::ComputeClosestCenters(curData.get(), curBlkSize, curChunkSize, curPivotData.get(),
                                          input.numCenters, 1, closestCenter.get());

#pragma omp parallel for schedule(static, 8192)
        for (int64_t j = 0; j < static_cast<int64_t>(curBlkSize); j++) {
            compressedData[j * input.numPQChunks + i] = closestCenter[j];
        }
    }
}

} // namespace diskann_pro
