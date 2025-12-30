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


#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#include <memory>

#include "CommonIncludes.h"
#include "DiskAssert.h"
#include "FixedChunkPQTable.h"
#include "Utils.h"

#include "Adapter/OpenGaussAdapter.h"

namespace {
    constexpr size_t CENTROID_NUM = 256; // number of centroids within each PQ subspace (i.e. an 8-bit quantizer)
    constexpr size_t INSTRUCTION_CONCURRENT = 4; // process 4 floats together with instruction set

    constexpr int VEC_MAX_LEN_LIMIT = 100000000;
    constexpr int DIM_LIMIT = 2000;

    const std::vector<int> FUNC_TYPE_SET = {1, 2, 3};

    // OpenGauss侧Vector数据结构的meta信息字节大小；包括vl_len_(4B) + dim(2B) + unused(2B)
    constexpr uint8_t VECTOR_META_SIZE = 8;
    constexpr int MAX_THREAD_USE = 96;

    enum class Metrics : int {
        INVALID_DIS_METRICS = 0,
        DIS_L2 = 1,
        DIS_IP = 2,
        DIS_COSINE = 3
    };
}

using diskann_pro::GenPQPivotInput;
using diskann_pro::GenPQPivotOutput;

/*==================================================================
                            工具类函数
==================================================================*/

static void CheckVectorArray(const VectorArrayData *vec)
{
    DISK_THROW_IF_NOT_MSG(vec != nullptr, "[CheckVectorArray] VectorArrayData input cannot be a nullptr.\n");
    DISK_THROW_IF_NOT_MSG(vec->maxlen >= 1 && vec->maxlen <= VEC_MAX_LEN_LIMIT,
        "[CheckVectorArray] VectorArrayData's maxlen must be in [1, 1e8].\n");
    DISK_THROW_IF_NOT_MSG(vec->length >= 1 && vec->length <= vec->maxlen,
        "[CheckVectorArray] VectorArrayData's actual length must be in [1, maxlen].\n");
    DISK_THROW_IF_NOT_MSG(vec->dim >= 1 && vec->dim <= DIM_LIMIT,
        "[CheckVectorArray] VectorArrayData's dim must be in [1, 2000].\n");
    DISK_THROW_IF_NOT_MSG(vec->items != nullptr,
        "[CheckVectorArray] VectorArrayData's actual data cannot be a nullptr.\n");
}

static void CheckPQParamsEmpty(const DiskPQParams *params)
{
    /* 生成码本阶段，需要PQParams内的相关参数为空指针；接口内部去申请内存并接收数据 */
    DISK_THROW_IF_NOT_MSG(params != nullptr, "[CheckPQParamsEmpty] DiskPQParams input cannot be a nullptr.\n");
    DISK_THROW_IF_NOT_MSG(params->pqTable == nullptr,
        "[CheckPQParamsEmpty] DiskPQParams's pqTable must be a nullptr to accept data.\n");
    DISK_THROW_IF_NOT_MSG(params->offsets == nullptr,
        "[CheckPQParamsEmpty] DiskPQParams's offsets must be a nullptr to accept data.\n");
    DISK_THROW_IF_NOT_MSG(params->tablesTransposed == nullptr,
        "[CheckPQParamsEmpty] DiskPQParams's tablesTransposed must be a nullptr to accept data.\n");
    DISK_THROW_IF_NOT_MSG(params->centroids == nullptr,
        "[CheckPQParamsEmpty] DiskPQParams's centroids must be a nullptr to accept data.\n");
    DISK_THROW_IF_NOT_MSG(std::find(FUNC_TYPE_SET.begin(), FUNC_TYPE_SET.end(),
        params->funcType) != FUNC_TYPE_SET.end(),
        "[CheckPQParamsEmpty] DiskPQParams's funcType must be one of: 1(L2), 2(IP), 3(Cosine).\n");
}

static void CheckPQParamsFilled(const DiskPQParams *params, bool needTransposed)
{
    /* 生成码字阶段，需要PQParams内的相关参数填充好数据 */
    DISK_THROW_IF_NOT_MSG(params != nullptr, "[CheckPQParamsFilled] DiskPQParams input cannot be a nullptr.\n");

    // 码字生成阶段不需要transposed，但检索阶段需要，因此使用bool变量判断何时需要对应的table变量
    if (needTransposed) {
        DISK_THROW_IF_NOT_MSG(params->tablesTransposed != nullptr,
            "[CheckPQParamsFilled] DiskPQParams's tablesTransposed cannot be a nullptr.\n");
    } else {
        DISK_THROW_IF_NOT_MSG(params->pqTable != nullptr,
            "[CheckPQParamsFilled] DiskPQParams's pqTable cannot be a nullptr.\n");
    }
    DISK_THROW_IF_NOT_MSG(params->offsets != nullptr,
        "[CheckPQParamsFilled] DiskPQParams's offsets cannot be a nullptr.\n");
    DISK_THROW_IF_NOT_MSG(params->centroids != nullptr,
        "[CheckPQParamsFilled] DiskPQParams's centroids cannot be a nullptr.\n");
    DISK_THROW_IF_NOT_MSG(std::find(FUNC_TYPE_SET.begin(), FUNC_TYPE_SET.end(),
        params->funcType) != FUNC_TYPE_SET.end(),
        "[CheckPQParamsEmpty] DiskPQParams's funcType must be one of: 1(L2), 2(IP), 3(Cosine).\n");
}

/* 训练阶段，检查2个入参中需要交互的参数的有效性 */
static void CheckCommonParams(const VectorArrayData *vec, const DiskPQParams *pqParams)
{
    DISK_THROW_IF_NOT_MSG(vec->dim == pqParams->dim,
        "[CheckCommonParams] VectorArrayData's dim and DiskPQParams' dim do not match.\n");
    DISK_THROW_IF_NOT_MSG(pqParams->pqChunks >= 1 && pqParams->pqChunks <= pqParams->dim,
        "[CheckCommonParams] DiskPQParams' pqChunk must be in [1, dim].\n");
}

/* GeneratePQPivots阶段，OpenGauss侧的items指向的数据为length长度的Vector数据类型，我们需要将这些Vector数据类型的数据集中
   起来为一个length * dim的数据; 定义为模板类去对未来其他数据类型进行适配 */
template <typename T>
void GetRawDataFromItems(char *items, size_t length, int dim, T *rawData)
{
    size_t structSize = VECTOR_META_SIZE + sizeof(T) * dim;
#pragma omp parallel for
    for (size_t i = 0; i < length; ++i) {
        T *itemsT = reinterpret_cast<T *>(items + i * structSize);
        std::copy(itemsT, itemsT + dim, rawData + i * dim);
    }
}

/*==================================================================
                            实现类函数
==================================================================*/

static void GeneratePQPivots(VectorArrayData *sample, DiskPQParams *params)
{
    CheckVectorArray(sample);
    CheckPQParamsEmpty(params);
    CheckCommonParams(sample, params);

    const int maxThreads = omp_get_max_threads();
    // Use at most 1/2 of the max supported omp thread num or 96, whichever is lower
    omp_set_num_threads(std::min(maxThreads / 2, MAX_THREAD_USE));

    GenPQPivotInput input;
    input.dim = static_cast<uint32_t>(sample->dim);
    std::vector<float> trainDataVec(static_cast<size_t>(sample->length) * static_cast<size_t>(sample->dim));
    GetRawDataFromItems(sample->items, static_cast<size_t>(sample->length), sample->dim, trainDataVec.data());
    input.trainData = trainDataVec.data();
    input.numTrain = static_cast<size_t>(sample->length);
    input.numPQChunks = static_cast<uint32_t>(params->pqChunks);
    input.numCenters = static_cast<uint32_t>(CENTROID_NUM);
    input.makeZeroMean = (params->funcType == 1) ? true : false; // 1 is L2 distance, 2 is IP distance

    GenPQPivotOutput output;
    GeneratePQPivotsImpl(input, output);

    // 将PQPivot相关结果传回params内
    params->offsets = output.chunkOffsets.release();
    params->centroids = reinterpret_cast<char *>(output.centroid.release());
    params->pqTable = reinterpret_cast<char *>(output.fullPivotData.release());
    auto tablesTrTmp = std::make_unique<float[]>(CENTROID_NUM * input.dim);
    for (size_t i = 0; i < CENTROID_NUM; i++) {
        for (size_t j = 0; j < input.dim; j++) {
            tablesTrTmp[j * CENTROID_NUM + i] = (reinterpret_cast<float *>(params->pqTable))[i * input.dim + j];
        }
    }
    params->tablesTransposed = reinterpret_cast<char *>(tablesTrTmp.release());
}

static void GeneratePQDataFromPivots(VectorArrayData *baseData, const DiskPQParams *params, uint8_t *pqCode)
{
    CheckVectorArray(baseData);
    CheckPQParamsFilled(params, false);
    CheckCommonParams(baseData, params);
    DISK_THROW_IF_NOT_MSG(pqCode != nullptr, "[GeneratePQDataFromPivots] pqCode cannot be a nullptr.\n");

    omp_set_num_threads(1); // 由于外部输入固定为1，先设置使用的线程数为1
    GenPQPivotInput input;
    input.dim = static_cast<uint32_t>(baseData->dim);
    input.trainData = reinterpret_cast<float *>(baseData->items);
    input.numTrain = static_cast<size_t>(baseData->length);
    input.numPQChunks = static_cast<uint32_t>(params->pqChunks);
    input.numCenters = static_cast<uint32_t>(CENTROID_NUM);

    GenPQPivotOutput pivot;
    pivot.fullPivotData = std::make_unique<float[]>(input.numCenters * input.dim);
    float *pqTableFloat = reinterpret_cast<float *>(params->pqTable);
    std::copy(pqTableFloat, pqTableFloat + static_cast<size_t>(input.numCenters) * input.dim,
              pivot.fullPivotData.get());

    pivot.centroid = std::make_unique<float[]>(input.dim);
    float *centroidsFloat = reinterpret_cast<float *>(params->centroids);
    std::copy(centroidsFloat, centroidsFloat + input.dim, pivot.centroid.get());

    pivot.chunkOffsets = std::make_unique<uint32_t[]>(input.numPQChunks + 1);
    std::copy(params->offsets, params->offsets + (input.numPQChunks + 1), pivot.chunkOffsets.get());

    std::unique_ptr<uint32_t[]> blockCompressedBase = std::make_unique<uint32_t[]>(input.numTrain * input.numPQChunks);
    GeneratePQDataFromPivotsImpl(pivot, input, blockCompressedBase.get());

    // 将uint32_t表示的blockCompressedBase转换后放置在pqCode内
    for (size_t i = 0; i < input.numTrain; ++i) {
        for (uint32_t j = 0; j < input.numPQChunks; ++j) {
            pqCode[i * input.numPQChunks + j] = static_cast<uint8_t>(blockCompressedBase[i * input.numPQChunks + j]);
        }
    }
}

static void PreprocessQuery(char *queryVec, float *queryVecProcessed, const DiskPQParams *params)
{
    float *queryVecFloat = reinterpret_cast<float *>(queryVec);
    float *centroidsFloat = reinterpret_cast<float *>(params->centroids);
    // L2距离除对向量按照centroid进行移动外无其他操作；之后需适配IP距离需要的对距离的预处理操作
    for (int d = 0; d < params->dim; d++) {
        queryVecProcessed[d] = (queryVecFloat[d] - centroidsFloat[d]);
    }
}

/**
 * @brief This function compute distance between a particular dim of query with centers of pqtable from that dim,
 *        based on the distance metrics
 * @param query pointer to the start of query
 * @param currDimCenters  pointer to the codebook's current dim (which has 256 centroid values)
 * @param currDim currently processing dim
 * @param metrics computation distance metrics (1 for L2, 2 for IP, 3 for Cosine)
 * @param chunkDists pointer to pq distance to be updated
 */
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
inline void DistComputeByMetricsSIMD(const float *query, const float *currDimCenters, size_t currDim,
                                     int metrics, float *chunkDists)
{
    float32x4_t vQueryVec = vdupq_n_f32(query[currDim]);
    for (size_t idx = 0; idx < CENTROID_NUM; idx += INSTRUCTION_CONCURRENT) {
        if (idx < CENTROID_NUM - INSTRUCTION_CONCURRENT) {
            diskann_pro::Prefetch(currDimCenters + idx + INSTRUCTION_CONCURRENT);
        }
        // Load 4 center values
        float32x4_t vCenters = vld1q_f32(currDimCenters + idx);
        if (metrics == static_cast<int>(Metrics::DIS_L2) || metrics == static_cast<int>(Metrics::DIS_COSINE)) {
            // Compute squared differences
            float32x4_t vDiff = vsubq_f32(vCenters, vQueryVec);
            float32x4_t vDiffSQ = vmulq_f32(vDiff, vDiff);
            // Accumulate results
            float32x4_t vChunkDists = vld1q_f32(chunkDists + idx);
            vChunkDists = vaddq_f32(vChunkDists, vDiffSQ);
            vst1q_f32(chunkDists + idx, vChunkDists);
        } else if (metrics == static_cast<int>(Metrics::DIS_IP)) {
            // Compute squared and accumulate the negative distance (since the outside queue is a minheap)
            float32x4_t vProduct = vmulq_f32(vCenters, vQueryVec);
            // Accumulate results
            float32x4_t vChunkDists = vld1q_f32(chunkDists + idx);
            vChunkDists = vsubq_f32(vChunkDists, vProduct);
            vst1q_f32(chunkDists + idx, vChunkDists);
        }
    }
}
#else
/* Identical functionality as above without SIMD optimization */
inline void DistComputeByMetrics(const float *query, const float *currDimCenters, size_t currDim,
                                 int metrics, float *chunkDists)
{
    for (size_t idx = 0; idx < CENTROID_NUM; idx++) {
        if (metrics == static_cast<int>(Metrics::DIS_L2) || metrics == static_cast<int>(Metrics::DIS_COSINE)) {
            float diff = currDimCenters[idx] - query[currDim];
            chunkDists[idx] += diff * diff;
        } else if (metrics == static_cast<int>(Metrics::DIS_IP)) {
            chunkDists[idx] -= (currDimCenters[idx] * query[currDim]);
        }
    }
}
#endif

static void PopulateChunkDistancesOPGS(const float *queryVec, float *distVec, const DiskPQParams *params)
{
    uint64_t chunksNum = static_cast<uint64_t>(params->pqChunks);
    uint32_t *chunkOffsets = params->offsets;
    float *tablesTransposed = reinterpret_cast<float *>(params->tablesTransposed);
    std::fill(distVec, distVec + CENTROID_NUM * chunksNum, 0.0f);
    for (size_t chunk = 0; chunk < chunksNum; chunk++) {
        float *chunkDists = distVec + (CENTROID_NUM * chunk);
        for (size_t j = chunkOffsets[chunk]; j < chunkOffsets[chunk + 1]; j++) {
            const float *centersDimVec = tablesTransposed + (CENTROID_NUM * j);
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
            DistComputeByMetricsSIMD(queryVec, centersDimVec, j, params->funcType, chunkDists);
#else
            DistComputeByMetrics(queryVec, centersDimVec, j, params->funcType, chunkDists);
#endif
        }
    }
}

static void PQDistLookup(const uint8_t *pqIDs, size_t pqChunksNum, const float *pqDists, float &distsOut)
{
    for (size_t chunk = 0; chunk < pqChunksNum; chunk++) {
        const float *chunkDists = pqDists + CENTROID_NUM * chunk;
        distsOut += chunkDists[pqIDs[chunk]];
    }
}

/*==================================================================
                            对外接口函数
==================================================================*/

int ComputePQTable(VectorArrayData *sample, DiskPQParams *params)
{
    try {
        GeneratePQPivots(sample, params);
    } catch (const std::exception& e) {
        std::cerr << "ComputePQTable errors: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}

int ComputeVectorPQCode(VectorArrayData *baseData, const DiskPQParams *params, uint8_t *pqCode)
{
    try {
        GeneratePQDataFromPivots(baseData, params, pqCode);
    } catch (const std::exception& e) {
        std::cerr << "ComputeVectorPQCode errors: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}

int GetPQDistanceTable(char *vec, const DiskPQParams *params, float *pqDistanceTable)
{
    try {
        DISK_THROW_IF_NOT_MSG(vec != nullptr, "[GetPQDistanceTable] Input vec cannot be a nullptr.\n");
        CheckPQParamsFilled(params, true);
        DISK_THROW_IF_NOT_MSG(pqDistanceTable != nullptr,
            "[GetPQDistanceTable] pqDistanceTable output cannot be a nullptr.\n");

        std::vector<float> queryVecTmp(params->dim);
        PreprocessQuery(vec, queryVecTmp.data(), params);
        PopulateChunkDistancesOPGS(queryVecTmp.data(), pqDistanceTable, params);
    } catch (const std::exception& e) {
        std::cerr << "GetPQDistanceTable errors: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}

int GetPQDistance(const uint8_t *basecode, const DiskPQParams *params, const float *pqDistanceTable, float &pqDistance)
{
    try {
        DISK_THROW_IF_NOT_MSG(basecode != nullptr, "[GetPQDistance] basecode cannot be a nullptr.\n");
        DISK_THROW_IF_NOT_MSG(pqDistanceTable != nullptr, "[GetPQDistance] pqDistanceTable cannot be a nullptr.\n");
        DISK_THROW_IF_NOT_MSG(params != nullptr, "[GetPQDistance] DiskPQParams cannot be a nullptr.\n");
        DISK_THROW_IF_NOT_MSG(params->pqChunks >= 1 && params->pqChunks <= params->dim,
            "[GetPQDistance] DiskPQParams' pqChunk must be in [1, dim].\n");

        PQDistLookup(basecode, static_cast<size_t>(params->pqChunks), pqDistanceTable, pqDistance);
    } catch (const std::exception& e) {
        std::cerr << "GetPQDistance errors: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}