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

#include <algorithm>
#include "ascend/utils/fp16.h"
#include "ascenddaemon/utils/AscendTensor.h"
#include "ascenddaemon/utils/Limits.h"
#include "common/utils/CommonUtils.h"
#include "ops/cpukernel/impl/utils/kernel_shared_def.h"
#include "index_custom/IndexInt8FlatApproxL2Aicpu.h"

namespace ascend {
namespace {
const int BURST_LEN = 64;
const int PAGE_BLOCKS = 32;

const float SCALE_INIT = 0.01;
const int DIM_SCALE = 64;
const int DIM_SCALE_DOUBLE = 128;
const int SCALE_INIT_MIN_SIZE_ALIGN = 4;
}

IndexInt8FlatApproxL2Aicpu::IndexInt8FlatApproxL2Aicpu(int dim, int64_t resourceSize, int blockSize)
    : IndexInt8Flat<float16_t>(dim, MetricType::METRIC_L2, resourceSize, blockSize)
{
    this->int8FlatIndexType = Int8FlatIndexType::INT8_FLAT_APPROXL2;
    this->searchBatchSizes = {128, 96, 64, 48, 36, 32, 24, 18, 16, 12, 8, 6, 4, 2, 1};

    // scale changed with dim
    this->scale =
        SCALE_INIT / std::min(dim / DIM_SCALE, std::max(dim / DIM_SCALE_DOUBLE + 1, SCALE_INIT_MIN_SIZE_ALIGN));
}

IndexInt8FlatApproxL2Aicpu::~IndexInt8FlatApproxL2Aicpu() {}

APP_ERROR IndexInt8FlatApproxL2Aicpu::init()
{
    APP_ERROR ret = resetTopkCompOp();
    APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, ret, "failed to reset topk op");
    return resetDistCompOp(codeBlockSize);
}


float IndexInt8FlatApproxL2Aicpu::ivecNormL2sqr(const int8_t *x, size_t d) const
{
    int32_t res = 0;
    for (size_t i = 0; i < d; i++) {
        res += static_cast<int32_t>(x[i]) * static_cast<int32_t>(x[i]);
    }
    return (res * this->scale);
}

void IndexInt8FlatApproxL2Aicpu::ivecNormsL2sqr(float16_t *nr, const int8_t *x, size_t d, size_t nx)
{
    std::vector<uint16_t> norms(nx);
    std::vector<float> normsFloat(nx);
#pragma omp parallel for num_threads(CommonUtils::GetThreadMaxNums())
    for (size_t i = 0; i < nx; i++) {
        normsFloat[i] = ivecNormL2sqr(x + i * d, d);
    }
    transform(normsFloat.data(), normsFloat.data() + static_cast<uint32_t>(nx), std::begin(norms),
        [](float temp) { return faiss::ascend::fp16(temp).data; });
    auto err = aclrtMemcpy(nr, nx * sizeof(float16_t), norms.data(), nx * sizeof(float16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(err == EOK, "Mem operator error %d", (int)err);
}

APP_ERROR IndexInt8FlatApproxL2Aicpu::computeNorm(AscendTensor<int8_t, DIMS_2> &rawData)
{
    int num = rawData.getSize(0);
    int dim = rawData.getSize(1);

    bool isFirst = true;
    int idx = 0;
    for (int i = 0; i < num; i++) {
        int idx1 = static_cast<int>((this->ntotal + static_cast<idx_t>(i)) / static_cast<idx_t>(codeBlockSize));
        int idx2 = static_cast<int>((this->ntotal + static_cast<idx_t>(i)) % static_cast<idx_t>(codeBlockSize));

        // if the baseShapedSlice is full or reach the last
        if (idx2 + 1 == codeBlockSize || i == num - 1) {
            float16_t *pNormBaseSlice = normBase[idx1]->data();

            // calc y^2 (the first time is different)
            if (isFirst) {
                ivecNormsL2sqr(pNormBaseSlice + this->ntotal % codeBlockSize, rawData[idx][0].data(), dim, i + 1);
                idx += (i + 1);
                isFirst = false;
            } else {
                ivecNormsL2sqr(pNormBaseSlice, rawData[idx][0].data(), dim, idx2 + 1);
                idx += (idx2 + 1);
            }
        }
    }
    return APP_ERR_OK;
}

APP_ERROR IndexInt8FlatApproxL2Aicpu::addVectors(AscendTensor<int8_t, DIMS_2> &rawData)
{
    // 1. save code
    APP_ERROR ret = IndexInt8Flat::addVectors(rawData);
    APPERR_RETURN_IF(ret, ret);

    // 2. compute the precompute data
    computeNorm(rawData);

    this->ntotal += static_cast<idx_t>(rawData.getSize(0));
    return APP_ERR_OK;
}

void IndexInt8FlatApproxL2Aicpu::runDistCompute(int batch,
    const std::vector<const AscendTensorBase *> &input, const std::vector<const AscendTensorBase *> &output,
    aclrtStream stream, uint32_t actualNum) const
{
    VALUE_UNUSED(actualNum);
    IndexTypeIdx indexType = IndexTypeIdx::ITI_INT8_APPROX_L2;
    std::vector<int> keys({batch, this->dims, codeBlockSize});
    OpsMngKey opsKey(keys);
    auto ret = DistComputeOpsManager::getInstance().runOp(indexType, opsKey, input, output, stream);
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "run operator failed: %i\n", ret);
}

APP_ERROR IndexInt8FlatApproxL2Aicpu::resetDistCompOp(int codeNum) const
{
    IndexTypeIdx indexType = IndexTypeIdx::ITI_INT8_APPROX_L2;
    std::string opTypeName = "DistanceInt8L2MinsWoQueryNorm";
    for (auto batch : searchBatchSizes) {
        std::vector<int64_t> queryShape({batch, dims});
        std::vector<int64_t> maskShape({batch, utils::divUp(codeNum, 8)});  // divUp to 8
        std::vector<int64_t> coarseCentroidsShape(
            {utils::divUp(codeNum, CUBE_ALIGN), utils::divUp(dims, CUBE_ALIGN_INT8), CUBE_ALIGN, CUBE_ALIGN_INT8});
        std::vector<int64_t> preNormsShape({codeNum});
        std::vector<int64_t> sizeShape({CORE_NUM, SIZE_ALIGN});
        std::vector<int64_t> distResultShape({batch, codeNum});
        std::vector<int64_t> minResultShape({batch, this->burstsOfBlock});
        std::vector<int64_t> flagShape({FLAG_NUM, FLAG_SIZE});

        std::vector<std::pair<aclDataType, std::vector<int64_t>>> input {
            { ACL_INT8, queryShape },
            { ACL_UINT8, maskShape },
            { ACL_INT8, coarseCentroidsShape },
            { ACL_FLOAT16, preNormsShape },
            { ACL_UINT32, sizeShape }
        };
        std::vector<std::pair<aclDataType, std::vector<int64_t>>> output {
            { ACL_FLOAT16, distResultShape },
            { ACL_FLOAT16, minResultShape },
            { ACL_UINT16, flagShape }
        };
        std::vector<int> keys({batch, dims, codeBlockSize});
        OpsMngKey opsKey(keys);
        auto ret = DistComputeOpsManager::getInstance().resetOp(opTypeName, indexType, opsKey, input, output);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "op init failed: %i", ret);
    }

    return APP_ERR_OK;
}
} // namespace ascend