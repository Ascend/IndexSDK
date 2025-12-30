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

#include<algorithm>
#include "ascenddaemon/utils/AscendTensor.h"
#include "ascenddaemon/utils/Limits.h"
#include "common/utils/CommonUtils.h"
#include "ops/cpukernel/impl/utils/kernel_shared_def.h"
#include "index/IndexInt8FlatL2Aicpu.h"

namespace ascend {
namespace {
const int BURST_LEN = 64;
const int PAGE_BLOCKS = 32;
}

IndexInt8FlatL2Aicpu::IndexInt8FlatL2Aicpu(int dim, int64_t resourceSize, int blockSize)
    : IndexInt8Flat<int32_t>(dim, MetricType::METRIC_L2, resourceSize, blockSize)
{
    this->isSupportMultiSearch = true;
    this->int8FlatIndexType = Int8FlatIndexType::INT8_FLAT_L2;
}

IndexInt8FlatL2Aicpu::~IndexInt8FlatL2Aicpu() {}

APP_ERROR IndexInt8FlatL2Aicpu::init()
{
    APP_ERROR ret = resetTopkCompOp();
    APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, ret, "failed to reset topk op");
    ret = resetMultisearchTopkCompOp();
    APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, ret, "failed to reset multisearch topk op");
    ret = resetDistCompOp(codeBlockSize);
    APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, ret, "failed to reset resetDistCompOp");
    return APP_ERR_OK;
}

APP_ERROR IndexInt8FlatL2Aicpu::addVectors(AscendTensor<int8_t, DIMS_2> &rawData)
{
    // 1. save code
    APP_ERROR ret = IndexInt8Flat::addVectors(rawData);
    APPERR_RETURN_IF(ret, ret);

    // 2. compute the precompute data
    computeNorm(rawData);

    this->ntotal += (idx_t)rawData.getSize(0);
    return APP_ERR_OK;
}

void IndexInt8FlatL2Aicpu::runDistCompute(int batch,
                                          const std::vector<const AscendTensorBase *> &input,
                                          const std::vector<const AscendTensorBase *> &output,
                                          aclrtStream stream, uint32_t actualNum) const
{
    IndexTypeIdx type = actualNum == static_cast<uint32_t>(this->codeBlockSize) ?
        IndexTypeIdx::ITI_INT8_L2_FULL : IndexTypeIdx::ITI_INT8_L2;
    if (faiss::ascend::SocUtils::GetInstance().IsAscend910B()) {
        type = IndexTypeIdx::ITI_INT8_L2;
    }
    std::vector<int> keys({batch, dims, codeBlockSize});
    OpsMngKey opsKey(keys);
    auto ret = DistComputeOpsManager::getInstance().runOp(type, opsKey, input, output, stream);
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "run operator failed: %i\n", ret);
}

APP_ERROR IndexInt8FlatL2Aicpu::resetDistCompOp(int codeNum)
{
    std::vector<IndexTypeIdx> distCompOpsIdxs = {IndexTypeIdx::ITI_INT8_L2, IndexTypeIdx::ITI_INT8_L2_FULL};
    std::vector<std::string> distCompOpsNames = {"DistanceInt8L2Mins", "DistanceInt8L2FullMins"};
    if (faiss::ascend::SocUtils::GetInstance().IsAscend910B()) {
        distCompOpsIdxs.clear();
        distCompOpsIdxs.emplace_back(IndexTypeIdx::ITI_INT8_L2);

        distCompOpsNames.clear();
        distCompOpsNames.emplace_back("AscendcDistInt8FlatL2");
    }
    for (size_t i = 0; i < distCompOpsIdxs.size(); i++) {
        std::string opTypeName = distCompOpsNames.at(i) ;
        IndexTypeIdx indexType = distCompOpsIdxs.at(i);
        for (auto batch : searchBatchSizes) {
            std::vector<int64_t> queryShape({ batch, dims });
            std::vector<int64_t> maskShape({ batch, utils::divUp(codeNum, 8) }); // divUp to 8
            std::vector<int64_t> coarseCentroidsShape({
                utils::divUp(codeNum, CUBE_ALIGN), utils::divUp(dims, CUBE_ALIGN_INT8), CUBE_ALIGN, CUBE_ALIGN_INT8});
            std::vector<int64_t> preNormsShape({ codeNum });
            std::vector<int64_t> sizeShape({ CORE_NUM, SIZE_ALIGN });
            std::vector<int64_t> distResultShape({ batch, codeNum });
            std::vector<int64_t> minResultShape({ batch, this->burstsOfBlock });
            std::vector<int64_t> flagShape({ flagNum, FLAG_SIZE });

            std::vector<std::pair<aclDataType, std::vector<int64_t>>> input {
                { ACL_INT8, queryShape },
                { ACL_UINT8, maskShape },
                { ACL_INT8, coarseCentroidsShape },
                { ACL_INT32, preNormsShape },
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
    }

    return APP_ERR_OK;
}

void IndexInt8FlatL2Aicpu::initSearchResult(int indexesSize, int n, int k, float16_t *distances, idx_t *labels)
{
    AscendTensor<float16_t, DIMS_3> outDistances(distances, { indexesSize, n, k });
    AscendTensor<idx_t, DIMS_3> outIndices(labels, { indexesSize, n, k });
    outDistances.initValue(Limits<float16_t>::getMax());
    outIndices.initValue(std::numeric_limits<idx_t>::max());
}
} // namespace ascend