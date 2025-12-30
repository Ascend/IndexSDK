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

#include "index_custom/IndexFlatAT.h"

#include "common/utils/OpLauncher.h"
#include "ops/cpukernel/impl/utils/kernel_shared_def.h"
#include "common/utils/CommonUtils.h"
#include "ascend/utils/fp16.h"

namespace ascend {

IndexFlatAT::IndexFlatAT(int dim, int baseSize, int64_t resourceSize) : Index(dim, resourceSize),
    queryBatch(QUERY_BATCH), baseSize(baseSize)
{
    ASCEND_THROW_IF_NOT(dims % CUBE_ALIGN == 0);

    // The size of the memory pool is strongly related to the baseSize, 8192
    this->searchPage = SEARCH_PAGE * 8192 / baseSize;
    // the result constain min value and index, the multi 2
    this->bursts = utils::divUp(this->baseSize, BURST_LEN) * 2;

    isTrained = true;
}

APP_ERROR IndexFlatAT::resetTopkCompOp()
{
    int dim1 = utils::divUp(this->baseSize, CODE_ALIGN);
    int coreNum = std::min(CORE_NUM, dim1);
    AscendOpDesc desc("TopkFlat");
    std::vector<int64_t> shape0 { 0, this->queryBatch, this->baseSize };
    std::vector<int64_t> shape1 { 0, this->queryBatch, this->bursts };
    std::vector<int64_t> shape2 { 0, coreNum, SIZE_ALIGN };
    std::vector<int64_t> shape3 { 0, CORE_NUM, FLAG_SIZE };
    std::vector<int64_t> shape4 { aicpu::TOPK_FLAT_ATTR_IDX_COUNT };
    std::vector<int64_t> shape5 { this->queryBatch, 0 };

    desc.addInputTensorDesc(ACL_FLOAT16, shape0.size(), shape0.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_FLOAT16, shape1.size(), shape1.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT32, shape2.size(), shape2.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT16, shape3.size(), shape3.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_INT64, shape4.size(), shape4.data(), ACL_FORMAT_ND);

    desc.addOutputTensorDesc(ACL_FLOAT16, shape5.size(), shape5.data(), ACL_FORMAT_ND);
    desc.addOutputTensorDesc(ACL_INT64, shape5.size(), shape5.data(), ACL_FORMAT_ND);

    topkComputeOps.reset();
    topkComputeOps = CREATE_UNIQUE_PTR(AscendOperator, desc);
    APPERR_RETURN_IF_NOT_LOG(topkComputeOps->init(), APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "topk op init failed");

    return APP_ERR_OK;
}
} // namespace ascend