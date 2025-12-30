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


#ifndef OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_ANN_INDEX_DELETE_FEATURE_H
#define OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_ANN_INDEX_DELETE_FEATURE_H
#include "ock/vsa/neighbor/hpp/impl/OckVsaAnnHPPIndexSystem.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait>
OckVsaErrorCode OckVsaAnnHPPIndex<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>::DeleteFeatureByLabel(uint64_t count,
    const int64_t *labels)
{
    if (labels == nullptr || count > MAX_GET_NUMBER) {
        OCK_VSA_HPP_LOG_ERROR("Input nullptr or delete count exceeds threshold[" << MAX_GET_NUMBER << "].");
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    if (count == 0) {
        return VSA_SUCCESS;
    }
    OCK_CHECK_RETURN_ERRORCODE(hppKernel->SetDroppedByLabel(count, reinterpret_cast<const uint64_t *>(labels)));
    return npuIndex->DeleteFeatureByLabel(count, labels);
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait>
OckVsaErrorCode OckVsaAnnHPPIndex<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>::DeleteFeatureByToken(uint64_t count,
    const uint32_t *tokens)
{
    if (tokens == nullptr || count > MAX_GET_NUMBER) {
        OCK_VSA_HPP_LOG_ERROR("Input nullptr or delete count exceeds threshold[" << MAX_GET_NUMBER << "].");
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    if (count == 0) {
        return VSA_SUCCESS;
    }
    OCK_CHECK_RETURN_ERRORCODE(hppKernel->SetDroppedByToken(count, tokens));
    return npuIndex->DeleteFeatureByToken(count, tokens);
}
} // namespace hpp
} // namespace neighbor
} // namespace vsa
} // namespace ock
#endif