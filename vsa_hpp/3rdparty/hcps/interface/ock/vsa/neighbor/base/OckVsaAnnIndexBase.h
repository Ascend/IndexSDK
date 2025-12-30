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


#ifndef OCK_VSA_ANN_INDEX_BASE_H
#define OCK_VSA_ANN_INDEX_BASE_H
#include <cstdint>
#include <memory>
#include "ock/hcps/algo/OckTopNQueue.h"
#include "ock/vsa/OckVsaErrorCode.h"
#include "ock/vsa/neighbor/base/OckVsaAnnQueryCondition.h"
#include "ock/vsa/neighbor/base/OckVsaAnnQueryResult.h"
#include "ock/vsa/neighbor/base/OckVsaAnnAddFeatureParam.h"
#include "ock/vsa/neighbor/base/OckVsaAnnCreateParam.h"

namespace ock {
namespace vsa {
namespace neighbor {

using OckFloatTopNQueue = hcps::algo::OckTopNQueue<float, uint64_t, hcps::algo::OckCompareDescAdapter<float, uint64_t>>;
/*
@param _Data 底库数据类型, 例如int8_t
@param DimSizeTemp 底库数据维度, 例如256维
@param KeyTraitTemp 关键属性描述，例如时间+空间属性
*/
template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
class OckVsaAnnIndexBase {
public:
    using DataT = DataTemp;
    using KeyTraitT = KeyTraitTemp;
    using KeyTypeTupleT = typename KeyTraitTemp::KeyTypeTuple;
    virtual ~OckVsaAnnIndexBase() noexcept = default;

    virtual OckVsaErrorCode AddFeature(const OckVsaAnnAddFeatureParam<DataTemp, KeyTraitTemp> &featureParam) = 0;

    virtual OckVsaErrorCode Search(const OckVsaAnnQueryCondition<DataTemp, DimSizeTemp, KeyTraitTemp> &queryCond,
        OckVsaAnnQueryResult<DataTemp, KeyTraitTemp> &outResult) = 0;

    /*
    @brief 获取当前特征数据条数
    */
    virtual uint64_t GetFeatureNum(void) const = 0;

    virtual OckVsaErrorCode GetFeatureByLabel(uint64_t count, const int64_t *labels, DataTemp *features) const = 0;
    virtual OckVsaErrorCode GetFeatureAttrByLabel(
        uint64_t count, const int64_t *labels, KeyTypeTupleT *attributes) const = 0;
    virtual uintptr_t GetCustomAttrByBlockId(uint32_t blockId, OckVsaErrorCode &errorCode) const = 0;
    virtual uint32_t GetCustomAttrBlockCount(void) const = 0;

    virtual OckVsaErrorCode DeleteFeatureByLabel(uint64_t count, const int64_t *labels) = 0;
    virtual OckVsaErrorCode DeleteFeatureByToken(uint64_t count, const uint32_t *tokens) = 0;
};
}  // namespace neighbor
}  // namespace vsa
}  // namespace ock
#endif