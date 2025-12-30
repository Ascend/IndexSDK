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

#include "ascendhost/include/impl/TSSuperBase.h"

#ifndef FEATURERETRIEVAL_HPPTS_H
#define FEATURERETRIEVAL_HPPTS_H
namespace ascend {
namespace {
constexpr uint64_t HPP_DIM = 256;
constexpr uint64_t NORM_BYTE_SIZE = 2;
} // namespace
class TSInt8FlatHPPCosFactory {
public:
    static std::shared_ptr<TSSuperBase> Create(uint32_t dims, uint32_t deviceId, uint32_t tokenNum,
        uint64_t maxFeatureRowCount, uint32_t extKeyAttrsByteSize, uint32_t extKeyAttrBlockSize);
};
}

#endif // FEATURERETRIEVAL_HPPTS_H