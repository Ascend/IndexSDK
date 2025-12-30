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


#ifndef OCK_HCPS_HFO_HASH_FEATURE_GEN_H
#define OCK_HCPS_HFO_HASH_FEATURE_GEN_H
#include <cstdint>
#include <cmath>
#include <iostream>
#include <securec.h>
#include <limits>
#include <bitset>
#include "ock/hcps/hfo/feature/OckCosFeature.h"
#include "ock/hcps/algo/OckBitSet.h"
#include "ock/utils/OckSafeUtils.h"

#if defined __aarch64__
#include <arm_neon.h>
#endif

namespace ock {
namespace hcps {
namespace hfo {
template <typename DataTemp, uint64_t DimSizeTemp, uint32_t BitLenPerDimTemp = 18UL,
    uint64_t PrecisionTemp = (1U << BitLenPerDimTemp) - 1U,
    typename BitDataTemp = algo::OckBitSet<DimSizeTemp * BitLenPerDimTemp, DimSizeTemp>>
struct OckHashFeatureGen {
    using DataT = DataTemp;
    using BitDataT = BitDataTemp;
    using OckCosLenT = OckCosLen<DataTemp, PrecisionTemp>;
    static_assert(BitLenPerDimTemp < 64UL, "BitLenPerDimTemp must less then 64.");
    static_assert(PrecisionTemp <= ((1ULL << BitLenPerDimTemp) - 1U), "PrecisionTemp must match BitLenPerDimTemp.");
    static void Gen(const DataTemp *fromData, BitDataTemp &toBitSet)
    {
        double factor = OckCosLenT::NormFactorImpl(fromData, DimSizeTemp);
        for (uint64_t i = 0; i < DimSizeTemp; ++i) {
            auto data = uint64_t(fromData[i] * factor + 0.5); // 四舍五入
            for (uint32_t interval = 0; interval < BitLenPerDimTemp; ++interval) {
                if (data & (1ULL << (BitLenPerDimTemp - interval - 1UL))) {
                    toBitSet.Set(interval * DimSizeTemp + i);
                }
            }
        }
    }
};
template <uint64_t DimSizeTemp, uint64_t PrecisionTemp, typename BitDataTemp>
struct OckHashFeatureGen<int8_t, DimSizeTemp, 16ULL, PrecisionTemp, BitDataTemp> {
    using DataT = int8_t;
    using BitDataT = BitDataTemp;
    using OckCosLenT = OckCosLen<int8_t, 16383UL>;
    using OckCosLenTSecond = OckCosLen<uint16_t, 65535UL>;
    static void Gen(const int8_t *fromData, BitDataTemp &toBitSet)
    {
        GenX86(fromData, toBitSet);
    }
    static void GenX86(const int8_t *fromData, BitDataTemp &toBitSet)
    {
        double factor = OckCosLenT::NormFactorImpl(fromData, DimSizeTemp);
        for (uint64_t i = 0; i < DimSizeTemp; ++i) {
            uint16_t data = uint16_t(int16_t(fromData[i] * factor + 0.5f) + 16384UL); // 四舍五入
            for (uint32_t interval = 0; interval < 16ULL; ++interval) {
                if (data & (1ULL << (16ULL - interval - 1UL))) {
                    toBitSet.Set(interval * DimSizeTemp + i);
                }
            }
        }
    }
    /*
    @param tmpDatas 是一个 uint16_t[DimSizeTemp]的数组，用户需要提前申请好内存
    */
    static void Gen(const int8_t *fromData, BitDataTemp &toBitSet, uint16_t *tmpDatas)
    {
        GenX86(fromData, toBitSet, tmpDatas);
    }
    static void GenX86(const int8_t *fromData, BitDataTemp &toBitSet, uint16_t *tmpDatas)
    {
        double factor = OckCosLenT::NormFactorImpl(fromData, DimSizeTemp);
        for (uint64_t i = 0; i < DimSizeTemp; ++i) {
            tmpDatas[i] = uint16_t(int16_t(fromData[i] * factor + 0.5f) + 16384UL); // 四舍五入
        }
        factor = OckCosLenTSecond::NormFactorImpl(tmpDatas, DimSizeTemp);
        for (uint64_t i = 0; i < DimSizeTemp; ++i) {
            uint16_t data = uint16_t(fromData[i] * factor + 0.5); // 四舍五入
            for (uint32_t interval = 0; interval < 16ULL; ++interval) {
                if (data & (1ULL << (16ULL - interval - 1UL))) {
                    toBitSet.Set(interval * DimSizeTemp + i);
                }
            }
        }
    }
};
} // namespace hfo
} // namespace hcps
} // namespace ock
#endif