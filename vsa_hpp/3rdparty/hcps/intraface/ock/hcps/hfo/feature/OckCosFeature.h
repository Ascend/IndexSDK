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


#ifndef OCK_HCPS_HFO_COS_FEATURE_H
#define OCK_HCPS_HFO_COS_FEATURE_H
#include <cstdint>
#include <cmath>
#include <ostream>
#include <limits>
#include <cfloat>
#include "ock/utils/OckSafeUtils.h"

#if defined __aarch64__
#include <arm_neon.h>
#endif

namespace ock {
namespace hcps {
namespace hfo {
const float FLOAT_ZERO_CMP = 1e-8F;
const double DISTANCE_ZERO_THRESHOLD = 0.0001;
template <typename DataTemp, uint64_t PrecisionTemp = ((1UL << (8UL * sizeof(DataTemp) + 8U)) - 1UL)> struct OckCosLen {
    template <typename _FeatureT> static double Len(const _FeatureT &feature, uint64_t dimSize)
    {
        return LenImpl(feature.data, dimSize);
    }
    template <typename _FeatureT> static double NormFactor(const _FeatureT &feature, uint64_t dimSize)
    {
        return NormFactorImpl(feature.data, dimSize);
    }
    static double NormFactorImpl(const DataTemp *data, uint64_t dimSize)
    {
        double lenSquare = 0;
        for (uint64_t i = 0; i < dimSize; ++i) {
            lenSquare += (double)data[i] * (double)data[i];
        }
        return ToFactor(lenSquare);
    }
    static double NormFactorImplARM(const uint8_t *data, uint64_t dimSize)
    {
#if defined __aarch64__
        double lenSquare = 0;
        uint64_t parallNum = 8;
        uint16x8_t mul_vec;
        uint8x8_t left_vec;
        uint8x8_t right_vec;
        for (uint64_t k = 0; k < dimSize; k += parallNum) {
            left_vec = vld1_u8(data + k);
            right_vec = vld1_u8(data + k);
            // uint16x8_t	vmull_u8	(uint8x8_t a, uint8x8_t b)
            mul_vec = vmull_u8(left_vec, right_vec);
            // uint32_t	    vaddlvq_u16	(uint16x8_t a)
            lenSquare += vaddlvq_u16(mul_vec);
        }
        return ToFactor(lenSquare);
#else
        return NormFactorImpl(data, dimSize);
#endif
    }
    static double ToFactor(double lenSquare)
    {
        if (fabs(lenSquare - 0) < FLOAT_ZERO_CMP) {
            return 0;
        }
        auto lenSqrt = std::sqrt(lenSquare);
        auto divResult =
            (fabs(lenSqrt) < DBL_EPSILON) ? std::numeric_limits<decltype(1.0 + lenSqrt)>::max() : (1.0 / lenSqrt);
        return divResult * PrecisionTemp;
    }
    static double LenImpl(const DataTemp *data, uint64_t dimSize)
    {
        double vecLen = 0;
        for (uint64_t i = 0; i < dimSize; ++i) {
            vecLen += (double)data[i] * (double)data[i];
        }
        return std::sqrt(vecLen);
    }
};
} // namespace hfo
} // namespace hcps
} // namespace ock
#endif
