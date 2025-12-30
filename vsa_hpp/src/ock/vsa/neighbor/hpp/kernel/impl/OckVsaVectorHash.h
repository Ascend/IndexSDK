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


#ifndef OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_KERNEL_VECTOR_HASH_H
#define OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_KERNEL_VECTOR_HASH_H
#include <cstdint>
#include <utility>
#include <deque>
#include <memory>
#include <ostream>
#include <securec.h>
#include "ock/hcps/algo/OckBitSet.h"
#include "ock/hcps/hfo/feature/OckHashFeatureGen.h"
#if defined __aarch64__
#include <arm_neon.h>
#endif
namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
template <uint64_t WordCountTemp, uint64_t ByteSizeTemp = WordCountTemp * sizeof(uint64_t),
    uint64_t BitCountTemp = ByteSizeTemp *__CHAR_BIT__>
struct OckVsaVectorHash {
    static uint64_t ByteSize(void)
    {
        return WordCountTemp * sizeof(uint64_t);
    }
    template <typename DataT> void CopyFromSign(const DataT *srcData)
    {
        auto ret = memset_s(data, ByteSizeTemp, 0, ByteSizeTemp);
        if (ret != 0) {
            return;
        }
        for (uint64_t i = 0; i < BitCountTemp; ++i) {
            if (srcData[i] < 0) {
                data[i / (__CHAR_BIT__ * sizeof(uint64_t))] |= (1ULL << i);
            }
        }
    }

    inline bool operator[](uint64_t bitPos) const
    {
        return data[bitPos / (sizeof(uint64_t) * __CHAR_BIT__)] &
            (1ULL << (bitPos % (sizeof(uint64_t) * __CHAR_BIT__)));
    }

    friend bool operator < (const OckVsaVectorHash &lhs, const OckVsaVectorHash &rhs)
    {
        for (uint64_t i = 0; i < WordCountTemp; ++i) {
            if (lhs.data[i] < rhs.data[i]) {
                return true;
            } else if (lhs.data[i] > rhs.data[i]) {
                return false;
            }
        }
        return false;
    }
    friend bool operator < (const OckVsaVectorHash &lhs, const hcps::algo::OckElasticBitSet &rhs)
    {
        for (uint64_t i = 0; i < WordCountTemp; ++i) {
            if (lhs.data[i] < rhs.Data()[i]) {
                return true;
            } else if (lhs.data[i] > rhs.Data()[i]) {
                return false;
            }
        }
        return false;
    }
    friend bool operator < (const hcps::algo::OckElasticBitSet &lhs, const OckVsaVectorHash &rhs)
    {
        for (uint64_t i = 0; i < WordCountTemp; ++i) {
            if (lhs.Data()[i] < rhs.data[i]) {
                return true;
            } else if (lhs.Data()[i] > rhs.data[i]) {
                return false;
            }
        }
        return false;
    }
    friend std::ostream &operator << (std::ostream &os, const OckVsaVectorHash &obj)
    {
        for (uint64_t bitPos = 0; bitPos < BitCountTemp; ++bitPos) {
            if (obj.data[bitPos / (sizeof(uint64_t) * __CHAR_BIT__)] &
                (1ULL << (bitPos % (sizeof(uint64_t) * __CHAR_BIT__)))) {
                os << "1";
            } else {
                os << "0";
            }
        }
        return os;
    }
    uint64_t data[WordCountTemp];
};
} // namespace hpp
} // namespace neighbor
} // namespace vsa
} // namespace ock
#endif