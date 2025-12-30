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


#ifndef OCK_HCPS_ALGO_BIT_SET_H
#define OCK_HCPS_ALGO_BIT_SET_H
#include <cstdint>
#include <iostream>
#include <securec.h>
#include <stack>
#include "ock/log/OckLogger.h"
#include "ock/hcps/algo/OckElasticBitSet.h"
namespace ock {
namespace hcps {
namespace algo {
/*
@brief 比std::bitset多消耗了24字节
*/
template <uint64_t BitCountTemp, uint64_t DimSizeData,
    uint64_t WordCountTemp = (BitCountTemp / (__CHAR_BIT__ * sizeof(uint64_t)) +
    ((BitCountTemp % (__CHAR_BIT__ * sizeof(uint64_t)) > 0) ? 1ULL : 0UL)),
    uint64_t BitCountOfDataTemp = BitCountTemp / DimSizeData>
struct OckBaseBitSet : public OckRefBitSet {
public:
    static_assert(BitCountTemp == BitCountOfDataTemp * DimSizeData, "_bitCount must be divisible by DimSizeData.");
    static_assert(WordCountTemp * __CHAR_BIT__ * sizeof(uint64_t) >= BitCountTemp,
        "The number of WordCountTemp allocated must be greater than or equal to BitCountTemp");
    using WordT = uint64_t;
    inline uint64_t DimSize(void) const
    {
        return DimSizeData;
    }
    inline uint64_t BitCountOfData(void) const
    {
        return BitCountOfDataTemp;
    }
    OckBaseBitSet(void) : OckRefBitSet(BitCountTemp, nullptr), dataHolder{ 0 }
    {
        this->data = dataHolder;
        if (WordCountTemp != 0) {
            auto ret = memset_s(data, WordCountTemp * sizeof(uint64_t), 0, WordCountTemp * sizeof(uint64_t));
            if (ret != EOK) {
                OCK_HMM_LOG_ERROR("memset_s failed, the errorCode is " << ret);
            }
        }
    }
    OckBaseBitSet(const std::string &bitStr) : OckRefBitSet(BitCountTemp, nullptr), dataHolder{ 0 }
    {
        this->data = dataHolder;
        if (WordCountTemp != 0) {
            auto ret = memset_s(data, WordCountTemp * sizeof(uint64_t), 0, WordCountTemp * sizeof(uint64_t));
            if (ret != EOK) {
                OCK_HMM_LOG_ERROR("memset_s failed, the errorCode is " << ret);
            }
        }
        for (uint64_t pos = 0; pos < bitStr.size() && pos < BitCountTemp; ++pos) {
            if (bitStr[pos] == '1') {
                dataHolder[pos / (__CHAR_BIT__ * sizeof(uint64_t))] |=
                    (1ULL << (pos % (__CHAR_BIT__ * sizeof(uint64_t))));
            }
        }
    }
    inline void SetExt(uint64_t dim, uint64_t valBitPos)
    {
        this->CheckDimPos(dim, "OckBaseBitSet::SetExt");
        this->CheckValBitPos(valBitPos, "OckBaseBitSet::SetExt");
        this->Set(valBitPos * DimSizeData + dim);
    }
    inline void RepairStruct(void)
    {
        this->data = dataHolder;
        this->bitCount = BitCountTemp;
        this->wordCount = WordCountTemp;
    }

    WordT dataHolder[WordCountTemp];

protected:
    void CheckValBitPos(uint64_t bitPos, const char *msg) const
    {
        if (bitPos >= BitCountOfDataTemp) {
            INNER_OCK_THROW(std::range_error,
                msg << ": bitPos(" << bitPos << ") >= BitCountOfData(" << BitCountOfDataTemp << ")", "HCPS");
        }
    }
    void CheckDimPos(uint64_t dimPos, const char *msg) const
    {
        if (dimPos >= DimSizeData) {
            INNER_OCK_THROW(std::range_error, msg << ": dimPos(" << dimPos << ") >= DimSize(" << DimSizeData << ")",
                "HCPS");
        }
    }
};
template <uint64_t BitCountTemp, uint64_t DimSizeData,
    uint64_t WordCountTemp = (BitCountTemp / (__CHAR_BIT__ * sizeof(uint64_t)) +
    ((BitCountTemp % (__CHAR_BIT__ * sizeof(uint64_t)) > 0) ? 1ULL : 0UL)),
    uint64_t BitCountOfDataTemp = BitCountTemp / DimSizeData>
struct OckBitSet : public OckBaseBitSet<BitCountTemp, DimSizeData> {
public:
    inline uint64_t HashValue(void) const
    {
        uint64_t ret = 0ULL;
        for (uint64_t i = 0; i < BitCountOfDataTemp; ++i) {
            for (uint64_t dim = 0; dim < DimSizeData; ++dim) {
                ret += this->At(i * DimSizeData + dim) * (1ULL << (BitCountOfDataTemp - 1UL - i));
            }
        }
        return ret;
    }
    inline uint64_t CountExt(uint64_t valBitPos) const
    {
        this->CheckValBitPos(valBitPos, "OckBaseBitSet::CountExt");
        uint64_t ret = 0;
        for (uint64_t i = 0; i < DimSizeData; ++i) {
            ret += this->At(valBitPos * DimSizeData + i);
        }
        return ret;
    }
};

template <uint64_t BitCountTemp, uint64_t DimSizeData>
struct OckBitSetWithPos : public OckBitSet<BitCountTemp, DimSizeData> {
    using BaseT = OckBitSet<BitCountTemp, DimSizeData>;
    using WordT = typename BaseT::WordT;
    OckBitSetWithPos(void) : BaseT(), pos(0) {}
    OckBitSetWithPos(const std::string &bitStr) : BaseT(bitStr), pos(0) {}
    uint32_t pos;
};

struct OckBitRefSetWithPos : public OckRefBitSet {
    using BaseT = OckRefBitSet;
    OckBitRefSetWithPos(uint64_t bitNum, uint64_t *pData) : BaseT(bitNum, pData), pos(0UL) {}
    OckBitRefSetWithPos(void) : BaseT(0ULL, nullptr), pos(0UL) {}
    uint32_t pos;
};
} // namespace algo
} // namespace hcps
} // namespace ock
#endif
