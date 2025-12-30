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


#ifndef OCK_HCPS_ALGO_ELASTIC_BIT_SET_H
#define OCK_HCPS_ALGO_ELASTIC_BIT_SET_H
#include <cstdint>
#include <sstream>
#include <securec.h>
#include <limits>
#include "ock/log/OckLogger.h"
namespace ock {
namespace hcps {
namespace algo {
/*
@brief 比std::bitset多消耗了24字节
note：实现时为保证性能，保留使用inline
*/
struct OckRefBitSet {
public:
    using WordT = uint64_t;
    ~OckRefBitSet(void) noexcept {}
    OckRefBitSet(uint64_t bitNum, uint64_t *pData)
        : bitCount(bitNum),
          wordCount(std::max((bitCount / (sizeof(uint64_t) * __CHAR_BIT__) +
                             ((bitCount % (sizeof(uint64_t) * __CHAR_BIT__)) > 0 ? 1ULL : 0ULL)), 1ULL)),
          data(pData)
    {}
    OckRefBitSet(const OckRefBitSet &other)
    {
        this->bitCount = other.bitCount;
        this->wordCount = other.wordCount;
        if (this->data != nullptr) {
            delete[] this->data;
            this->data = nullptr;
        }
        this->data = new uint64_t[wordCount];
        uint64_t bytes = this->wordCount * sizeof(WordT);
        if (data != nullptr) {
            if (bytes != 0) {
                auto ret = memcpy_s(data, bytes, other.data, other.wordCount * sizeof(WordT));
                if (ret != EOK) {
                    OCK_HMM_LOG_ERROR("memcpy_s failed, the errorCode is " << ret);
                    delete[] data;
                    data = nullptr;
                }
            }
        }
    }

    OckRefBitSet &operator=(const OckRefBitSet &other)
    {
        this->bitCount = other.bitCount;
        this->wordCount = other.wordCount;
        if (this->data != nullptr) {
            delete[] this->data;
            this->data = nullptr;
        }
        this->data = new uint64_t[wordCount];
        uint64_t bytes = this->wordCount * sizeof(WordT);
        if (data != nullptr) {
            if (bytes != 0) {
                auto ret = memcpy_s(data, bytes, other.data, other.wordCount * sizeof(WordT));
                if (ret != EOK) {
                    OCK_HMM_LOG_ERROR("memcpy_s failed, the errorCode is " << ret);
                    delete[] data;
                    data = nullptr;
                }
            }
        }
        return *this;
    }
    inline uint64_t Size(void) const
    {
        return bitCount;
    }
    inline uint64_t WordCount(void) const
    {
        return wordCount;
    }
    inline const uint64_t *Data(void) const
    {
        return data;
    }
    inline uint64_t *Data(void)
    {
        return data;
    }
    inline void Set(uint64_t bitPos)
    {
        CheckInputPos(bitPos, "OckRefBitSet::Set");
        if (data == nullptr) {
            OCK_HMM_LOG_ERROR("The data is a null pointer.");
            return;
        }
        data[bitPos / (sizeof(uint64_t) * __CHAR_BIT__)] |= (1ULL << (bitPos % (sizeof(uint64_t) * __CHAR_BIT__)));
    }
    inline void Set(uint64_t bitPos, bool value)
    {
        if (value) {
            this->Set(bitPos);
            return;
        }
        CheckInputPos(bitPos, "OckRefBitSet::Set");
        if (data == nullptr) {
            OCK_HMM_LOG_ERROR("The data is a null pointer.");
            return;
        }
        data[bitPos / (sizeof(uint64_t) * __CHAR_BIT__)] &= ~(1ULL << (bitPos % (sizeof(uint64_t) * __CHAR_BIT__)));
    }
    inline void UnSetAll(void)
    {
        if (sizeof(WordT) * wordCount != 0 && data != nullptr) {
            auto ret = memset_s(data, sizeof(WordT) * wordCount, 0, sizeof(WordT) * wordCount);
            if (ret != EOK) {
                OCK_HMM_LOG_ERROR("memset_s failed, the errorCode is " << ret);
                return;
            }
        }
    }
    inline void SetAll(void)
    {
        uint64_t wordBits = sizeof(WordT) * __CHAR_BIT__;
        uint64_t mask = (1ULL << (bitCount % wordBits)) - 1ULL;
        if (data == nullptr) {
            OCK_HMM_LOG_ERROR("The data is a null pointer.");
            return;
        }
        for (uint64_t i = 0; i < wordCount; ++i) {
            data[i] = std::numeric_limits<WordT>::max();
        }
        if ((bitCount % wordBits) != 0 && wordCount != 0) {
            data[wordCount - 1UL] &= mask;
        }
    }
    /*
    @brief: 从startPos开始的count个bit位设置为0
    */
    inline void UnSetRange(uint64_t startPos, uint64_t count)
    {
        CheckInputPos(startPos + count - 1ULL, "OckRefBitSet::UnSetRange");
        uint64_t wordBits = sizeof(WordT) * __CHAR_BIT__;
        uint64_t startSeg = startPos / wordBits;
        uint64_t startOffset = startPos % wordBits;
        uint64_t endSeg = (startPos + count - 1ULL) / wordBits;
        uint64_t endOffset = (startPos + count - 1ULL) % wordBits;

        if (data == nullptr) {
            OCK_HMM_LOG_ERROR("The data is a null pointer.");
            return;
        }

        for (uint64_t i = startSeg + 1ULL; i < endSeg; i++) {
            data[i] = 0ULL;
        }
        data[startSeg] &= ((1ULL << startOffset) - 1ULL);
        data[endSeg] &= ((endOffset == (wordBits - 1ULL)) ? 0ULL : ~((1ULL << (endOffset + 1ULL)) - 1ULL));
    }
    /*
    @brief: 从startPos开始的count个bit位设置为1
    */
    inline void SetRange(uint64_t startPos, uint64_t count)
    {
        CheckInputPos(startPos + count - 1ULL, "OckRefBitSet::SetRange");
        uint64_t wordBits = sizeof(WordT) * __CHAR_BIT__;
        uint64_t startSeg = startPos / wordBits;
        uint64_t startOffset = startPos % wordBits;
        uint64_t endSeg = (startPos + count - 1ULL) / wordBits;
        uint64_t endOffset = (startPos + count - 1ULL) % wordBits;

        if (data == nullptr) {
            OCK_HMM_LOG_ERROR("The data is a null pointer.");
            return;
        }

        for (uint64_t i = startSeg + 1ULL; i < endSeg; i++) {
            data[i] = std::numeric_limits<WordT>::max();
        }
        data[startSeg] |= (~((1ULL << startOffset) - 1ULL));
        data[endSeg] |= ((endOffset == (wordBits - 1ULL)) ? std::numeric_limits<WordT>::max() :
                                                            (1ULL << (endOffset + 1ULL)) - 1ULL);
    }
    inline bool At(uint64_t bitPos) const
    {
        CheckInputPos(bitPos, "OckRefBitSet::At");
        if (data == nullptr) {
            OCK_HMM_LOG_ERROR("The data is a null pointer.");
            return false;
        }
        return data[bitPos / (sizeof(uint64_t) * __CHAR_BIT__)] &
            (1ULL << (bitPos % (sizeof(uint64_t) * __CHAR_BIT__)));
    }
    inline bool operator[](uint64_t bitPos) const
    {
        CheckInputPos(bitPos, "OckRefBitSet::[]");
        if (data == nullptr) {
            OCK_HMM_LOG_ERROR("The data is a null pointer.");
            return false;
        }
        return data[bitPos / (sizeof(uint64_t) * __CHAR_BIT__)] &
            (1ULL << (bitPos % (sizeof(uint64_t) * __CHAR_BIT__)));
    }
    inline bool HasSetWord(uint64_t startWord, uint64_t endWord) const
    {
        if (data == nullptr) {
            OCK_HMM_LOG_ERROR("The data is a null pointer.");
            return false;
        }
        if (startWord >= wordCount || endWord > wordCount) {
            OCK_HMM_LOG_ERROR("The word position is out of range!");
            return false;
        }
        for (uint64_t pos = startWord; pos < endWord; ++pos) {
            if (data[pos]) {
                return true;
            }
        }
        return false;
    }
    inline bool HasSetBit(uint64_t startPos, uint64_t posCount) const
    {
        CheckInputPos(startPos + posCount, "OckRefBitSet::HasSetBit");
        uint64_t startWord = startPos / (sizeof(uint64_t) * __CHAR_BIT__);
        uint64_t alignStartOffset = startPos % (sizeof(uint64_t) * __CHAR_BIT__);
        uint64_t endWord = (startPos + posCount) / (sizeof(uint64_t) * __CHAR_BIT__);
        uint64_t alignEndOffset = (startPos + posCount) % (sizeof(uint64_t) * __CHAR_BIT__);
        if (alignStartOffset > 0) {
            if (HasSetBitImpl(startPos,
                startPos + std::min(posCount, (sizeof(uint64_t) * __CHAR_BIT__) - alignStartOffset))) {
                return true;
            }
            if (alignEndOffset > 0) {
                if (HasSetBitImpl(std::max(startPos, endWord * (sizeof(uint64_t) * __CHAR_BIT__)),
                    endWord * (sizeof(uint64_t) * __CHAR_BIT__) + alignEndOffset)) {
                    return true;
                }
                if (endWord > 0) {
                    return HasSetWord(startWord + 1ULL, endWord - 1ULL);
                }
                return false;
            }
            return HasSetWord(startWord + 1ULL, endWord);
        } else {
            if (alignEndOffset > 0) {
                if (HasSetBitImpl(std::max(startPos, endWord * (sizeof(uint64_t) * __CHAR_BIT__)),
                    endWord * (sizeof(uint64_t) * __CHAR_BIT__) + alignEndOffset)) {
                    return true;
                }
                if (endWord > 0) {
                    return HasSetWord(startWord, endWord - 1ULL);
                }
                return false;
            }
            return HasSetWord(startWord, endWord);
        }
    }
    inline WordT WordAt(uint64_t wordPos) const
    {
        uint64_t ret = 0;
        if (data == nullptr || wordPos >= wordCount) {
            OCK_HMM_LOG_ERROR("The data is a null pointer or the wordPos(" << wordPos << ")is out of range.");
            return ret;
        }
        return data[wordPos];
    }
    inline uint64_t WordBitCount(uint64_t wordPos) const
    {
        uint64_t ret = 0;
        if (data == nullptr || wordPos >= wordCount) {
            OCK_HMM_LOG_ERROR("The data is a null pointer or the wordPos(" << wordPos << ")is out of range.");
            return ret;
        }
        return __builtin_popcountl(data[wordPos]);
    }
    inline uint64_t Count(void) const
    {
        uint64_t ret = 0;
        if (data == nullptr || wordCount == 0) {
            OCK_HMM_LOG_ERROR("The bitset is null.");
            return ret;
        }
        for (uint64_t i = 0; i < wordCount; ++i) {
            ret += __builtin_popcountl(data[i]);
        }
        return ret;
    }
    /*
    @brief: 返回0： 相等， 小于0：小于other, 大于0：大于other
    note：每次对bitset比较64bit，不一定适用于其他排序场景
    */
    inline int Compare(const OckRefBitSet &other) const
    {
        if (data == nullptr || other.data == nullptr) {
            OCK_HMM_LOG_ERROR("The data is a null pointer.");
            return 0;
        }
        uint64_t commonWordCount = std::min(wordCount, other.WordCount());
        for (uint64_t i = 0; i < commonWordCount; ++i) {
            if (data[i] < other.data[i]) {
                return -1;
            } else if (data[i] > other.data[i]) {
                return 1;
            }
        }
        return 0;
    }

    inline bool operator < (const OckRefBitSet &other) const
    {
        return Compare(other) < 0L;
    }
    inline bool operator <= (const OckRefBitSet &other) const
    {
        return Compare(other) <= 0L;
    }
    inline bool operator > (const OckRefBitSet &other) const
    {
        return Compare(other) > 0L;
    }
    inline bool operator >= (const OckRefBitSet &other) const
    {
        return Compare(other) >= 0L;
    }
    inline bool operator == (const OckRefBitSet &other) const
    {
        return Compare(other) == 0L;
    }
    inline bool operator != (const OckRefBitSet &other) const
    {
        return Compare(other) != 0L;
    }
    inline void CopyFrom(const OckRefBitSet &other, uint64_t fromWordPos, uint64_t dstWordPos, uint64_t copyWordLen)
    {
        if (copyWordLen * sizeof(WordT) != 0 && (dstWordPos + copyWordLen - 1ULL) < wordCount &&
            (fromWordPos + copyWordLen - 1ULL) < other.WordCount()) {
            auto ret = memcpy_s(&data[dstWordPos], sizeof(WordT) * (wordCount - dstWordPos), &other.data[fromWordPos],
                                sizeof(WordT) * copyWordLen);
            if (ret != EOK) {
                OCK_HMM_LOG_ERROR("copy data form other bitset failed, the errorCode is " << ret);
                return;
            }
        }
    }
    inline bool Intersect(const OckRefBitSet &other) const
    {
        if (data == nullptr || other.data == nullptr) {
            OCK_HMM_LOG_ERROR("The data is a null pointer.");
            return false;
        }
        uint64_t commonWordCount = std::min(wordCount, other.WordCount());
        for (uint64_t i = 0; i < commonWordCount; ++i) {
            if (data[i] & other.data[i]) {
                return true;
            }
        }
        return false;
    }
    inline uint64_t IntersectCount(const OckRefBitSet &other) const
    {
        uint64_t ret = 0ULL;
        if (data == nullptr || other.data == nullptr) {
            OCK_HMM_LOG_ERROR("The data is a null pointer.");
            return ret;
        }
        uint64_t commonWordCount = std::min(wordCount, other.WordCount());
        for (uint64_t i = 0; i < commonWordCount; ++i) {
            ret += __builtin_popcountl(data[i] & other.data[i]);
        }
        return ret;
    }
    friend std::ostream &operator << (std::ostream &os, const OckRefBitSet &obj)
    {
        for (uint64_t i = 0; i < obj.bitCount; ++i) {
            if (obj[i]) {
                os << "1";
            } else {
                os << "0";
            }
        }
        return os;
    }
    inline void AndWith(const OckRefBitSet &other)
    {
        if (data == nullptr || other.data == nullptr) {
            OCK_HMM_LOG_ERROR("The data is a null pointer.");
            return;
        }
        uint64_t commonWordCount = std::min(wordCount, other.WordCount());
        for (uint64_t i = 0; i < commonWordCount; ++i) {
            this->data[i] &= other.data[i];
        }
    }
    inline void OrWith(const OckRefBitSet &other)
    {
        if (data == nullptr || other.data == nullptr) {
            OCK_HMM_LOG_ERROR("The data is a null pointer.");
            return;
        }
        uint64_t wordBits = sizeof(WordT) * __CHAR_BIT__;
        uint64_t mask = (1ULL << (bitCount % wordBits)) - 1ULL;
        uint64_t commonWordCount = std::min(wordCount, other.WordCount());
        for (uint64_t i = 0; i < commonWordCount; ++i) {
            this->data[i] |= other.data[i];
        }
        if ((bitCount % wordBits) != 0 && wordCount != 0) {
            data[wordCount - 1UL] &= mask;
        }
    }
    /*
    @brief 输入数据的维度占的单词数
    */
    inline uint64_t HashValue(uint64_t dimWord = 4ULL) const
    {
        if (dimWord == 0) {
            OCK_HMM_LOG_ERROR("Input param is 0.");
            return 0;
        }
        if (wordCount % dimWord != 0) {
            std::ostringstream osStr;
            osStr << "HashValue, wordCount(" << wordCount << ") mod dimWord(" << dimWord << ") != 0";
            throw std::runtime_error(osStr.str());
        }
        uint64_t ret = 0ULL;
        if (data == nullptr) {
            OCK_HMM_LOG_ERROR("The data is a null pointer.");
            return ret;
        }
        for (uint64_t i = 0; i < wordCount / dimWord; ++i) {
            uint64_t setCount = 0ULL;
            for (uint64_t j = 0; j < dimWord; ++j) {
                setCount += __builtin_popcountl(data[i * dimWord + j]);
            }
            ret += setCount * (1ULL << (wordCount / dimWord - i));
        }
        return ret;
    }

protected:
    inline bool HasSetBitImpl(uint64_t startPos, uint64_t endPos) const
    {
        for (uint64_t pos = startPos; pos < endPos; ++pos) {
            if (At(pos)) {
                return true;
            }
        }
        return false;
    }
    void CheckInputPos(uint64_t bitPos, const char *msg) const
    {
        if (bitPos >= bitCount) {
            std::ostringstream osStr;
            osStr << msg << ": bitPos(" << bitPos << ") >= bitCount(" << bitCount << ")";
            throw std::range_error(osStr.str());
        }
    }

    uint64_t bitCount{ 0 };
    uint64_t wordCount{ 0 };
    uint64_t *data{ nullptr };
};
struct OckElasticBitSet : public OckRefBitSet {
public:
    using WordT = uint64_t;
    ~OckElasticBitSet(void) noexcept
    {
        delete[] dataHolder;
    }
    OckElasticBitSet(uint64_t bitNum) : OckRefBitSet(bitNum, nullptr), dataHolder(new uint64_t[this->wordCount])
    {
        if (dataHolder != nullptr) {
            this->data = dataHolder;
            if (wordCount != 0) {
                auto ret = memset_s(data, wordCount * sizeof(uint64_t), 0, wordCount * sizeof(uint64_t));
                if (ret != EOK) {
                    OCK_HMM_LOG_ERROR("memset_s failed, the errorCode is " << ret);
                    delete[] dataHolder;
                    dataHolder = nullptr;
                }
            }
        }
    }
    OckElasticBitSet(const std::string &bitStr)
        : OckRefBitSet(bitStr.size(), nullptr), dataHolder(new uint64_t[this->wordCount])
    {
        if (dataHolder != nullptr) {
            this->data = dataHolder;
            if (wordCount != 0) {
                auto ret = memset_s(data, wordCount * sizeof(uint64_t), 0, wordCount * sizeof(uint64_t));
                if (ret != EOK) {
                    OCK_HMM_LOG_ERROR("memset_s failed, the errorCode is " << ret);
                    delete[] dataHolder;
                    dataHolder = nullptr;
                }
            }
            for (uint64_t pos = 0; pos < bitStr.size(); ++pos) {
                if (bitStr[pos] == '1') {
                    data[pos / (sizeof(uint64_t) * __CHAR_BIT__)] |=
                        (1ULL << (pos % (sizeof(uint64_t) * __CHAR_BIT__)));
                }
            }
        }
    }
    OckElasticBitSet(const OckElasticBitSet &other) : OckRefBitSet(other)
    {
        this->bitCount = other.bitCount;
        this->wordCount = other.wordCount;
        size_t bytes = wordCount * sizeof(OckRefBitSet::WordT);
        if (dataHolder != nullptr) {
            delete[] dataHolder;
            dataHolder = nullptr;
        }
        if (this->data != nullptr) {
            delete[] this->data;
            this->data = nullptr;
        }
        dataHolder = new uint64_t[wordCount];
        if (dataHolder != nullptr) {
            if (bytes != 0) {
                auto ret = memcpy_s(dataHolder, bytes, other.data, bytes);
                if (ret != EOK) {
                    OCK_HMM_LOG_ERROR("memcpy_s failed, the errorCode is " << ret);
                    delete[] dataHolder;
                    dataHolder = nullptr;
                }
            }
            this->data = dataHolder;
        }
    }
    OckElasticBitSet &operator=(const OckElasticBitSet &other)
    {
        this->bitCount = other.bitCount;
        this->wordCount = other.wordCount;
        size_t bytes = wordCount * sizeof(OckRefBitSet::WordT);
        if (dataHolder != nullptr) {
            delete[] dataHolder;
            dataHolder = nullptr;
        }
        if (this->data != nullptr) {
            delete[] this->data;
            this->data = nullptr;
        }
        dataHolder = new uint64_t[wordCount];
        if (dataHolder != nullptr) {
            if (bytes != 0) {
                auto ret = memcpy_s(dataHolder, bytes, other.data, bytes);
                if (ret != EOK) {
                    OCK_HMM_LOG_ERROR("memcpy_s failed, the errorCode is " << ret);
                    delete[] dataHolder;
                    dataHolder = nullptr;
                }
            }
            this->data = dataHolder;
        }
        return *this;
    }

    uint64_t *dataHolder{ nullptr };
};
} // namespace algo
} // namespace hcps
} // namespace ock
#endif
