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


#ifndef OCK_HCPS_ALGO_CUSTOMER_ATTR_SHAPE_IMPL_H
#define OCK_HCPS_ALGO_CUSTOMER_ATTR_SHAPE_IMPL_H
#include <cstdint>
#include <memory>
#include <ostream>

namespace ock {
namespace hcps {
namespace algo {
template <typename _IdMapT>
OckCustomerAttrShape<_IdMapT>::OckCustomerAttrShape(uint8_t *address, uint32_t attributeCount, uint32_t blockNum,
    uint64_t blockRowNum)
    : addr(address),
      attrCount(attributeCount),
      blockCount(blockNum),
      blockRowCount(blockRowNum),
      blockSize(attrCount * blockRowCount)
{}

template <typename _IdMapT>
void OckCustomerAttrShape<_IdMapT>::CopyFrom(const OckCustomerAttrShape &other, const _IdMapT *idMap, uint32_t blockId)
{
    if (attrCount == 0) {
        CopyFromZeroAttr(other, idMap, blockId);
    } else if (attrCount == 1) {
        CopyFromOneAttr(other, idMap, blockId);
    } else {
        CopyFromBoost(other, idMap, blockId);
    }
}

template <typename _IdMapT>
void OckCustomerAttrShape<_IdMapT>::CopyFromZeroAttr(const OckCustomerAttrShape &other, const _IdMapT *idMap,
    uint32_t blockId)
{
    return;
}

template <typename _IdMapT>
void OckCustomerAttrShape<_IdMapT>::CopyFromOneAttr(const OckCustomerAttrShape &other, const _IdMapT *idMap,
    uint32_t blockId)
{
    uint8_t *pSrc = other.addr + blockId * blockSize;
    uint8_t *pDst = addr + blockId * blockSize;
    for (uint32_t i = 0; i < blockRowCount; i++) {
        *(pDst + i) = *(pSrc + idMap[i]->pos);
    }
}

template <typename _IdMapT>
void OckCustomerAttrShape<_IdMapT>::CopyFromBoost(const OckCustomerAttrShape &other, const _IdMapT *idMap,
    uint32_t blockId)
{
#if defined __aarch64__
    uint8_t *pSrc = other.addr + blockId * blockSize;
    uint8_t *pDst = addr + blockId * blockSize + blockRowCount - 16UL;
    __uint128_t bigNum = 0;
    for (uint32_t i = 0; i < attrCount; i++) {
        for (int32_t j = blockRowCount - 1; j >= 0; j--) {
            bigNum <<= __CHAR_BIT__;
            bigNum |= *(pSrc + idMap[j]->pos);
            if ((j & 0b1111) == 0) {
                *(__int128_t *)(pDst) = bigNum;
                bigNum = 0;
                pDst -= sizeof(__uint128_t);
            }
        }
        pSrc += blockRowCount;
        pDst += 2UL * blockRowCount;
    }
#else
    uint8_t *pSrc = other.addr + blockId * blockSize;
    uint8_t *pDst = addr + blockId * blockSize;

    uint64_t bigNum = 0;
    int32_t shiftCount = 0;
    for (uint32_t i = 0; i < attrCount; i++) {
        for (uint32_t j = 0; j < blockRowCount; j += sizeof(uint64_t)) {
            shiftCount = sizeof(uint64_t);
            bigNum = 0;
            while (--shiftCount >= 0) {
                bigNum <<= __CHAR_BIT__;
                bigNum |= *(pSrc + idMap[j + shiftCount]->pos);
            }
            *(uint64_t *)(pDst + j) = bigNum;
        }
        pSrc += blockRowCount;
        pDst += blockRowCount;
    }
#endif
}

template <typename _IdMapT>
void OckCustomerAttrShape<_IdMapT>::CopyFromGeneral(const OckCustomerAttrShape &other, const _IdMapT *idMap,
    uint32_t blockId)
{
    uint8_t *pSrc = other.addr + blockId * blockSize;
    uint8_t *pDst = addr + blockId * blockSize;
    for (uint32_t i = 0; i < attrCount; i++) {
        for (uint32_t j = 0; j < blockRowCount; j++) {
            *(pDst + j) = *(pSrc + idMap[j]->pos);
        }
        pSrc += blockRowCount;
        pDst += blockRowCount;
    }
}
} // namespace algo
} // namespace hcps
} // namespace ock
#endif // OCK_HCPS_ALGO_CUSTOMER_ATTR_SHAPE_IMPL_H
