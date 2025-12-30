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


#ifndef OCK_HCPS_ALGO_SHAPE_IMPL_H
#define OCK_HCPS_ALGO_SHAPE_IMPL_H
#include <cstdint>
#include <memory>
#include <vector>
#include "securec.h"
#include "ock/hcps/error/OckHcpsErrorCode.h"
#include "ock/log/OckHcpsLogger.h"

namespace ock {
namespace hcps {
namespace algo {
template <typename DataTemp, uint64_t DimSizeData, uint64_t CubeAlignTemp, uint64_t CubeBandWidthBytesTemp>
OckShape<DataTemp, DimSizeData, CubeAlignTemp, CubeBandWidthBytesTemp>::OckShape(uintptr_t address, uint64_t byteCount,
    uint64_t usedRowCount)
    : validateRowCount(usedRowCount),
      subDim(CubeBandWidthBytesTemp / sizeof(DataTemp)),
      byteSize(byteCount),
      addr(address),
      halfCubeBandWidthBytes(CubeBandWidthBytesTemp >> 1),
      subNum(DimSizeData / subDim),
      currentAddr(addr +
    ((validateRowCount / CubeAlignTemp) * CubeAlignTemp * subNum + (validateRowCount % CubeAlignTemp)) *
    CubeBandWidthBytesTemp)
{}

template <typename DataTemp, uint64_t DimSizeData, uint64_t CubeAlignTemp, uint64_t CubeBandWidthBytesTemp>
void OckShape<DataTemp, DimSizeData, CubeAlignTemp, CubeBandWidthBytesTemp>::AddData(const DataTemp *pSrc)
{
    if (pSrc == nullptr) {
        OCK_HCPS_LOG_ERROR("In function \"AddData\", the source vector is invalid");
        return;
    }
    if (validateRowCount * DimSizeData * sizeof(DataTemp) >= byteSize) {
        OCK_HCPS_LOG_ERROR("In function \"AddData\", the shape(" << *this << ") is already full.");
        return;
    }
    DataTemp *pDst = reinterpret_cast<DataTemp *>(currentAddr);
    for (uint64_t i = 0; i < subNum; ++i) {
        *(__uint128_t *)(pDst) = *(reinterpret_cast<const __uint128_t *>(pSrc));
        *(__uint128_t *)(pDst + halfCubeBandWidthBytes / sizeof(DataTemp)) =
            *(reinterpret_cast<const __uint128_t *>(pSrc + halfCubeBandWidthBytes / sizeof(DataTemp)));
        pDst += subDim * CubeAlignTemp;
        pSrc += subDim;
    }
    validateRowCount++;
    if (validateRowCount % CubeAlignTemp != 0) {
        currentAddr += CubeBandWidthBytesTemp;
    } else {
        // (subDim + (subNum - 1) * CubeAlignTemp * subDim) * sizeof(DataTemp)
        currentAddr += (1 + (subNum - 1) * CubeAlignTemp) * CubeBandWidthBytesTemp;
    }
}

template <typename DataTemp, uint64_t DimSizeData, uint64_t CubeAlignTemp, uint64_t CubeBandWidthBytesTemp>
void OckShape<DataTemp, DimSizeData, CubeAlignTemp, CubeBandWidthBytesTemp>::AddData(const DataTemp *pSrc,
    uint64_t rowCount)
{
    if (pSrc == nullptr) {
        OCK_HCPS_LOG_ERROR("In function \"AddData\", the source address is invalid");
        return;
    }
    if ((validateRowCount + rowCount) * DimSizeData * sizeof(DataTemp) > byteSize) {
        OCK_HCPS_LOG_ERROR("In function \"AddData\", the shape(" << *this << ") is too small to accept other " <<
            rowCount << " vectors.");
        return;
    }
    DataTemp *pDst = reinterpret_cast<DataTemp *>(currentAddr);
    const uint64_t blockSize = subDim * CubeAlignTemp;
    uint64_t regionNum = validateRowCount / CubeAlignTemp;
    uint64_t regionSize = DimSizeData * CubeAlignTemp;
    for (uint64_t i = 0; i < rowCount; ++i) {
        uint64_t idx = validateRowCount / CubeAlignTemp - regionNum;
        uint64_t offset = i % CubeAlignTemp;
        DataTemp *outPtr = pDst + idx * regionSize + offset * subDim;
        for (uint64_t j = 0; j < subNum; ++j) {
            *(__uint128_t *)(outPtr) = *(reinterpret_cast<const __uint128_t *>(pSrc));
            *(__uint128_t *)(outPtr + halfCubeBandWidthBytes / sizeof(DataTemp)) =
                *(reinterpret_cast<const __uint128_t *>(pSrc + halfCubeBandWidthBytes / sizeof(DataTemp)));
            pSrc += subDim;
            outPtr += blockSize;
        }
        validateRowCount++;
    }
    currentAddr += ((validateRowCount / CubeAlignTemp - regionNum) * regionSize + (rowCount % CubeAlignTemp) * subDim) *
        sizeof(DataTemp);
}

template <typename DataTemp, uint64_t DimSizeData, uint64_t CubeAlignTemp, uint64_t CubeBandWidthBytesTemp>
void OckShape<DataTemp, DimSizeData, CubeAlignTemp, CubeBandWidthBytesTemp>::GetData(uint64_t rowId,
    DataTemp *pOut) const
{
    if (rowId >= validateRowCount) {
        OCK_HCPS_LOG_ERROR("In function \"GetData\", The rowId(" << rowId << ") >= validateRowCount(" <<
            validateRowCount << ").");
        return;
    }
    DataTemp *pSrc = reinterpret_cast<DataTemp *>(addr) + (rowId / CubeAlignTemp) * CubeAlignTemp * DimSizeData +
        (rowId % CubeAlignTemp) * subDim;
    for (uint64_t i = 0; i < subNum; ++i) {
        *(__uint128_t *)(pOut) = *(reinterpret_cast<const __uint128_t *>(pSrc));
        *(__uint128_t *)(pOut + halfCubeBandWidthBytes / sizeof(DataTemp)) =
            *(reinterpret_cast<const __uint128_t *>(pSrc + halfCubeBandWidthBytes / sizeof(DataTemp)));
        pSrc += subDim * CubeAlignTemp;
        pOut += subDim;
    }
}

template <typename DataTemp, uint64_t DimSizeData, uint64_t CubeAlignTemp, uint64_t CubeBandWidthBytesTemp>
void OckShape<DataTemp, DimSizeData, CubeAlignTemp, CubeBandWidthBytesTemp>::CopyFrom(uint64_t curRowId,
    const OckShape<DataTemp, DimSizeData> &other, uint64_t otherRowId)
{
    // if语句为真的概率很小，提示编译器优化该条if语句
    if (__builtin_expect(curRowId >= validateRowCount, 0)) {
        OCK_HCPS_LOG_ERROR("In function \"CopyFrom\", the rowId(" << curRowId << ") >= validateRowCount(" <<
            validateRowCount << ").");
        return;
    }
    if (__builtin_expect(otherRowId >= other.ValidateRowCount(), 0)) {
        OCK_HCPS_LOG_ERROR("In function \"CopyFrom\", the other's rowId(" << otherRowId <<
            ") >= other's validateRowCount(" << other.ValidateRowCount() << ").");
        return;
    }

    DataTemp *pSrc = reinterpret_cast<DataTemp *>(other.Addr()) +
        (otherRowId / CubeAlignTemp) * CubeAlignTemp * DimSizeData + (otherRowId % CubeAlignTemp) * subDim;
    DataTemp *pDst = reinterpret_cast<DataTemp *>(addr) + (curRowId / CubeAlignTemp) * CubeAlignTemp * DimSizeData +
        (curRowId % CubeAlignTemp) * subDim;

    for (uint64_t i = 0; i < subNum; ++i) {
        *(__uint128_t *)(pDst) = *(reinterpret_cast<const __uint128_t *>(pSrc));
        *(__uint128_t *)(pDst + halfCubeBandWidthBytes / sizeof(DataTemp)) =
            *(reinterpret_cast<const __uint128_t *>(pSrc + halfCubeBandWidthBytes / sizeof(DataTemp)));
        pSrc += subDim * CubeAlignTemp;
        pDst += subDim * CubeAlignTemp;
    }
}

template <typename DataTemp, uint64_t DimSizeData, uint64_t CubeAlignTemp, uint64_t CubeBandWidthBytesTemp>
void OckShape<DataTemp, DimSizeData, CubeAlignTemp, CubeBandWidthBytesTemp>::Restore(DataTemp *pOut) const
{
    if (pOut == nullptr) {
        OCK_HCPS_LOG_ERROR("In function \"Restore\", the destination address is invalid");
        return;
    }
    uintptr_t blockPtr = addr;
    for (uint64_t i = 1; i <= validateRowCount; i++) {
        DataTemp *pSrc = reinterpret_cast<DataTemp *>(blockPtr);
        for (uint64_t j = 0; j < subNum; j++) {
            *(__uint128_t *)(pOut) = *(reinterpret_cast<const __uint128_t *>(pSrc));
            *(__uint128_t *)(pOut + halfCubeBandWidthBytes / sizeof(DataTemp)) =
                *(reinterpret_cast<const __uint128_t *>(pSrc + halfCubeBandWidthBytes / sizeof(DataTemp)));
            pSrc += subDim * CubeAlignTemp;
            pOut += subDim;
        }
        if (i % CubeAlignTemp != 0) {
            blockPtr += CubeBandWidthBytesTemp;
        } else {
            blockPtr += (1 + (subNum - 1) * CubeAlignTemp) * CubeBandWidthBytesTemp;
        }
    }
}

template <typename DataTemp, uint64_t DimSizeData, uint64_t CubeAlignTemp, uint64_t CubeBandWidthBytesTemp>
void OckShape<DataTemp, DimSizeData, CubeAlignTemp, CubeBandWidthBytesTemp>::AddFrom(
    const OckShape<DataTemp, DimSizeData> &other, uint64_t otherRowId)
{
    if (__builtin_expect(validateRowCount * DimSizeData * sizeof(DataTemp) >= byteSize, 0)) {
        OCK_HCPS_LOG_ERROR("In function \"AddFrom\", the shape(" << *this << ") is already full.");
        return;
    }
    if (__builtin_expect(otherRowId >= other.ValidateRowCount(), 0)) {
        OCK_HCPS_LOG_ERROR("In function \"AddFrom\", the other's rowId(" << otherRowId << ") >= validateRowCount(" <<
            other.ValidateRowCount() << ").");
        return;
    }

    DataTemp *pSrc = reinterpret_cast<DataTemp *>(other.Addr()) +
        (otherRowId / CubeAlignTemp) * CubeAlignTemp * DimSizeData + (otherRowId % CubeAlignTemp) * subDim;
    DataTemp *pDst = reinterpret_cast<DataTemp *>(currentAddr);

    for (uint64_t i = 0; i < subNum; ++i) {
        *(__uint128_t *)(pDst) = *(reinterpret_cast<const __uint128_t *>(pSrc));
        *(__uint128_t *)(pDst + halfCubeBandWidthBytes / sizeof(DataTemp)) =
            *(reinterpret_cast<const __uint128_t *>(pSrc + halfCubeBandWidthBytes / sizeof(DataTemp)));
        pSrc += subDim * CubeAlignTemp;
        pDst += subDim * CubeAlignTemp;
    }
    validateRowCount++;
    if (validateRowCount % CubeAlignTemp != 0) {
        currentAddr += CubeBandWidthBytesTemp;
    } else {
        currentAddr += (1 + (subNum - 1) * CubeAlignTemp) * CubeBandWidthBytesTemp;
    }
}

template <typename DataTemp, uint64_t DimSizeData, uint64_t CubeAlignTemp, uint64_t CubeBandWidthBytesTemp>
void OckShape<DataTemp, DimSizeData, CubeAlignTemp, CubeBandWidthBytesTemp>::AddSegment(
    const OckShape<DataTemp, DimSizeData> &other, uint64_t fromRowId, uint64_t rowCount)
{
    if ((fromRowId + rowCount) > other.ValidateRowCount()) {
        OCK_HCPS_LOG_ERROR("In function \"AddSegment\", the fromRowId(" << fromRowId << ") + rowCount(" << rowCount <<
            ") > otherValidateRowCount(" << other.ValidateRowCount() << ").");
        return;
    }
    if ((validateRowCount + rowCount) * DimSizeData * sizeof(DataTemp) > byteSize) {
        OCK_HCPS_LOG_ERROR("In function \"AddSegment\", the shape(" << *this << ") is too small to accept other " <<
            rowCount << " vectors.");
        return;
    }
    if (validateRowCount % CubeAlignTemp == 0 && fromRowId % CubeAlignTemp == 0 && rowCount % CubeAlignTemp == 0) {
        DataTemp *pSrc = reinterpret_cast<DataTemp *>(other.Addr()) + fromRowId * DimSizeData;
        DataTemp *pDst = reinterpret_cast<DataTemp *>(currentAddr);
        if (rowCount * DimSizeData * sizeof(DataTemp) != 0) {
            auto ret = memcpy_s(pDst, rowCount * DimSizeData * sizeof(DataTemp), pSrc,
                                rowCount * DimSizeData * sizeof(DataTemp));
            if (ret != EOK) {
                OCK_HCPS_LOG_ERROR("memcpy_s failed, the errorCode is " << ret);
                return;
            }
        }
        validateRowCount += rowCount;
        currentAddr += rowCount * DimSizeData * sizeof(DataTemp);
        return;
    }
    if (validateRowCount % CubeAlignTemp == 0 && fromRowId % CubeAlignTemp == 0) {
        uint64_t spareCount = rowCount % CubeAlignTemp;
        DataTemp *pSrc = reinterpret_cast<DataTemp *>(other.Addr()) + fromRowId * DimSizeData;
        DataTemp *pDst = reinterpret_cast<DataTemp *>(currentAddr);
        if ((rowCount - spareCount) * DimSizeData * sizeof(DataTemp) != 0) {
            auto ret = memcpy_s(pDst, (rowCount - spareCount) * DimSizeData * sizeof(DataTemp), pSrc,
                                (rowCount - spareCount) * DimSizeData * sizeof(DataTemp));
            if (ret != EOK) {
                OCK_HCPS_LOG_ERROR("memcpy_s failed, the errorCode is " << ret);
                return;
            }
        }
        validateRowCount += (rowCount - spareCount);
        currentAddr += (rowCount - spareCount) * DimSizeData * sizeof(DataTemp);
        fromRowId += (rowCount - spareCount);
        for (uint64_t i = 0; i < spareCount; ++i) {
            AddFrom(other, fromRowId + i);
        }
        return;
    }
    for (uint64_t i = 0; i < rowCount; ++i) {
        AddFrom(other, fromRowId + i);
    }
}

template <typename DataTemp, uint64_t DimSizeData, uint64_t CubeAlignTemp, uint64_t CubeBandWidthBytesTemp>
void OckShape<DataTemp, DimSizeData, CubeAlignTemp, CubeBandWidthBytesTemp>::GetSegment(uint64_t fromRowId,
    uint64_t rowCount, DataTemp *dstData, uint64_t maxDstDataSize)
{
    if (dstData == nullptr) {
        OCK_HCPS_LOG_ERROR("In function \"GetSegment\", the destination address is invalid");
        return;
    }
    if ((fromRowId + rowCount) > validateRowCount) {
        OCK_HCPS_LOG_ERROR("In function \"GetSegment\", the fromRowId(" << fromRowId << ") + rowCount(" << rowCount <<
            ") > ValidateRowCount(" << validateRowCount << ").");
        return;
    }
    if (rowCount * DimSizeData * sizeof(DataTemp) > maxDstDataSize) {
        OCK_HCPS_LOG_ERROR("In function \"GetSegment\", the data size(" << rowCount * DimSizeData * sizeof(DataTemp) <<
            ") > maxDstDataSize(" << maxDstDataSize << ").");
        return;
    }
    if (fromRowId % CubeAlignTemp != 0 || rowCount % CubeAlignTemp != 0) {
        OCK_HCPS_LOG_ERROR("In function \"GetSegment\", the fromRowId(" << fromRowId << ")/rowCount(" << rowCount <<
            ") is not align by " << CubeAlignTemp << ", please check again.");
        return;
    }
    DataTemp *pSrc = reinterpret_cast<DataTemp *>(addr) + fromRowId * DimSizeData;
    if (maxDstDataSize != 0 && rowCount * DimSizeData * sizeof(DataTemp) != 0) {
        auto ret = memcpy_s(dstData, maxDstDataSize, pSrc, rowCount * DimSizeData * sizeof(DataTemp));
        if (ret != EOK) {
            OCK_HCPS_LOG_ERROR("memcpy_s failed, the errorCode is " << ret);
            return;
        }
    }
}

template <typename DataTemp, uint64_t DimSizeData, uint64_t CubeAlignTemp, uint64_t CubeBandWidthBytesTemp>
uint64_t OckShape<DataTemp, DimSizeData, CubeAlignTemp, CubeBandWidthBytesTemp>::ValidateRowCount(void) const
{
    return validateRowCount;
}

template <typename DataTemp, uint64_t DimSizeData, uint64_t CubeAlignTemp, uint64_t CubeBandWidthBytesTemp>
uint64_t OckShape<DataTemp, DimSizeData, CubeAlignTemp, CubeBandWidthBytesTemp>::ByteSize(void) const
{
    return byteSize;
}

template <typename DataTemp, uint64_t DimSizeData, uint64_t CubeAlignTemp, uint64_t CubeBandWidthBytesTemp>
uintptr_t OckShape<DataTemp, DimSizeData, CubeAlignTemp, CubeBandWidthBytesTemp>::Addr(void) const
{
    return addr;
}

template <typename DataTemp, uint64_t DimSizeData, uint64_t CubeAlignTemp, uint64_t CubeBandWidthBytesTemp>
std::ostream &operator << (std::ostream &os,
    const OckShape<DataTemp, DimSizeData, CubeAlignTemp, CubeBandWidthBytesTemp> &ockShape)
{
    return os << "{'size':" << ockShape.ByteSize() << ", 'validateRowCount':" << ockShape.ValidateRowCount() << "}";
}
} // namespace algo
} // namespace hcps
} // namespace ock
#endif // OCK_HCPS_ALGO_SHAPE_IMPL_H
