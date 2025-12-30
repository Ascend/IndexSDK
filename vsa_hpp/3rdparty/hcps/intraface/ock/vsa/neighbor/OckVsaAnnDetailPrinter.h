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


#ifndef OCK_VSA_NEIGHBOR_DETAIL_PRINTER_H
#define OCK_VSA_NEIGHBOR_DETAIL_PRINTER_H
#include <sstream>
#include "ock/log/OckHcpsLogger.h"
#include "ock/acladapter/utils/OckAscendFp16.h"
#include "ock/vsa/neighbor/base/OckVsaAnnFeatureSet.h"
#include "ock/vsa/neighbor/npu/OckVsaAnnNpuBlockGroup.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace printer {
template <typename DataTemp, typename _ToDataT>
struct DataConvertorAdapter {
    _ToDataT operator()(DataTemp data) const
    {
        return _ToDataT(data);
    }
};
template <typename _ToDataT>
struct DataConvertorAdapter<int8_t, _ToDataT> {
    _ToDataT operator()(int8_t data) const
    {
        return _ToDataT(static_cast<uint8_t>(data));
    }
};
struct SpaceRemainderConvertorAdapter {
    uint16_t operator()(uint16_t data) const
    {
        return static_cast<uint16_t>(data) & 0x00FFUL;
    }
};
template <typename DataTemp>
struct InvalidDataAssertAdapter {
    InvalidDataAssertAdapter(DataTemp value) : expectValue(value)
    {}
    bool operator()(DataTemp data) const
    {
        return data == expectValue;
    }
    DataTemp expectValue;
};
template <typename DataTemp, typename DataCvtT = DataConvertorAdapter<DataTemp, DataTemp>>
void PrintAnyFeature(hmm::OckHmmHMObject &hmo, uint64_t dimSize, bool forcePrint = false,
    const std::string &name = "Feature", uint64_t maxPrintCount = 10ULL)
{
    if (forcePrint == false && OckLogger::Instance().GetStartLevel() > OCK_LOG_LEVEL_INFO) {
        return;
    }
    if (dimSize == 0UL) {
        OCK_HCPS_LOG_DEBUG(name << ", typename=" << typeid(DataTemp).name() << ",dimSize=" << dimSize);
        return;
    }
    uint64_t maxRowCount = hmo.GetByteSize() / (sizeof(DataTemp) * dimSize);
    uint64_t printCount = std::min(maxRowCount, maxPrintCount);
    uint64_t printDimSize = std::min(dimSize, maxPrintCount);
    auto buffer =
        hmo.GetBuffer(hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 0, printCount * (sizeof(DataTemp) * dimSize));
    if (buffer.get() == nullptr) {
        OCK_HCPS_LOG_ERROR("GetBuffer(" << hmo << ") return nullptr. name=" << name);
        return;
    }
    auto convertor = DataCvtT();
    const DataTemp *pData = reinterpret_cast<DataTemp *>(buffer->Address());
    std::ostringstream os;
    os << name << ", typename=" << typeid(DataTemp).name() << ", DimSize=" << dimSize << ", maxRowCount=" << maxRowCount
       << "\n";
    for (uint64_t rowId = 0; rowId < printCount; ++rowId) {
        if (rowId > 256ULL && rowId % 256ULL != 0) {
            continue;
        }
        os << "[" << rowId << "]";
        for (uint64_t dim = 0; dim < printDimSize; ++dim, ++pData) {
            if (dim != 0ULL) {
                os << ",";
            }
            os << convertor(*pData);
        }
        if (printDimSize < dimSize) {
            os << "...";
        }
        os << std::endl;
    }
    if (printCount < maxRowCount) {
        os << "...";
    }
    OCK_HCPS_LOG_INFO(os.str());
}

template <typename DataTemp, uint64_t DimSizeTemp>
void PrintFeature(hmm::OckHmmHMObject &hmo, bool forcePrint = false, const std::string &name = "Feature",
    uint64_t maxPrintCount = 10ULL)
{
    PrintAnyFeature<DataTemp, DataConvertorAdapter<DataTemp, uint64_t>>(hmo, DimSizeTemp, forcePrint, name,
        maxPrintCount);
}

inline void PrintNorm(hmm::OckHmmHMObject &hmo, bool forcePrint = false, uint64_t maxPrintCount = 10ULL)
{
    if (forcePrint == false && OckLogger::Instance().GetStartLevel() > OCK_LOG_LEVEL_INFO) {
        return;
    }
    uint64_t maxRowCount = hmo.GetByteSize() / sizeof(OckFloat16);
    uint64_t printCount = std::min(maxRowCount, maxPrintCount);
    auto buffer = hmo.GetBuffer(hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 0, printCount * sizeof(OckFloat16));
    if (buffer.get() == nullptr) {
        OCK_HCPS_LOG_ERROR("GetBuffer(" << hmo << ") return nullptr");
        return;
    }
    const OckFloat16 *pData = reinterpret_cast<OckFloat16 *>(buffer->Address());
    std::ostringstream os;
    os << "name=Norm, maxRowCount=" << maxRowCount << ", ";
    for (uint64_t rowId = 0; rowId < printCount; ++rowId, ++pData) {
        if (rowId > 256ULL && rowId % 256ULL != 0) {
            continue;
        }
        if (rowId != 0) {
            os << ",";
        }
        os << acladapter::OckAscendFp16::Fp16ToFloat(*pData);
    }
    OCK_HCPS_LOG_INFO(os.str());
}

inline void PrintMask(hmm::OckHmmHMObject &hmo, bool forcePrint = false, uint64_t maxPrintCount = 10ULL)
{
    PrintAnyFeature<uint64_t>(hmo, 1ULL, forcePrint, "Mask", maxPrintCount);
}

template <typename DataTemp, uint64_t DimSizeTemp>
void Print(adapter::OckVsaAnnFeature &feature, bool forcePrint = false, uint64_t maxPrintCount = 10ULL)
{
    if (forcePrint == false && OckLogger::Instance().GetStartLevel() > OCK_LOG_LEVEL_INFO) {
        return;
    }
    OCK_HCPS_LOG_DEBUG("maxRowCount:" << feature.maxRowCount << ", validRowCount:" << feature.validateRowCount);
    PrintFeature<DataTemp, DimSizeTemp>(*feature.feature, forcePrint, "Feature", maxPrintCount);
    PrintNorm(*feature.norm, forcePrint, maxPrintCount);
    if (feature.mask.get() != nullptr) {
        PrintMask(*feature.mask, forcePrint, maxPrintCount);
    }
}
inline void Print(npu::OckVsaAnnKeyAttrInfo &attrInfo, uint64_t extAttrByteSize, bool forcePrint = false,
    uint64_t maxPrintCount = 10ULL)
{
    if (forcePrint == false && OckLogger::Instance().GetStartLevel() > OCK_LOG_LEVEL_INFO) {
        return;
    }
    PrintFeature<uint32_t, 1ULL>(*attrInfo.keyAttrTime, forcePrint, "Time", maxPrintCount);
    PrintFeature<uint32_t, 1ULL>(*attrInfo.keyAttrQuotient, forcePrint, "Quotient", maxPrintCount);
    PrintAnyFeature<uint16_t, SpaceRemainderConvertorAdapter>(
        *attrInfo.keyAttrRemainder, 1UL, forcePrint, "Remainder", maxPrintCount);
    PrintFeature<uint16_t, 1ULL>(*attrInfo.keyAttrRemainder, forcePrint, "Remainder", maxPrintCount);
    if (attrInfo.extKeyAttr.get() != nullptr) {
        PrintAnyFeature<uint8_t>(*attrInfo.extKeyAttr, extAttrByteSize, forcePrint, "ExtAttr", maxPrintCount);
    }
}
template <typename DataTemp, uint64_t DimSizeTemp>
void Print(npu::OckVsaAnnRawBlockInfo &attrInfo, uint64_t extAttrByteSize, bool forcePrint = false,
    uint64_t maxPrintCount = 10ULL)
{
    if (forcePrint == false && OckLogger::Instance().GetStartLevel() > OCK_LOG_LEVEL_INFO) {
        return;
    }
    PrintFeature<uint32_t, 1ULL>(*attrInfo.keyAttrTime, forcePrint, "Time", maxPrintCount);
    PrintFeature<uint32_t, 1ULL>(*attrInfo.keyAttrQuotient, forcePrint, "Quotient", maxPrintCount);
    PrintAnyFeature<uint16_t, SpaceRemainderConvertorAdapter>(
        *attrInfo.keyAttrRemainder, 1UL, forcePrint, "Remainder", maxPrintCount);
    if (attrInfo.extKeyAttr.get() != nullptr) {
        PrintAnyFeature<uint8_t>(*attrInfo.extKeyAttr, extAttrByteSize, forcePrint, "ExtAttr", maxPrintCount);
    }
    PrintFeature<DataTemp, DimSizeTemp>(*attrInfo.feature, forcePrint, "Feature", maxPrintCount);
    PrintNorm(*attrInfo.norm, forcePrint, maxPrintCount);
}

// 无效值打印
template <typename DataTemp, typename _AssertT = InvalidDataAssertAdapter<DataTemp>>
void PrintInvalidFeature(hmm::OckHmmHMObject &hmo, uint64_t dimSize, const _AssertT &assertFun, bool forcePrint = false,
                         const std::string &name = "Feature")
{
    if (forcePrint == false) {
        return;
    }
    if (dimSize == 0UL) {
        OCK_HCPS_LOG_DEBUG(name << ", typename=" << typeid(DataTemp).name() << ",dimSize=" << dimSize);
        return;
    }

    uint64_t maxRowCount = hmo.GetByteSize() / (sizeof(DataTemp) * dimSize);
    auto buffer = hmo.GetBuffer(hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 0,
        maxRowCount * (sizeof(DataTemp) * dimSize));
    if (buffer.get() == nullptr) {
        OCK_HCPS_LOG_ERROR("GetBuffer(" << hmo << ") return nullptr. name=" << name);
        return;
    }
    const DataTemp *pData = reinterpret_cast<DataTemp *>(buffer->Address());
    std::ostringstream os;
    os << name << ", typename=" << typeid(DataTemp).name() << ", DimSize=" << dimSize << ", maxRowCount=" << maxRowCount
       << "\n";
    for (uint64_t rowId = 0; rowId < maxRowCount; ++rowId) {
        for (uint64_t dim = 0; dim < dimSize; ++dim, ++pData) {
            if (!assertFun(*pData)) {
                os << "[" << rowId << "][" << dim << "]" << *pData << std::endl;
            }
        }
    }
    if (!os.str().empty()) {
        OCK_HCPS_LOG_ERROR(os.str());
    }
}
}  // namespace printer
}  // namespace neighbor
}  // namespace vsa
}  // namespace ock
#endif