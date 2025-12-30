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


#ifndef OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_ANN_INDEX_FACTORY_H
#define OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_ANN_INDEX_FACTORY_H
#include <securec.h>
#include "ock/vsa/neighbor/base/OckVsaAnnFactory.h"
#include "ock/vsa/neighbor/hpp/OckVsaAnnHPPIndex.h"
#include "ock/vsa/neighbor/hpp/impl/OckVsaAnnHPPCapacity.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
template <typename Data, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait>
class OckVsaAnnHPPIndexFactoryImpl : public OckVsaAnnIndexFactory<Data, DimSizeT, NormTypeByteSizeT, KeyTrait> {
public:
    using BaseT = OckVsaAnnIndexFactory<Data, DimSizeT, NormTypeByteSizeT, KeyTrait>;
    using DataT = Data;
    using KeyTraitT = KeyTrait;
    using KeyTypeTupleT = typename KeyTrait::KeyTypeTuple;
    using RegisterT = OckVsaAnnIndexFactoryRegister<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>;
    using IndexT = OckVsaAnnIndexBase<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>;
    using IndexImplT = OckVsaAnnHPPIndex<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>;
    using CapacityT = OckVsaAnnHPPCapacity<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>;
    virtual ~OckVsaAnnHPPIndexFactoryImpl() noexcept = default;

    std::shared_ptr<IndexT> Create(std::shared_ptr<OckVsaAnnCreateParam> param, const KeyTrait &dftTrait,
        OckVsaErrorCode &errorCode) const override
    {
        // 入参校验
        if (param == nullptr) {
            OCK_VSA_HPP_LOG_ERROR("create indexFactory failed, the param is nullptr");
            return std::shared_ptr<IndexT>();
        }
        if (errorCode != hmm::HMM_SUCCESS) {
            return std::shared_ptr<IndexT>();
        }
        errorCode = OckVsaAnnCreateParam::CheckValid(*param);
        if (errorCode != hmm::HMM_SUCCESS) {
            OCK_VSA_HPP_LOG_ERROR("create indexFactory parameter verification failed.");
            return std::shared_ptr<IndexT>();
        }

        uint32_t maxGroupCountInNpu = std::min(MAX_FEATURE_GROUP_IN_DEVICE, param->MaxGroupCount());
        uint32_t maxGroupCountInHost =
            param->MaxGroupCount() - std::min(maxGroupCountInNpu, MIN_FEATURE_GROUP_IN_DEVICE);
        auto hostParam = param->Copy(maxGroupCountInHost * param->GroupRowCount());
        hmm::OckHmmMemorySpecification spec;
        spec.hostSpec.maxDataCapacity = CapacityT::HostSpace(*hostParam);
        spec.hostSpec.maxSwapCapacity = CapacityT::HostSwapSpace(*hostParam);

        auto devParam = param->Copy(maxGroupCountInNpu * param->GroupRowCount());
        spec.devSpec.maxSwapCapacity = CapacityT::DeviceSwapSpace(*devParam);

        OCK_VSA_HPP_LOG_INFO("The result of the calculation: Space:" << spec << " HostParam:" << *hostParam <<
            " NpuParam:" << *devParam);

        // 先不做容量计算，POC直接申请大空间
        spec.devSpec.maxDataCapacity = 40ULL * 1024ULL * 1024ULL * 1024ULL;

        OCK_VSA_HPP_LOG_INFO("Space:" << spec << " HostParam:" << *hostParam << " NpuParam:" << *devParam);

        auto handler = hcps::handler::OckHeteroHandler::CreateSingleDeviceHandler(param->DeviceId(), param->CpuSet(),
            spec, errorCode);

        if (errorCode != hmm::HMM_SUCCESS) {
            return std::shared_ptr<IndexT>();
        }
        auto hppIndex = new IndexImplT(handler, dftTrait, param, devParam, hostParam);
        if (hppIndex == nullptr) {
            return std::shared_ptr<IndexT>();
        }
        errorCode = hppIndex->Init();
        if (errorCode != hmm::HMM_SUCCESS) {
            return std::shared_ptr<IndexT>();
        }
        return std::shared_ptr<IndexT>(hppIndex);
    }
};
static OckVsaAnnIndexFactoryAutoReg<OckVsaAnnHPPIndexFactoryImpl<int8_t, 256ULL, 2ULL, attr::OckTimeSpaceAttrTrait>>
    s_AutoRegHppTsIndexSearchFactory("HPPTS");
} // namespace hpp
} // namespace neighbor
} // namespace vsa
} // namespace ock
#endif