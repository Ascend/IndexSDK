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

#include "ock/utils/OckSafeUtils.h"
#include "ock/hmm/mgr/OckHmmMemorySpecification.h"
#include "ock/vsa/neighbor/base/OckVsaAnnIndexBase.h"
#include "ock/vsa/attr/OckTimeSpaceAttrTrait.h"
#include "ock/vsa/neighbor/npu/OckVsaAnnNpuIndex.h"
#include "ock/vsa/neighbor/base/OckVsaAnnFactory.h"
#include "ock/vsa/neighbor/npu/impl/OckVsaAnnIndexCapacity.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace npu {
template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
class OckVsaAnnNpuIndexFactoryImpl
    : public OckVsaAnnIndexFactory<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp> {
public:
    using BaseT = OckVsaAnnIndexFactory<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>;
    using DataT = DataTemp;
    using KeyTraitT = KeyTraitTemp;
    using KeyTypeTupleT = typename KeyTraitTemp::KeyTypeTuple;
    using RegisterT = OckVsaAnnIndexFactoryRegister<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>;
    using IndexT = OckVsaAnnIndexBase<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>;
    using IndexImplT = OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>;
    virtual ~OckVsaAnnNpuIndexFactoryImpl() noexcept = default;

    std::shared_ptr<IndexT> Create(std::shared_ptr<OckVsaAnnCreateParam> param, const KeyTraitTemp &dftTrait,
        OckVsaErrorCode &errorCode) const override
    {
        if (param.get() == nullptr) {
            return std::shared_ptr<IndexT>();
        }
        const uint64_t blockByteSize = param->BlockRowCount() * sizeof(DataTemp) * DimSizeTemp;
        hmm::OckHmmMemorySpecification spec;
        spec.devSpec.maxDataCapacity =
            OckVsaAnnIndexCapacity::DeviceSpace(*param, sizeof(DataTemp) * DimSizeTemp, NormTypeByteSize);
        spec.devSpec.maxSwapCapacity = blockByteSize * 4ULL;  // 4 * 64MB
        spec.hostSpec.maxDataCapacity =
            OckVsaAnnIndexCapacity::HostSpace(*param, sizeof(DataTemp) * DimSizeTemp, NormTypeByteSize);
        spec.hostSpec.maxSwapCapacity = spec.devSpec.maxSwapCapacity;

        auto handler = hcps::handler::OckHeteroHandler::CreateSingleDeviceHandler(
            param->DeviceId(), param->CpuSet(), spec, errorCode);

        if (param->TokenNum() > dftTrait.maxTokenNumber) {
            OCK_HCPS_LOG_ERROR("param->TokenNum() shouldn't be smaller than " << dftTrait.maxTokenNumber);
            errorCode = VSA_ERROR_TOKEN_NUM_OUT;
            return std::shared_ptr<IndexT>();
        }
        if (errorCode != hmm::HMM_SUCCESS) {
            return std::shared_ptr<IndexT>();
        }

        auto npuIndex = new IndexImplT(handler, dftTrait, param);
        if (npuIndex == nullptr) {
            return std::shared_ptr<IndexT>();
        }
        errorCode = npuIndex->Init();
        return std::shared_ptr<IndexT>(npuIndex);
    }
};
}  // namespace npu
}  // namespace neighbor
}  // namespace vsa
}  // namespace ock