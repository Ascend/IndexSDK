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


#ifndef OCK_VSA_ANN_INDEX_FACTORY_H
#define OCK_VSA_ANN_INDEX_FACTORY_H
#include <cstdint>
#include <memory>
#include <unordered_map>
#include "ock/vsa/OckVsaErrorCode.h"
#include "ock/vsa/neighbor/base/OckVsaAnnQueryCondition.h"
#include "ock/vsa/neighbor/base/OckVsaAnnQueryResult.h"
#include "ock/vsa/neighbor/base/OckVsaAnnIndexBase.h"
#include "ock/vsa/neighbor/base/OckVsaAnnCreateParam.h"

namespace ock {
namespace vsa {
namespace neighbor {
/*
@param _Data 底库数据类型, 例如int8_t
@param DimSizeTemp 底库数据维度, 例如256维
@param KeyTraitTemp 关键属性描述，例如时间+空间属性
@param _ExtKeyByteSizeT 扩展属性的字节数，扩展属性由Index负责存储，用户负责扩展属性数值的计算
*/
template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
class OckVsaAnnIndexFactory {
public:
    using DataT = DataTemp;
    using KeyTraitT = KeyTraitTemp;
    using KeyTypeTupleT = typename KeyTraitTemp::KeyTypeTuple;

    virtual ~OckVsaAnnIndexFactory() noexcept = default;

    virtual std::shared_ptr<OckVsaAnnIndexBase<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>> Create(
        std::shared_ptr<OckVsaAnnCreateParam> param, const KeyTraitTemp &dftTrait,
        OckVsaErrorCode &errorCode) const = 0;
};

template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
class OckVsaAnnIndexFactoryRegister {
public:
    using FactoryT = OckVsaAnnIndexFactory<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>;
    static OckVsaAnnIndexFactoryRegister &Instance(void)
    {
        static OckVsaAnnIndexFactoryRegister ins;
        return ins;
    }
    const FactoryT *GetFactory(const std::string &vsaName = "HPPTS")
    {
        auto iter = factoryMap.find(vsaName);
        if (iter == factoryMap.end()) {
            return nullptr;
        }
        return iter->second.get();
    }
    void RegisterFactory(const std::string &vsaName, std::shared_ptr<FactoryT> factory)
    {
        factoryMap[vsaName] = factory;
    }

private:
    OckVsaAnnIndexFactoryRegister(void) = default;
    std::unordered_map<std::string, std::shared_ptr<FactoryT>> factoryMap{};
};

template <typename _FactoryImplT>
class OckVsaAnnIndexFactoryAutoReg {
public:
    OckVsaAnnIndexFactoryAutoReg(std::string const &vsaName)
    {
        _FactoryImplT::RegisterT::Instance().RegisterFactory(
            vsaName, std::make_shared<_FactoryImplT>());
    }
};
}  // namespace neighbor
}  // namespace vsa
}  // namespace ock
#endif