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


#ifndef OCK_VSA_ATTR_MAX_MIN_TRAIT_IMPL_H
#define OCK_VSA_ATTR_MAX_MIN_TRAIT_IMPL_H
#include <type_traits>
#include <algorithm>
#include "ock/utils/OckTypeTraits.h"
#include "ock/utils/OckSafeUtils.h"
#include "ock/utils/OckLimits.h"
namespace ock {
namespace vsa {
namespace attr {
namespace impl {
template <typename _TraitTuple, typename _KeyTypeTuple, size_t NTemp>
struct OckMultiKeyTupleAdapter {
    static void Add(_TraitTuple &traits, const _KeyTypeTuple &value)
    {
        std::get<NTemp - 1ULL>(traits).Add(std::get<NTemp - 1ULL>(value));
        OckMultiKeyTupleAdapter<_TraitTuple, _KeyTypeTuple, NTemp - 1ULL>::Add(traits, value);
    }
    static void Add(_TraitTuple &traits, const _TraitTuple &other)
    {
        std::get<NTemp - 1ULL>(traits).Add(std::get<NTemp - 1ULL>(other));
        OckMultiKeyTupleAdapter<_TraitTuple, _KeyTypeTuple, NTemp - 1ULL>::Add(traits, other);
    }
    static bool In(const _TraitTuple &traits, const _KeyTypeTuple &value)
    {
        return OckMultiKeyTupleAdapter<_TraitTuple, _KeyTypeTuple, NTemp - 1ULL>::In(traits, value) &&
               std::get<NTemp - 1ULL>(traits).In(std::get<NTemp - 1ULL>(value));
    }
    static void Add(_TraitTuple &traits, uintptr_t dataAddr)
    {
        std::get<NTemp - 1ULL>(traits).Add(utils::traits::TupleParse<NTemp - 1ULL, _KeyTypeTuple>::Parse(dataAddr));
        OckMultiKeyTupleAdapter<_TraitTuple, _KeyTypeTuple, NTemp - 1ULL>::Add(traits, dataAddr);
    }
    static bool In(const _TraitTuple &traits, uintptr_t dataAddr)
    {
        return OckMultiKeyTupleAdapter<_TraitTuple, _KeyTypeTuple, NTemp - 1ULL>::In(traits, dataAddr) &&
            std::get<NTemp - 1ULL>(traits).In(utils::traits::TupleParse<NTemp - 1ULL, _KeyTypeTuple>::Parse(dataAddr));
    }
    static bool Intersect(const _TraitTuple &traits, const _TraitTuple &other)
    {
        return OckMultiKeyTupleAdapter<_TraitTuple, _KeyTypeTuple, NTemp - 1ULL>::Intersect(traits, other) &&
               std::get<NTemp - 1ULL>(traits).Intersect(std::get<NTemp - 1ULL>(other));
    }
    static double CoverageRate(const _TraitTuple &traits, const _TraitTuple &other)
    {
        return OckMultiKeyTupleAdapter<_TraitTuple, _KeyTypeTuple, NTemp - 1ULL>::CoverageRate(traits, other) *
               std::get<NTemp - 1ULL>(traits).CoverageRate(std::get<NTemp - 1ULL>(other));
    }
};
template <typename _TraitTuple, typename _KeyTypeTuple>
struct OckMultiKeyTupleAdapter<_TraitTuple, _KeyTypeTuple, 1ULL> {
    static void Add(_TraitTuple &traits, const _KeyTypeTuple &value)
    {
        std::get<0UL>(traits).Add(std::get<0UL>(value));
    }
    static void Add(_TraitTuple &traits, const _TraitTuple &other)
    {
        std::get<0UL>(traits).Add(std::get<0UL>(other));
    }
    static bool In(const _TraitTuple &traits, const _KeyTypeTuple &value)
    {
        return std::get<0UL>(traits).In(std::get<0UL>(value));
    }
    static void Add(_TraitTuple &traits, uintptr_t dataAddr)
    {
        std::get<0UL>(traits).Add(utils::traits::TupleParse<0UL, _KeyTypeTuple>::Parse(dataAddr));
    }
    static bool In(const _TraitTuple &traits, uintptr_t dataAddr)
    {
        return std::get<0UL>(traits).In(utils::traits::TupleParse<0UL, _KeyTypeTuple>::Parse(dataAddr));
    }
    static bool Intersect(const _TraitTuple &traits, const _TraitTuple &other)
    {
        return std::get<0ULL>(traits).Intersect(std::get<0ULL>(other));
    }
    static double CoverageRate(const _TraitTuple &traits, const _TraitTuple &other)
    {
        return std::get<0ULL>(traits).CoverageRate(std::get<0ULL>(other));
    }
};

}  // namespace impl
template <typename _KeyType>
inline void OckMaxMinTrait<_KeyType>::Add(const _KeyType &value)
{
    maxValue = std::max(maxValue, value);
    minValue = std::min(minValue, value);
}
template <typename _KeyType>
inline void OckMaxMinTrait<_KeyType>::Add(const OckMaxMinTrait<_KeyType> &other)
{
    maxValue = std::max(maxValue, other.maxValue);
    minValue = std::min(minValue, other.minValue);
}
template <typename _KeyType>
inline bool OckMaxMinTrait<_KeyType>::In(const _KeyType &value) const
{
    return value <= maxValue && value >= minValue;
}
template <typename _KeyType>
inline bool OckMaxMinTrait<_KeyType>::Intersect(const OckMaxMinTrait<_KeyType> &other) const
{
    return std::min(maxValue, other.maxValue) >= std::max(minValue, other.minValue);
}
template <typename _KeyType>
inline double OckMaxMinTrait<_KeyType>::CoverageRate(const OckMaxMinTrait<_KeyType> &other) const
{
    _KeyType tmpMax = std::min(maxValue, other.maxValue);
    _KeyType tmpMin = std::max(minValue, other.minValue);
    return utils::SafeDiv(double(tmpMax - tmpMin), (double)(maxValue - minValue));
}

template <typename... _Trait>
void OckMultiKeyTrait<_Trait...>::Add(const KeyTypeTuple &value)
{
    impl::OckMultiKeyTupleAdapter<TraitTuple, KeyTypeTuple, std::tuple_size<TraitTuple>::value>::Add(traits, value);
}
template <typename... _Trait>
void OckMultiKeyTrait<_Trait...>::Add(const OckMultiKeyTrait<_Trait...> &other)
{
    impl::OckMultiKeyTupleAdapter<TraitTuple, KeyTypeTuple, std::tuple_size<TraitTuple>::value>::Add(traits, other);
}
template <typename... _Trait>
bool OckMultiKeyTrait<_Trait...>::In(const KeyTypeTuple &value) const
{
    return impl::OckMultiKeyTupleAdapter<TraitTuple, KeyTypeTuple, std::tuple_size<TraitTuple>::value>::In(
        traits, value);
}
template <typename... _Trait>
void OckMultiKeyTrait<_Trait...>::Add(uintptr_t dataAddr)
{
    impl::OckMultiKeyTupleAdapter<TraitTuple, KeyTypeTuple, std::tuple_size<TraitTuple>::value>::Add(traits, dataAddr);
}
template <typename... _Trait>
bool OckMultiKeyTrait<_Trait...>::In(uintptr_t dataAddr) const
{
    return impl::OckMultiKeyTupleAdapter<TraitTuple, KeyTypeTuple, std::tuple_size<TraitTuple>::value>::In(
        traits, dataAddr);
}
template <typename... _Trait>
bool OckMultiKeyTrait<_Trait...>::Intersect(const OckMultiKeyTrait &other) const
{
    return impl::OckMultiKeyTupleAdapter<TraitTuple, KeyTypeTuple, std::tuple_size<TraitTuple>::value>::Intersect(
        traits, other.traits);
}
template <typename... _Trait>
double OckMultiKeyTrait<_Trait...>::CoverageRate(const OckMultiKeyTrait &other) const
{
    return impl::OckMultiKeyTupleAdapter<TraitTuple, KeyTypeTuple, std::tuple_size<TraitTuple>::value>::CoverageRate(
        traits, other.traits);
}

}  // namespace attr
}  // namespace vsa
}  // namespace ock
#endif // OCK_VSA_ATTR_MAX_MIN_TRAIT_IMPL_H
