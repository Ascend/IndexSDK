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


#ifndef OCK_VSA_ATTR_MAX_MIN_TRAIT_H
#define OCK_VSA_ATTR_MAX_MIN_TRAIT_H
#include "ock/vsa/attr/OckKeyTrait.h"
namespace ock {
namespace vsa {
namespace attr {
template <typename _MultiTraits, size_t NTemp> struct InitFromHelper {
    static void RecurInit(_MultiTraits &ockMultiTraitsObj, const _MultiTraits &other)
    {
        std::get<NTemp - 1ULL>(ockMultiTraitsObj.traits).InitFrom(std::get<NTemp - 1ULL>(other.traits));
        InitFromHelper<_MultiTraits, NTemp - 1ULL>::recurInit(ockMultiTraitsObj, other);
    }
};

template <typename _MultiTraits> struct InitFromHelper<_MultiTraits, 1ULL> {
    static void RecurInit(_MultiTraits &ockMultiTraitsObj, const _MultiTraits &other)
    {
        std::get<0ULL>(ockMultiTraitsObj.traits).InitFrom(std::get<0ULL>(other.traits));
    }
};

template <typename _KeyType> struct OckMaxMinTrait : OckKeyTrait {
    using KeyType = _KeyType;
    using KeyTypeTuple = std::tuple<KeyType>;

    static OckMaxMinTrait<_KeyType> InitFrom(const OckMaxMinTrait<_KeyType> &other)
    {
        return OckMaxMinTrait<_KeyType>();
    }
    void Add(const _KeyType &value);
    void Add(const OckMaxMinTrait &other);
    bool In(const KeyType &value) const;
    bool Intersect(const OckMaxMinTrait &other) const;
    double CoverageRate(const OckMaxMinTrait &other) const;
    friend inline std::ostream &operator << (std::ostream &os, const OckMaxMinTrait<_KeyType> &trait)
    {
        return os << "{'min':" << trait.minValue << ",'max':" << trait.maxValue << "}";
    }

    _KeyType maxValue{ utils::Limits<_KeyType>::Min() };
    _KeyType minValue{ utils::Limits<_KeyType>::Max() };
};

template <size_t IdxTemp, typename _Tuple, size_t NmTemp, typename... _Tp>
struct OckMakeTraitKeyTypeTuple : public OckMakeTraitKeyTypeTuple<IdxTemp - 1UL, _Tuple, NmTemp,
        typename std::tuple_element<IdxTemp - 1UL, _Tuple>::type::KeyType, _Tp...> {};
template <typename _Tuple, size_t NmTemp, typename... _Tp>
struct OckMakeTraitKeyTypeTuple<1UL, _Tuple, NmTemp, _Tp...> {
    using KeyTypeTuple = std::tuple<typename std::tuple_element<0UL, _Tuple>::type::KeyType, _Tp...>;
};

template <typename... _Trait> struct OckMultiKeyTrait : public std::tuple<_Trait...> {
    using TraitTuple = std::tuple<_Trait...>;
    using KeyTypeTuple = typename OckMakeTraitKeyTypeTuple<std::tuple_size<TraitTuple>::value, TraitTuple,
            std::tuple_size<TraitTuple>::value>::KeyTypeTuple;

    static OckMultiKeyTrait<_Trait...> InitFrom(const OckMultiKeyTrait<_Trait...> &other)
    {
        constexpr size_t otherTraitSize = std::tuple_size<decltype(other.traits)>::value;
        OckMultiKeyTrait<_Trait...> ockMultiKeyTraitObj;
        InitFromHelper<OckMultiKeyTrait<_Trait...>, otherTraitSize>::recurInit(ockMultiKeyTraitObj, other);
        return ockMultiKeyTraitObj;
    }

    void Add(const KeyTypeTuple &value);
    void Add(const OckMultiKeyTrait &other);
    bool In(const KeyTypeTuple &value) const;
    bool Intersect(const OckMultiKeyTrait &other) const;
    void Add(uintptr_t dataAddr);
    bool In(uintptr_t dataAddr) const;
    double CoverageRate(const OckMultiKeyTrait &other) const;

    friend inline std::ostream &operator << (std::ostream &os, const OckMultiKeyTrait<_Trait...> &keyTrait)
    {
        return utils::traits::TuplePrinter<TraitTuple>::Print(os, keyTrait.traits);
    }

    TraitTuple traits;
};
} // namespace attr
} // namespace vsa
} // namespace ock
#include "impl/OckMaxMinTraitImpl.h"
#endif // OCK_VSA_ATTR_MAX_MIN_TRAIT_H
