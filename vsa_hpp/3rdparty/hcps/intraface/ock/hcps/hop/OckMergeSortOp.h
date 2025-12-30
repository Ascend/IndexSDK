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


#ifndef OCK_HCPS_OCK_MERGE_SORT_OP_H
#define OCK_HCPS_OCK_MERGE_SORT_OP_H
#include <cstdint>
#include <memory>
#include <algorithm>
#include <type_traits>
#include "ock/utils/OckContainerInfo.h"
#include "ock/hmm/mgr/OckHmmErrorCode.h"
#include "ock/hcps/stream/OckHeteroOperatorBase.h"
namespace ock {
namespace hcps {
namespace hop {

/*
@brief 约束： 这里要求result的空间是提前分配好的
*/
template <typename _ContainerInfoA, typename _ContainerInfoB, typename _ContainerInfoOut, typename _Compare>
class OckMergeSortOp : public  OckHeteroOperatorGen<acladapter::OckTaskResourceType::HOST_CPU> {
    static_assert(utils::OckContainerInfoTraits<_ContainerInfoA>::value, "_ContainerInfoA must like ContainerInfo");
    static_assert(utils::OckContainerInfoTraits<_ContainerInfoB>::value, "_ContainerInfoA must like ContainerInfo");
    static_assert(utils::OckContainerInfoTraits<_ContainerInfoOut>::value, "_ContainerInfoA must like ContainerInfo");
    using _InputIteratorA = typename _ContainerInfoA::IteratorT;
    using _InputIteratorB = typename _ContainerInfoB::IteratorT;
    using _OutputIterator = typename _ContainerInfoOut::IteratorT;
    static_assert(std::is_same<_InputIteratorA, _InputIteratorB>::value &&
                      std::is_same<typename _ContainerInfoA::IteratorT, typename _ContainerInfoB::IteratorT>::value,
        "The iterator type of 'dataA' must be same with iterator type of 'dataB'");
    static_assert(
        std::is_same<typename std::remove_reference<typename std::iterator_traits<_OutputIterator>::value_type>::type,
            typename std::remove_reference<typename std::iterator_traits<_InputIteratorA>::value_type>::type>::value,
        "The iterator type of 'dataA' must be same with iterator type of 'result'");

public:
    virtual ~OckMergeSortOp() noexcept = default;
    OckMergeSortOp(_ContainerInfoA compareDataA, _ContainerInfoB compareDataB, _ContainerInfoOut outResult,
        const _Compare &compareFunc);
    hmm::OckHmmErrorCode Run(OckHeteroStreamContext &context) override;

    _ContainerInfoA dataA;
    _ContainerInfoB dataB;
    _ContainerInfoOut result;
    _Compare compare;
};

template <typename _ContainerInfoA, typename _ContainerInfoB, typename _ContainerInfoOut, typename _Compare>
std::shared_ptr<OckHeteroOperatorBase> MakeOckMergeSortOp(
    _ContainerInfoA dataA, _ContainerInfoB dataB, _ContainerInfoOut result, const _Compare &compare);

template <typename _ContainerInfoA, typename _ContainerInfoB, typename _ContainerInfoOut, typename _Compare>
std::shared_ptr<OckHeteroOperatorGroup> MakeOckMergeSortOpList(_ContainerInfoA dataA,
    _ContainerInfoB dataB, _ContainerInfoOut result, const _Compare &compare, uint64_t splitThreshold);

}  // namespace hop
}  // namespace hcps
}  // namespace ock
#include "ock/hcps/hop/impl/OckMergeSortOpImpl.h"
#endif