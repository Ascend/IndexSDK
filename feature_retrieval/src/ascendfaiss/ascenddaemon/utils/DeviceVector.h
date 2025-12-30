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


#ifndef ASCEND_DEVICEVECTOR_INCLUDED
#define ASCEND_DEVICEVECTOR_INCLUDED

#include <vector>
#include <memory>
#include "ascenddaemon/utils/MemorySpace.h"
#include "DevVecMemStrategyIntf.h"
#include "PureDevMemStrategy.h"
#include "HeteroMemStrategy.h"

namespace ascend {

template<typename T, typename P = ExpandPolicy>
class DeviceVector {
public:
    explicit DeviceVector(MemorySpace space = MemorySpace::DEVICE)
    {
        memStrategy = std::unique_ptr<PureDevMemStrategy<T, P>>(new PureDevMemStrategy<T, P>(space));
    }

    explicit DeviceVector(std::shared_ptr<HmmIntf> hmm)
    {
        memStrategy = std::unique_ptr<HeteroMemStrategy<T, P>>(new HeteroMemStrategy<T, P>(hmm));
    }

    ~DeviceVector() {}

    void clear()
    {
        memStrategy->Clear();
    }

    size_t size() const
    {
        return memStrategy->Size();
    }

    size_t capacity() const
    {
        return memStrategy->Capacity();
    }

    T* data() const
    {
        return memStrategy->Data();
    }

    T& operator[](size_t pos)
    {
        return (*memStrategy)[pos];
    }

    const T& operator[](size_t pos) const
    {
        return (*memStrategy)[pos];
    }

    std::vector<T> copyToStlVector() const
    {
        return memStrategy->CopyToStlVector();
    }

    void append(const T* d, size_t n, bool reserveExact = false)
    {
        memStrategy->Append(d, n, reserveExact);
    }

    void resize(size_t newSize, bool reserveExact = false)
    {
        memStrategy->Resize(newSize, reserveExact);
    }

    size_t reclaim(bool exact)
    {
        return memStrategy->Reclaim(exact);
    }

    void reserve(size_t newCapacity)
    {
        memStrategy->Reserve(newCapacity);
    }

    void pushData(bool dataChanged = true)
    {
        memStrategy->PushData(dataChanged);
    }

    std::shared_ptr<AscendHMO> getHmo() const
    {
        return memStrategy->GetHmo();
    }

private:
    std::unique_ptr<DevVecMemStrategyIntf<T>> memStrategy;
};

}  // namespace ascend

#endif  // ASCEND_DEVICEVECTOR_H
