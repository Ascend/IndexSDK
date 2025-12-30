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


#ifndef OCK_MEMORY_BRIDGE_OCK_ADAPTER_PARAM_OCK_MEMORY_MALLOC_PARAM_H
#define OCK_MEMORY_BRIDGE_OCK_ADAPTER_PARAM_OCK_MEMORY_MALLOC_PARAM_H
#include <ostream>
#include "ock/acladapter/executor/OckAsyncTaskBase.h"
namespace ock {
namespace acladapter {

class OckMemoryMallocParam : public OckAsyncTaskParamBase {
public:
    virtual ~OckMemoryMallocParam() noexcept = default;
    /*
    @param location 这里当前只支持 HMM_LOCAL_HOST_MEMORY HMM_DEVICE_DDR 两种形式
    */
    explicit OckMemoryMallocParam(void **ptrAddress, size_t mallocSize, hmm::OckHmmHeteroMemoryLocation memLocation);

    OckMemoryMallocParam(const OckMemoryMallocParam &) = delete;
    OckMemoryMallocParam &operator=(const OckMemoryMallocParam &) = delete;

    void **PtrAddr(void);
    const void *Addr(void) const;
    size_t Size(void) const;
    hmm::OckHmmHeteroMemoryLocation Location(void) const;

private:
    void **ptrAddr;
    size_t byteSize;
    hmm::OckHmmHeteroMemoryLocation location;
};

std::ostream &operator<<(std::ostream &os, const OckMemoryMallocParam &param);
}  // namespace acladapter
}  // namespace ock
#endif