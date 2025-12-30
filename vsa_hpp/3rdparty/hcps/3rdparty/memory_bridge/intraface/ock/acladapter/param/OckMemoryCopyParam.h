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


#ifndef OCK_MEMORY_BRIDGE_OCK_ADAPTER_PARAM_OCK_MEMORY_COPY_PARAM_H
#define OCK_MEMORY_BRIDGE_OCK_ADAPTER_PARAM_OCK_MEMORY_COPY_PARAM_H
#include <ostream>
#include "ock/acladapter/data/OckMemoryCopyKind.h"
#include "ock/acladapter/executor/OckAsyncTaskBase.h"
namespace ock {
namespace acladapter {

class OckMemoryCopyParam : public OckAsyncTaskParamBase {
public:
    virtual ~OckMemoryCopyParam() noexcept = default;
    explicit OckMemoryCopyParam(void *destination, size_t destinationMax, const void *source, size_t copySize,
        OckMemoryCopyKind copyKind);

    OckMemoryCopyParam(const OckMemoryCopyParam &) = delete;
    OckMemoryCopyParam &operator=(const OckMemoryCopyParam &) = delete;

    void *DestAddr(void);
    const void *DestAddr(void) const;
    size_t DestMax(void) const;
    const void *SrcAddr(void) const;
    size_t SrcCount(void) const;
    OckMemoryCopyKind Kind(void) const;

private:
    void *dst;
    size_t destMax;
    const void *src;
    size_t count;
    OckMemoryCopyKind kind;
};

std::ostream &operator<<(std::ostream &os, const OckMemoryCopyParam &param);
}  // namespace acladapter
}  // namespace ock
#endif