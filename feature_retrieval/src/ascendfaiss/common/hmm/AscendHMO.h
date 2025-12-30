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


#ifndef ASCEND_HMO_H
#define ASCEND_HMO_H

#include <memory>
#include "ErrorCode.h"

namespace ascend {

class AscendHMO {
public:
    virtual ~AscendHMO() = default;

    virtual void Clear() = 0;

    virtual uintptr_t GetAddress() const = 0;

    virtual bool Empty() const = 0;

    virtual bool IsHostHMO() const = 0;

    virtual APP_ERROR CopyTo(std::shared_ptr<AscendHMO> dstHmo, size_t dstOffset, size_t offset, size_t len) const = 0;

    /*
     * 同步获取HMO在device侧的缓冲区地址
     */
    virtual APP_ERROR ValidateBuffer() = 0;

    /*
     * 异步获取HMO在device侧的缓冲区地址
     */
    virtual APP_ERROR ValidateBufferAsync() = 0;

    /*
     * 释放HMO在device侧的缓冲区地址
     */
    virtual APP_ERROR InvalidateBuffer() = 0;

    /*
     * 将HMO在device侧缓冲区内容刷新到真实内存区
     */
    virtual APP_ERROR FlushData() = 0;
};

}

#endif