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


#ifndef ASCEND_HMO_IMPL_H
#define ASCEND_HMO_IMPL_H

#include <ock/hmm/mgr/OckHmmHeteroMemoryMgr.h>
#include <ock/hmm/mgr/OckHmmHMObject.h>
#include <ock/hmm/mgr/OckHmmAsyncResult.h>
#include "AscendHMO.h"

namespace ascend {

class AscendHMOImpl : public AscendHMO {
public:
    explicit AscendHMOImpl(std::shared_ptr<ock::hmm::OckHmmHeteroMemoryMgrBase> hmm,
        std::shared_ptr<ock::hmm::OckHmmHMObject> object);

    ~AscendHMOImpl() override;

    void Clear() override;

    uintptr_t GetAddress() const override;

    bool Empty() const override;

    bool IsHostHMO() const override;

    APP_ERROR CopyTo(std::shared_ptr<AscendHMO> dstHmo, size_t dstOffset, size_t offset, size_t len) const override;

    /*
     * 同步获取HMO在device侧的缓冲区地址
     */
    APP_ERROR ValidateBuffer() override;

    /*
     * 异步获取HMO在device侧的缓冲区地址
     */
    APP_ERROR ValidateBufferAsync() override;

    /*
     * 释放HMO在device侧的缓冲区地址
     */
    APP_ERROR InvalidateBuffer() override;

    /*
     * 将HMO在device侧缓冲区内容刷新到真实内存区
     */
    APP_ERROR FlushData() override;

    std::shared_ptr<ock::hmm::OckHmmHMObject> GetObject() const;

    AscendHMOImpl(const AscendHMOImpl&) = delete;
    AscendHMOImpl& operator=(const AscendHMOImpl&) = delete;

private:
    std::shared_ptr<ock::hmm::OckHmmHeteroMemoryMgrBase> hmm = nullptr;
    std::shared_ptr<ock::hmm::OckHmmHMObject> object = nullptr;
    std::shared_ptr<ock::hmm::OckHmmHMOBuffer> buffer = nullptr;
    std::shared_ptr<ock::hmm::OckHmmAsyncResult<ock::hmm::OckHmmHMOBuffer>> asynBuffer = nullptr;
};

}

#endif