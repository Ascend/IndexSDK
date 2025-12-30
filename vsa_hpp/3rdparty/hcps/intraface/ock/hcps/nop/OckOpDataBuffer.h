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

#ifndef OCK_HCPS_OCK_OP_DATA_BUFFER_H
#define OCK_HCPS_OCK_OP_DATA_BUFFER_H
#include "acl/acl.h"
#include "ock/hmm/mgr/OckHmmHeteroMemoryMgr.h"
#include "ock/hcps/nop/OckDataBuffer.h"
namespace ock {
namespace hcps {
namespace nop {
class OckOpDataBuffer {
public:
    virtual ~OckOpDataBuffer() noexcept = default;
    virtual OckHcpsErrorCode AllocBuffers(const std::shared_ptr<hmm::OckHmmHeteroMemoryMgrBase> &devMgr) = 0;
    virtual OckHcpsErrorCode AllocInputBuffers(const std::shared_ptr<hmm::OckHmmHeteroMemoryMgrBase> &devMgr) = 0;
    virtual OckHcpsErrorCode AllocOutputBuffers(const std::shared_ptr<hmm::OckHmmHeteroMemoryMgrBase> &devMgr) = 0;
    virtual std::vector<std::shared_ptr<OckDataBuffer>> &GetInputParams() = 0;
    virtual std::vector<std::shared_ptr<OckDataBuffer>> &GetOutputParams() = 0;
    virtual std::vector<int64_t> &GetParamsByteSizes() = 0;
    virtual std::vector<std::vector<int64_t>> &GetParamsShapes() = 0;
};

} // namespace nop
} // namespace hcps
} // namespace ock
#endif // OCK_HCPS_OCK_OP_DATA_BUFFER_H