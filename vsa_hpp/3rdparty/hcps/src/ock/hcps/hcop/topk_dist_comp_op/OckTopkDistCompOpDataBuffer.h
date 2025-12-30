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


#ifndef OCK_HCPS_HCOP_TOPK_DIST_COMP_OP_DATA_BUFFER
#define OCK_HCPS_HCOP_TOPK_DIST_COMP_OP_DATA_BUFFER
#include <memory>
#include <set>
#include "ock/hcps/nop/topk_flat_op/OckTopkFlatOpDataBuffer.h"
#include "ock/hcps/nop/dist_int8_cos_max_op/OckDistInt8CosMaxOpDataBuffer.h"
#include "ock/hcps/hcop/topk_dist_comp_op/OckTopkDistCompMeta.h"
namespace ock {
namespace hcps {
namespace hcop {
struct OckTopkDistCompHostData {
    std::vector<int8_t> queries;                  // 连续的，dist专用
    std::vector<OckFloat16> queriesNorm;           // 连续的，dist专用
    std::vector<uint8_t> mask;                    // 连续的，dist专用
    std::vector<std::vector<int8_t>> base;        // 离散的，dist专用
    std::vector<std::vector<OckFloat16>> baseNorm; // 离散的，dist专用
    std::vector<int64_t> topkAttrs;               // 全局的，topk专用
    std::vector<uint32_t> sizes;                  // 全局的，topk与dist公用(dist的actualSize)
    std::vector<uint16_t> outFlags;               // 全局输，topk与dist公用：需初始化为0
    std::vector<OckFloat16> outDists;              // 全局的，topk与dist公用
    std::vector<int64_t> outLabels;               // 全局的，topk与dist公用
};

class OckTopkDistCompOpDataBuffer : public nop::OckOpDataBuffer {
public:
    virtual ~OckTopkDistCompOpDataBuffer() noexcept = default;
    virtual std::shared_ptr<nop::OckTopkFlatOpDataBuffer> &GetTopkBuffer() = 0;
    virtual std::vector<std::shared_ptr<nop::OckDistInt8CosMaxOpDataBuffer>> &GetDistBuffers() = 0;
    virtual OckHcpsErrorCode AllocBuffersFromHmoGroup(const std::shared_ptr<OckTopkDistCompOpHmoGroup> &hmoGroup,
        const std::shared_ptr<hmm::OckHmmHeteroMemoryMgrBase> &devMgr) = 0;
    virtual void FillBuffers(OckTopkDistCompHostData &hostData) = 0;
    virtual void SetHyperParameters(std::shared_ptr<OckTopkDistCompOpHmoGroup> hmoGroup) = 0;
    virtual hmm::OckHmmErrorCode PrepareHyperParameters(std::shared_ptr<OckTopkDistCompOpHmoGroup> hmoGroup) = 0;
    static std::shared_ptr<OckTopkDistCompOpDataBuffer> Create(const OckTopkDistCompOpMeta &opSpec,
        const OckTopkDistCompBufferMeta &bufferSpec);
};
} // namespace hcop
} // namespace hcps
} // namespace ock
#endif // OCK_HCPS_HCOP_TOPK_DIST_COMP_OP_DATA_BUFFER