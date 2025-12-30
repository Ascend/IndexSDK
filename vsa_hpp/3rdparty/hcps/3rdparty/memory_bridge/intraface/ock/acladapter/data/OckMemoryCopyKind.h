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


#ifndef OCK_MEMORY_BRIDGE_OCK_ADAPTER_PARAM_OCK_MEMORY_COPY_KIND_H
#define OCK_MEMORY_BRIDGE_OCK_ADAPTER_PARAM_OCK_MEMORY_COPY_KIND_H
#include <ostream>
#include "ock/hmm/mgr/OckHmmHeteroMemoryLocation.h"
namespace ock {
namespace acladapter {

// 这里定义为bit枚举，方便TopoDetect做侦测类型组合赋值
enum class OckMemoryCopyKind : uint32_t {
    HOST_TO_HOST = 1U,
    HOST_TO_DEVICE = 2U,
    DEVICE_TO_HOST = 4U,
    DEVICE_TO_DEVICE = 8U
};

OckMemoryCopyKind CalcMemoryCopyKind(hmm::OckHmmHeteroMemoryLocation src, hmm::OckHmmHeteroMemoryLocation dst);
std::ostream &operator<<(std::ostream &os, OckMemoryCopyKind kind);
/*
@brief 这里解析为uint32_t， 从而支持bit枚举组合
@return ret.first 代表解析是否成功
        ret.second 代表解析后的结果
*/
std::pair<bool, uint32_t> ParseMemoryCopyKind(const std::string &buff);

}  // namespace acladapter
}  // namespace ock
#endif