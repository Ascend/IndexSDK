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


#ifndef OCK_HCPS_HCOP_TOPK_FLAT_META
#define OCK_HCPS_HCOP_TOPK_FLAT_META
#include <string>
#include "ock/hcps/nop/OckOpConst.h"
namespace ock {
namespace hcps {
namespace nop {
struct OckTopkFlatOpMeta {
    std::string outLabelsDataType{ "int64" }; // 仅支持"int64" 和 "uint16"
    int64_t batch{ 1 };                       // 仅支持 { 64, 48, 36, 32, 24, 18, 16, 12, 8, 6, 4, 2, 1 }
    int64_t codeBlockSize{ FLAT_DEFAULT_DIST_COMPUTE_BATCH };
};

struct OckTopkFlatBufferMeta {
    int64_t k{ 1 };
    int64_t blockNum{ 1 };
};
} // namespace nop
} // namespace hcps
} // namespace ock
#endif // OCK_HCPS_HCOP_TOPK_FLAT_META