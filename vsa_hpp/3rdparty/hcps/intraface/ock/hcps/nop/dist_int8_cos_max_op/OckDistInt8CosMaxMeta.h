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


#ifndef OCK_HCPS_HCOP_DISTINT8COSMAX_META
#define OCK_HCPS_HCOP_DISTINT8COSMAX_META
#include "ock/hcps/nop/OckOpConst.h"
namespace ock {
namespace hcps {
namespace nop {
struct OckDistInt8CosMaxOpMeta {
    int64_t batch{ 16 };
    int64_t dims{ 256 };
    int64_t codeBlockSize{ FLAT_DEFAULT_DIST_COMPUTE_BATCH };
};

struct OckDistInt8CosMaxBufferMeta {
    int64_t ntotal{ FLAT_DEFAULT_DIST_COMPUTE_BATCH };
};
} // namespace nop
} // namespace hcps
} // namespace ock
#endif // OCK_HCPS_HCOP_DISTINT8COSMAX_META