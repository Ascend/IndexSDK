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

#ifndef OCK_ACL_ADAPTER_OCK_ASCEND_FP16_H
#define OCK_ACL_ADAPTER_OCK_ASCEND_FP16_H

#include <cstdint>

namespace ock {
using OckFloat16 = uint16_t;
namespace acladapter {
class OckAscendFp16 {
public:
    static float Fp16ToFloat(OckFloat16 fp16);
    static OckFloat16 FloatToFp16(float fp);
};
} // namespace acladapter
} // namespace ock
#endif // OCK_ACL_ADAPTER_OCK_ASCEND_FP16_H