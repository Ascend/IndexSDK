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

#include <memory>
#include "acl/acl.h"
#include "ock/acladapter/utils/OckAscendFp16.h"
namespace ock {
namespace acladapter {
float OckAscendFp16::Fp16ToFloat(OckFloat16 fp16)
{
    return aclFloat16ToFloat(fp16);
}

OckFloat16 OckAscendFp16::FloatToFp16(float fp)
{
    return aclFloatToFloat16(fp);
}
} // namespace acladapter
} // namespace ock