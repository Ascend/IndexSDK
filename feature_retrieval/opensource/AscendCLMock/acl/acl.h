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

#ifndef ACL_H
#define ACL_H

#include "acl_rt.h"
#include "acl_op.h"
#include "acl_mdl.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef UNUSED
#define UNUSED(x) (x) __attribute__((unused))
#endif

aclError aclInit(const char *configPath);
aclError aclFinalize();
void setMaxSize(size_t size);

#ifdef __cplusplus
}
#endif

#endif // ACL_H