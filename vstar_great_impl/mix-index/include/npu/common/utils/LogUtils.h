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


#ifndef ASCEND_LOG_UTILS_H
#define ASCEND_LOG_UTILS_H

#include "acl/acl.h"

namespace ascendSearchacc {
#define APP_LOG_DEBUG(fmt, ...) aclAppLog(ACL_DEBUG, __FUNCTION__, __FILE__, __LINE__, fmt, ##__VA_ARGS__)

#define APP_LOG_INFO(fmt, ...) aclAppLog(ACL_INFO, __FUNCTION__, __FILE__, __LINE__, fmt, ##__VA_ARGS__)

#define APP_LOG_WARNING(fmt, ...) aclAppLog(ACL_WARNING, __FUNCTION__, __FILE__, __LINE__, fmt, ##__VA_ARGS__)

#define APP_LOG_ERROR(fmt, ...) aclAppLog(ACL_ERROR, __FUNCTION__, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
}  // namespace ascendSearchacc
#endif  // ASCEND_LOG_UTILS_H
