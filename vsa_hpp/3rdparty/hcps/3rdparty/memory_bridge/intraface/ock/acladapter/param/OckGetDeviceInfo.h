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


#ifndef MEMORY_BRIDGE_OCKPARAMCHECK_H
#define MEMORY_BRIDGE_OCKPARAMCHECK_H

#include "acl/acl.h"
#include "ock/hmm/mgr/OckHmmHeteroMemoryMgr.h"
#include "ock/log/OckLogger.h"
namespace ock {
namespace acladapter {
class OckGetDeviceInfo {
public:
    static void Init(const hmm::OckHmmDeviceId deviceId);
    static void FreeResources(const hmm::OckHmmDeviceId deviceId);
    static aclError GetDeviceCount(uint32_t *count);
    static aclError GetMemInfo(aclrtMemAttr attr, size_t *free, size_t *total);
};
}
}


#endif // MEMORY_BRIDGE_OCKPARAMCHECK_H
