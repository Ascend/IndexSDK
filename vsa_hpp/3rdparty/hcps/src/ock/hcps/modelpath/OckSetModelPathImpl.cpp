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

#include <mutex>
#include "acl/acl.h"
#include "ock/log/OckLogger.h"
#include "ock/hcps/modelpath/OckModelPath.h"
#include "ock/hcps/modelpath/OckSetModelPath.h"
namespace ock {
namespace hcps {

class OckSetModelPathImpl : public OckSetModelPath {
public:
    void NotifyDevice(void)
    {
        std::lock_guard<std::mutex> lock(setModelPathMutex);
        if (!alreadyNotified) {
            aclopSetModelDir(OckModelPath::Instance().Path().c_str());
            alreadyNotified = true;
        } else {
            OCK_HMM_LOG_WARN("acl op model dir has already been set and does not need to be set again.");
        }
    };
private:
    static bool alreadyNotified;
    mutable std::mutex setModelPathMutex{};
};
bool OckSetModelPathImpl::alreadyNotified = false;

OckSetModelPath &OckSetModelPath::Instance(void)
{
    static OckSetModelPathImpl ockSetModelPath;
    return ockSetModelPath;
}

OckSetModelPath &g_instance_OckSetModelPath = OckSetModelPath::Instance();

}  // namespace hcps
}  // namespace ock