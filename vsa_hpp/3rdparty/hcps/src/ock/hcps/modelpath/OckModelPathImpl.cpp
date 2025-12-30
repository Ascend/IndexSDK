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
#include "ock/log/OckLogger.h"
#include "ock/hcps/modelpath/OckModelPath.h"
namespace ock {
namespace hcps {

class OckModelPathImpl : public OckModelPath {
public:
    void SetPath(const std::string &path) override
    {
        std::lock_guard<std::mutex> lock(modelPathMutex);
        if (!modelPathAlreadySet) {
            modelPath = path;
            modelPathAlreadySet = true;
        } else {
            OCK_HMM_LOG_WARN("modelpath ('" << modelPath << "') has already been set. It cannot be changed.");
        }
    };

    std::string Path(void) const override
    {
        std::lock_guard<std::mutex> lock(modelPathMutex);
        return modelPath;
    };

private:
    std::string modelPath {"modelPath"};
    static bool modelPathAlreadySet;
    mutable std::mutex modelPathMutex{};
};
bool OckModelPathImpl::modelPathAlreadySet = false;

OckModelPath &OckModelPath::Instance(void)
{
    static OckModelPathImpl ockModelPath;
    return ockModelPath;
}

OckModelPath &g_instance_OckModelPath = OckModelPath::Instance();
}  // namespace hcps
}  // namespace ock
