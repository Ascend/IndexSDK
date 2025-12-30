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


#ifndef HMM_ADAPTOR_H
#define HMM_ADAPTOR_H

#include <ock/hmm/mgr/OckHmmHeteroMemoryMgr.h>
#include "HmmIntf.h"

namespace ascend {

class HmmAdaptor : public HmmIntf {
public:
    HmmAdaptor() = default;

    ~HmmAdaptor() override = default;

    APP_ERROR Init(const HmmMemoryInfo &memoryInfo) override;

    std::pair<APP_ERROR, std::shared_ptr<AscendHMO>> CreateHmo(size_t size) override;

private:
    void SetHmmLog() const;

private:
    std::shared_ptr<ock::hmm::OckHmmHeteroMemoryMgrBase> hmm;
};

}

#endif // HMM_INTF_H