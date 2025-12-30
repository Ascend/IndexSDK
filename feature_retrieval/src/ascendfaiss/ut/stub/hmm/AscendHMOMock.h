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


#ifndef ASCEND_HMO_MOCK_H
#define ASCEND_HMO_MOCK_H

#include <gmock/gmock.h>
#include "hmm/AscendHMO.h"

namespace ascend {

class AscendHMOMock : public AscendHMO {
public:
    MOCK_METHOD(void, Clear, ());
    MOCK_METHOD(uintptr_t, GetAddress, (), (const));
    MOCK_METHOD(bool, Empty, (), (const));
    MOCK_METHOD(bool, IsHostHMO, (), (const));
    MOCK_METHOD(APP_ERROR, CopyTo, (std::shared_ptr<AscendHMO>, size_t, size_t, size_t), (const));
    MOCK_METHOD(APP_ERROR, ValidateBuffer, ());
    MOCK_METHOD(APP_ERROR, ValidateBufferAsync, ());
    MOCK_METHOD(APP_ERROR, InvalidateBuffer, ());
    MOCK_METHOD(APP_ERROR, FlushData, ());
};

}

#endif