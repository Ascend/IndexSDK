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


#include "ascenddaemon/utils/AscendRWLock.h"

namespace ascend {

AscendRWLock::AscendRWLock(std::mutex *readMtx, size_t *readMtxCnt, std::mutex *writeMtx)
    : readMtx(readMtx), readMtxCnt(readMtxCnt), writeMtx(writeMtx), lockType(READ_LOCK)
{
    readMtx->lock();
    if (*readMtxCnt == 0) {
        writeMtx->lock();
    }
    (*readMtxCnt)++;
    readMtx->unlock();
}

AscendRWLock::AscendRWLock(std::mutex *writeMtx) : writeMtx(writeMtx), lockType(WRITE_LOCK)
{
    writeMtx->lock();
}

AscendRWLock::~AscendRWLock()
{
    if (lockType == READ_LOCK) {
        readMtx->lock();
        if (*readMtxCnt == 1) {
            writeMtx->unlock();
        }
        (*readMtxCnt)--;
        readMtx->unlock();
    } else if (lockType == WRITE_LOCK) {
        writeMtx->unlock();
    }
};
} // namespace ascend
