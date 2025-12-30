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


#ifndef ASCEND_LOCK_INCLUDED
#define ASCEND_LOCK_INCLUDED

#include <mutex>

namespace ascend {
enum LockTypes {
    READ_LOCK = 0,
    WRITE_LOCK = 1
};
class AscendRWLock {
public:
    AscendRWLock(std::mutex *readMtx, size_t *readMtxCnt, std::mutex *writeMtx);
    AscendRWLock(std::mutex *writeMtx);
    ~AscendRWLock();

    AscendRWLock(const AscendRWLock&) = delete;
    AscendRWLock& operator=(const AscendRWLock&) = delete;

private:
    std::mutex *readMtx = nullptr;
    size_t *readMtxCnt = nullptr;
    std::mutex *writeMtx = nullptr;
    size_t lockType;
};
} // ascend

#endif