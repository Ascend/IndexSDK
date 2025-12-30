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


#include "AscendHMOImpl.h"
#include <ock/hmm/OckHmmFactory.h>
#include <ock/log/OckHmmLogHandler.h>
#include "common/utils/LogUtils.h"

using namespace ascend;
using namespace ock::hmm;

AscendHMOImpl::AscendHMOImpl(std::shared_ptr<ock::hmm::OckHmmHeteroMemoryMgrBase> hmm,
    std::shared_ptr<ock::hmm::OckHmmHMObject> object)
    : hmm(hmm), object(object)
{}

AscendHMOImpl::~AscendHMOImpl()
{
    Clear();
}

uintptr_t AscendHMOImpl::GetAddress() const
{
    return (buffer == nullptr) ? reinterpret_cast<uintptr_t>(nullptr) : buffer->Address();
}

bool AscendHMOImpl::Empty() const
{
    return object == nullptr;
}

bool AscendHMOImpl::IsHostHMO() const
{
    return Empty() ? false : object->Location() == OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY;
}

void AscendHMOImpl::Clear()
{
    if ((hmm == nullptr) || Empty()) {
        return;
    }

    if (buffer != nullptr) {
        object->ReleaseBuffer(buffer);
        buffer.reset();
    }

    if (asynBuffer != nullptr) {
        asynBuffer->Cancel();
        object->ReleaseBuffer(asynBuffer->WaitResult());
        asynBuffer.reset();
    }

    hmm->Free(object);
    object.reset();
    
    return;
}

std::shared_ptr<ock::hmm::OckHmmHMObject> AscendHMOImpl::GetObject() const
{
    return object;
}

APP_ERROR AscendHMOImpl::CopyTo(std::shared_ptr<AscendHMO> dstHmo, size_t dstOffset, size_t srcOffset, size_t len) const
{
    if (hmm == nullptr) {
        APP_LOG_ERROR("hmm is nullptr!");
        return APP_ERR_INNER_ERROR;
    }

    if (dstHmo == nullptr || dstHmo->Empty() || Empty()) {
        APP_LOG_ERROR("hmo is nullptr or empty");
        return APP_ERR_INVALID_PARAM;
    }

    auto dstHmoImpl = dynamic_cast<AscendHMOImpl *>(dstHmo.get());
    if (dstHmoImpl == nullptr) {
        APP_LOG_ERROR("dstHmo is not AscendHMOImpl");
        return APP_ERR_INVALID_PARAM;
    }

    auto dstObject = dstHmoImpl->GetObject();
    if (dstObject == nullptr) {
        APP_LOG_ERROR("dstObject is nullptr");
        return APP_ERR_INVALID_PARAM;
    }

    auto ret = hmm->CopyHMO(*dstObject, dstOffset, *object, srcOffset, len);
    if (ret != HMM_SUCCESS) {
        APP_LOG_ERROR("CopyHMO failed, ret[%d], dstOffset[%zu], srcOffset[%zu], len[%zu]",
            ret, dstOffset, srcOffset, len);
        return APP_ERR_INNER_ERROR;
    }

    return APP_ERR_OK;
}

APP_ERROR AscendHMOImpl::ValidateBuffer()
{
    if (Empty()) {
        APP_LOG_ERROR("HMO Empty!");
        return APP_ERR_INNER_ERROR;
    }

    if (buffer != nullptr) {
        return APP_ERR_OK;
    }

    if (asynBuffer != nullptr) {
        buffer = asynBuffer->WaitResult();
        if (buffer == nullptr) {
            APP_LOG_ERROR("WaitResult overtime!");
            return APP_ERR_INNER_ERROR;
        }
        asynBuffer.reset();
        return APP_ERR_OK;
    }

    buffer = object->GetBuffer(OckHmmHeteroMemoryLocation::DEVICE_DDR, 0, object->GetByteSize());
    if (buffer == nullptr) {
        APP_LOG_ERROR("GetBuffer failed, size[%zu]", object->GetByteSize());
        return APP_ERR_INNER_ERROR;
    }
    if (buffer->ErrorCode() != HMM_SUCCESS) {
        APP_LOG_ERROR("GetBuffer error, size[%zu], ret[%d]", object->GetByteSize(), buffer->ErrorCode());
        buffer.reset();
        return APP_ERR_INNER_ERROR;
    }

    return APP_ERR_OK;
}

APP_ERROR AscendHMOImpl::ValidateBufferAsync()
{
    if (Empty()) {
        APP_LOG_ERROR("HMO Empty!");
        return APP_ERR_INNER_ERROR;
    }

    if ((buffer != nullptr) || (asynBuffer != nullptr)) {
        return APP_ERR_OK;
    }

    asynBuffer = object->GetBufferAsync(OckHmmHeteroMemoryLocation::DEVICE_DDR, 0, object->GetByteSize());
    if (asynBuffer == nullptr) {
        APP_LOG_ERROR("GetBufferAsync failed, size[%zu]", object->GetByteSize());
        return APP_ERR_INNER_ERROR;
    }

    return APP_ERR_OK;
}

APP_ERROR AscendHMOImpl::InvalidateBuffer()
{
    if (Empty()) {
        APP_LOG_ERROR("HMO Empty!");
        return APP_ERR_INNER_ERROR;
    }

    if (buffer != nullptr) {
        object->ReleaseBuffer(buffer);
        buffer.reset();
        return APP_ERR_OK;
    }

    if (asynBuffer != nullptr) {
        asynBuffer->Cancel();
        object->ReleaseBuffer(asynBuffer->WaitResult());
        asynBuffer.reset();
        return APP_ERR_OK;
    }

    APP_LOG_WARNING("no valid buffer!");
    return APP_ERR_OK;
}

APP_ERROR AscendHMOImpl::FlushData()
{
    if (Empty()) {
        APP_LOG_ERROR("HMO Empty!");
        return APP_ERR_INNER_ERROR;
    }

    if (buffer != nullptr) {
        auto ret = buffer->FlushData();
        if (ret != HMM_SUCCESS) {
            APP_LOG_ERROR("FlushData error, ret[%d]!", ret);
            return APP_ERR_INNER_ERROR;
        }
        return APP_ERR_OK;
    }

    if (asynBuffer != nullptr) {
        buffer = asynBuffer->WaitResult();
        if (buffer == nullptr) {
            APP_LOG_ERROR("WaitResult overtime!");
            return APP_ERR_INNER_ERROR;
        }
        auto ret = buffer->FlushData();
        if (ret != HMM_SUCCESS) {
            APP_LOG_ERROR("FlushData error, ret[%d]!", ret);
            return APP_ERR_INNER_ERROR;
        }
        return APP_ERR_OK;
    }

    APP_LOG_WARNING("no valid buffer!");
    return APP_ERR_OK;
}