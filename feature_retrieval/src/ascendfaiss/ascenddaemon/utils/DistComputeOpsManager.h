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


#ifndef ASCEND_DIST_COMPUTE_OPS_MANAGER_INCLUDED
#define ASCEND_DIST_COMPUTE_OPS_MANAGER_INCLUDED

#include <map>
#include <vector>
#include <memory>
#include "ascenddaemon/utils/AscendOperator.h"
#include "ascenddaemon/utils/AscendTensor.h"
#include "common/ErrorCode.h"
#include "common/utils/CommonUtils.h"
#include "common/utils/LogUtils.h"
#include "common/utils/SocUtils.h"

namespace ascend {
enum class IndexTypeIdx {
    ITI_FLAT_IP = 0,
    ITI_FLAT_L2,
    ITI_INT8_COS,
    ITI_INT8_COS_FILTER,
    ITI_INT8_L2,
    ITI_INT8_APPROX_L2,
    ITI_SQ_CID_FILTER,
    ITI_SQ_DIST_IP,
    ITI_SQ_DIST_DIM64_IP,
    ITI_SQ_DIST_MASK_IP,
    ITI_SQ_DIST_MASK_DIM64_IP,
    ITI_SQ_DIST_L2,
    ITI_SQ_DIST_DIM64_L2,
    ITI_SQ_DIST_MASK_L2,
    ITI_SQ_DIST_MASK_DIM64_L2,
    ITI_TOPK_FLAT,
    ITI_TOPK_MULTISEARCH,
    ITI_FLAT_IP_TOPK_MULTISEARCH,
    ITI_FLAT_L2_TOPK_MULTISEARCH,
    ITI_TOPK_IVF,
    ITI_INT8_L2_FULL,
    ITI_FLAT_L2_MINS_AT,
    ITI_L2_NORM,
    ITI_FLAT_L2_MINS_INT8_AT,
    ITI_L2_NORM_TYPING_INT8,
    ITI_L2_NORM_FLAT_SUB,
    ITI_FLAT_L2_AT,
    ITI_RESIDUAL_IVF,
    ITI_IVFSQ8_IP8,
    ITI_IVFSQ8_IP4,
    ITI_IVFSQ8_L2,
    ITI_INT8_COS_MASK,
    ITI_INT8_COS_MASK_EXTRASCORE,
    ITI_INT8_COS_SHARE_MASK,
    ITI_INT8_COS_SHARE_MASK_EXTRASCORE,
    ITI_INT8_L2_SHARE_MASK,
    ITI_INT8_L2_MASK,
    ITI_INT8_L2_FULL_SHARE_MASK,
    ITI_INT8_L2_FULL_MASK,
    ITI_MASK_GENERATOR,
    ITI_MASK_WITH_EXTRA_GENERATOR,
    ITI_MASK_WITH_EXTRA_AND_BASEMASK_GENERATOR,
    ITI_MASK_WITH_VAL_GENERATOR,
    ITI_HAMMING_MASK,
    ITI_HAMMING_SHARE_MASK,
    ITI_FLAT_IP_MASK,
    ITI_FLAT_IP_SHARE_MASK,
    ITI_FLAT_IP_MASK_EXTRA_SCORE,
    ITI_FLAT_IP_SHARE_MASK_EXTRA_SCORE,
    ITI_FLAT_IP_EXTRA_SCORE_AND_SCALE,
    ITI_FLAT_IP_NOSCORE_AND_SCALE,
    ASCENDC_ITI_MASK_GENERATOR,
    ASCENDC_ITI_MASK_WITH_EXTRA_GENERATOR,
    ITI_MAX
};

class OpsMngKey {
public:
    explicit OpsMngKey(std::vector<int> &opsKeys) : opsKeys(opsKeys) {};
    bool operator < (const OpsMngKey &k) const
    {
        size_t cmpNum = std::min(this->opsKeys.size(), k.opsKeys.size());
        for (size_t i = 0; i < cmpNum; i++) {
            if (this->opsKeys[i] < k.opsKeys[i]) {
                return true;
            } else if (this->opsKeys[i] > k.opsKeys[i]) {
                return false;
            }
        }
        return false;
    }
    std::vector<int> opsKeys;
};

class DistComputeOpsManager {
public:
    static DistComputeOpsManager &getInstance()
    {
        return *getShared();
    }

    static std::shared_ptr<DistComputeOpsManager> &getShared();

    DistComputeOpsManager() {}

    virtual ~DistComputeOpsManager() {}

    virtual void initialize()
    {
        for (int i = 0; i < static_cast<int>(IndexTypeIdx::ITI_MAX); ++i) {
            std::map<OpsMngKey, std::unique_ptr<AscendOperator>> ops;
            distComputeOps.push_back(std::move(ops));
        }
    }

    virtual void uninitialize()
    {
        distComputeOps.clear();
    }

    virtual std::map<OpsMngKey, std::unique_ptr<AscendOperator>> &getDistComputeOps(IndexTypeIdx type)
    {
        return distComputeOps[(int)type];
    }

    virtual APP_ERROR resetOp(const std::string &opTypeName,
                              IndexTypeIdx indexType,
                              OpsMngKey &opsKey,
                              const std::vector<std::pair<aclDataType, std::vector<int64_t>>> &input,
                              const std::vector<std::pair<aclDataType, std::vector<int64_t>>> &output)
    {
        if (indexType >= IndexTypeIdx::ITI_MAX) {
            APP_LOG_ERROR("invalid indexType: %d\n", static_cast<int>(indexType));
            return APP_ERR_INVALID_PARAM;
        }
        auto& distComputeOpMap = distComputeOps[static_cast<int>(indexType)];

        AscendOpDesc desc(opTypeName);
        for (auto &data : input) {
            desc.addInputTensorDesc(data.first, data.second.size(), data.second.data(), ACL_FORMAT_ND);
        }
        
        for (auto &data : output) {
            desc.addOutputTensorDesc(data.first, data.second.size(), data.second.data(), ACL_FORMAT_ND);
        }
        distComputeOpMap[opsKey] = CREATE_UNIQUE_PTR(AscendOperator, desc);
        bool ret = distComputeOpMap[opsKey]->init();
        APPERR_RETURN_IF_NOT_FMT(ret, APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
            "op init failed, index type: %d, name:%s\n", static_cast<int>(indexType), opTypeName.c_str());
        return APP_ERR_OK;
    }

    virtual APP_ERROR runOp(IndexTypeIdx indexType,
                            OpsMngKey &opsKey,
                            const std::vector<const AscendTensorBase *> &input,
                            const std::vector<const AscendTensorBase *> &output,
                            aclrtStream stream)
    {
        if (indexType >= IndexTypeIdx::ITI_MAX) {
            APP_LOG_ERROR("invalid indexType: %d\n", static_cast<int>(indexType));
            return APP_ERR_INVALID_PARAM;
        }
        AscendOperator *distSqOp = nullptr;
        auto& distComputeOpMap = distComputeOps[static_cast<int>(indexType)];
        if (distComputeOpMap.find(opsKey) != distComputeOpMap.end()) {
            distSqOp = distComputeOpMap[opsKey].get();
        }
        if (distSqOp == nullptr) {
            APP_LOG_ERROR("op not found, index type: %d\n", static_cast<int>(indexType));
            return APP_ERR_ACL_OP_NOT_FOUND;
        }

        std::shared_ptr<std::vector<const aclDataBuffer *>> distSqOpInput(
            new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
        for (auto &data : input) {
            distSqOpInput->emplace_back(aclCreateDataBuffer(data->getVoidData(), data->getSizeInBytes()));
        }

        std::shared_ptr<std::vector<aclDataBuffer *>> distSqOpOutput(
            new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
        for (auto &data : output) {
            distSqOpOutput->emplace_back(aclCreateDataBuffer(data->getVoidData(), data->getSizeInBytes()));
        }

        distSqOp->exec(*distSqOpInput, *distSqOpOutput, stream);

        return APP_ERR_OK;
    }

private:
    std::vector<std::map<OpsMngKey, std::unique_ptr<AscendOperator>>> distComputeOps;
};
}
#endif // ASCEND_DIST_COMPUTE_OPS_MANAGER_INCLUDED