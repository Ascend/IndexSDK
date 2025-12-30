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


#include "ascend/custom/impl/AscendIndexIVFSQFuzzyImpl.h"

#include <set>

#include "ascend/AscendIndexQuantizerImpl.h"
#include "ascend/utils/AscendIVFAddInfo.h"
#include "AscendUtils.h"

namespace faiss {
namespace ascend {
// Default dim in case of nullptr index
const int DEFAULT_DIM = 64;

// Default nlist in case of nullptr index
const int DEFAULT_NLIST = 8192;

// For copy index we append information to sq.trained, but standard sq trained size is 2
const int SQ_VAL_SIZE = 2;

const size_t FUZZY_PARAM_COUNT = 3;
const size_t FUZZY_K_INDEX = 2;
const size_t THRESHOLD_INDEX = 1;

const float EPSILON = 1e-6;
const int MAX_FUZZY_K = 10;

AscendIndexIVFSQFuzzyImpl::AscendIndexIVFSQFuzzyImpl(AscendIndexIVFSQ *intf, int dims, int nlist, bool dummy,
    faiss::ScalarQuantizer::QuantizerType qtype, faiss::MetricType metric, AscendIndexIVFSQConfig config)
    : AscendIndexIVFSQImpl(intf, dims, nlist, false, qtype, metric, false, config), ivfsqFuzzyConfig(config)
{
    FAISS_THROW_IF_NOT_MSG(metric == MetricType::METRIC_INNER_PRODUCT, "MetricType must be METRIC_INNER_PRODUCT.");
    VALUE_UNUSED(dummy);
}

AscendIndexIVFSQFuzzyImpl::~AscendIndexIVFSQFuzzyImpl() {}

void AscendIndexIVFSQFuzzyImpl::copyFrom(const faiss::IndexIVFScalarQuantizer *index)
{
    APP_LOG_INFO("AscendIndexIVFSQFuzzy copyFrom operation started.\n");
    FAISS_THROW_IF_NOT_MSG(index != nullptr, "index is nullptr.");
    initFuzzyTotal(index);
    copyFromInner(index);
    APP_LOG_INFO("AscendIndexIVFSQFuzzy copyFrom operation finished.\n");
}

void AscendIndexIVFSQFuzzyImpl::initFuzzyTotal(const faiss::IndexIVFScalarQuantizer *index)
{
    // get fuzzyTotal from index->invlists
    APP_LOG_INFO("AscendIndexIVFSQFuzzy initFuzzyTotal operation started.\n");
    fuzzyTotal = 0;
    InvertedLists *ivf = index->invlists;
    if (ivf != nullptr) {
        auto arrayIvf = dynamic_cast<ArrayInvertedLists *>(ivf);
        FAISS_THROW_IF_NOT_MSG(arrayIvf != nullptr, "index->invlists is invalild type");
        FAISS_THROW_IF_NOT_MSG(arrayIvf->ids.size() == static_cast<size_t>(nlist),
            "ids of invlists is invalild");
        for (int i = 0; i < nlist; i++) {
            fuzzyTotal += ivf->list_size(i);
        }
    } else {
        APP_LOG_ERROR("index->invlists is nullptr.");
    }
    APP_LOG_INFO("AscendIndexIVFSQFuzzy initFuzzyTotal operation finished.\n");
}

/**
 * append fuzzy params to sq.trained to implement Copy Ascend Index to CPU
 */
void AscendIndexIVFSQFuzzyImpl::appendTrained()
{
    APP_LOG_INFO("AscendIndexIVFSQFuzzy appendTrained operation started.\n");
    sq.trained.push_back((float(fuzzyType)));
    sq.trained.push_back((float(fuzzyK)));
    sq.trained.push_back((float(threshold)));
    APP_LOG_INFO("AscendIndexIVFSQFuzzy appendTrained operation finished.\n");
}

/**
 * split fuzzy params to sq.trained to implement Copy CPU Index to Ascend
 */
void AscendIndexIVFSQFuzzyImpl::splitTrained()
{
    APP_LOG_INFO("AscendIndexIVFSQFuzzy splitTrained operation started.\n");
    size_t sqSize = sq.trained.size();
    if (sqSize >= sq.d * SQ_VAL_SIZE + FUZZY_PARAM_COUNT) {
        this->fuzzyType = int(sq.trained[sqSize - FUZZY_PARAM_COUNT]);
        this->fuzzyK = static_cast<size_t>(sq.trained[sqSize - FUZZY_K_INDEX]);
        this->threshold = float(sq.trained[sqSize - THRESHOLD_INDEX]);

        FAISS_THROW_IF_NOT_FMT(this->fuzzyType == TYPE_T,
            "copyFrom: fuzzyType %d is invalid", this->fuzzyType);
        FAISS_THROW_IF_NOT_FMT(this->fuzzyK > 0 && this->fuzzyK <= static_cast<size_t>(MAX_FUZZY_K),
            "copyFrom: fuzzyK %zu is invalid", this->fuzzyK);
        FAISS_THROW_IF_NOT_FMT(this->threshold >= 0 && this->threshold <= static_cast<float>(fuzzyK - 1),
            "copyFrom: threshold %f is invalid", this->threshold);

        sq.trained.resize(sqSize - FUZZY_PARAM_COUNT);
    } else {
        FAISS_THROW_FMT("AscendIndexIVFSQFuzzy splitTrained sqSize %zu is invalid, expect %zu",
            sqSize, sq.d * SQ_VAL_SIZE + FUZZY_PARAM_COUNT);
    }
    APP_LOG_INFO("AscendIndexIVFSQFuzzy splitTrained operation finished.\n");
}

void AscendIndexIVFSQFuzzyImpl::updateDeviceSQTrainedValue()
{
    APP_LOG_INFO("AscendIndexIVFSQFuzzy updateDeviceSQTrainedValue operation started.\n");
    if (sq.trained.size() > sq.d * SQ_VAL_SIZE) {
        splitTrained();
    }
    FAISS_THROW_IF_NOT_MSG(sq.trained.size() == SQ_VAL_SIZE * sq.d,
        "sq.trained.size wrong before updateDeviceSQTrainedValue");
    AscendIndexIVFSQImpl::updateDeviceSQTrainedValue();
    APP_LOG_INFO("AscendIndexIVFSQFuzzy updateDeviceSQTrainedValue operation finished.\n");
}

void AscendIndexIVFSQFuzzyImpl::copyTo(faiss::IndexIVFScalarQuantizer *index)
{
    APP_LOG_INFO("AscendIndexIVFSQFuzzy copyTo operation started.\n");
    FAISS_THROW_IF_NOT_MSG(index != nullptr, "index is nullptr.");
    appendTrained();
    // For AscendIndexIVFSQ resize index->sq.trained, so this index cannot call AscendIndexIVFSQ::copyTo(index);
    AscendIndexIVFImpl::copyTo(index);
    index->by_residual = false;
    index->sq = sq;
    index->code_size = sq.code_size;

    // Recovery sq status
    sq.trained.resize(SQ_VAL_SIZE * sq.d);

    InvertedLists *ivf = new ArrayInvertedLists(nlist, index->code_size);
    index->replace_invlists(ivf, true);

    if (this->intf_->is_trained && this->intf_->ntotal > 0) {
        // use for(deviceList) rather than for(auto& index : indexMap),
        // to ensure merged codes and ids in sequence
        for (size_t i = 0; i < indexConfig.deviceList.size(); i++) {
            int deviceId = indexConfig.deviceList[i];
            indexIVFFastGetListCodes(deviceId, nlist, ivf);
        }
    }
    APP_LOG_INFO("AscendIndexIVFSQFuzzy copyTo operation finished.\n");
}

void AscendIndexIVFSQFuzzyImpl::train(idx_t n, const float *x)
{
    AscendIndexIVFSQImpl::train(n, x);
}

void AscendIndexIVFSQFuzzyImpl::setFuzzyK(int value)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APP_LOG_INFO("AscendIndexIVFSQFuzzy setFuzzyK value=%d.\n", value);
    if (value <= 0 || value > MAX_FUZZY_K) {
        APP_LOG_ERROR("invalid value %d, should be (0, %d]\n", value, MAX_FUZZY_K);
        return;
    }
    if (threshold + 1.0 > static_cast<float>(value)) {
        APP_LOG_ERROR("invalid value %d, value need >= %f + 1\n", value, threshold);
        return;
    }
    fuzzyK = static_cast<size_t>(value);
}

int AscendIndexIVFSQFuzzyImpl::getFuzzyK() const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    return static_cast<int>(fuzzyK);
}

void AscendIndexIVFSQFuzzyImpl::setThreshold(float value)
{
    APP_LOG_INFO("AscendIndexIVFSQFuzzy setThreshold value=%f.\n", value);
    if (value < 0 || value > static_cast<float>(fuzzyK) - 1) {
        APP_LOG_ERROR("invalid value %f, value should be [0, %f]\n", value, static_cast<float>(fuzzyK) - 1);
        return;
    }
    threshold = value;
}

float AscendIndexIVFSQFuzzyImpl::getThreshold() const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    return threshold;
}

void AscendIndexIVFSQFuzzyImpl::reset()
{
    APP_LOG_INFO("AscendIndexIVFSQFuzzy reset operation started.\n");
    threadUnsafeReset();
    this->fuzzyTotal = 0;
    APP_LOG_INFO("AscendIndexIVFSQFuzzy reset operation finished.\n");
}
} // ascend
} // faiss
