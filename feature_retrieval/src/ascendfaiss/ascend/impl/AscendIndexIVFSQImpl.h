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


#ifndef ASCEND_INDEX_IVFSQ_IMPL_INCLUDED
#define ASCEND_INDEX_IVFSQ_IMPL_INCLUDED

#include <faiss/IndexScalarQuantizer.h>

#include "ascend/AscendIndexIVFSQ.h"
#include "ascend/impl/AscendIndexIVFImpl.h"
#include "ascenddaemon/impl/IndexIVFSQ.h"

struct fp16;

namespace faiss {
namespace ascend {
class AscendIndexIVFSQImpl : public AscendIndexIVFImpl {
public:
    // Construct an index from CPU IndexIVFSQ
    AscendIndexIVFSQImpl(AscendIndexIVFSQ *intf, const faiss::IndexIVFScalarQuantizer *index,
        AscendIndexIVFSQConfig config = AscendIndexIVFSQConfig());

    // Construct an empty index
    AscendIndexIVFSQImpl(AscendIndexIVFSQ *intf, int dims, int nlist,
        faiss::ScalarQuantizer::QuantizerType qtype = ScalarQuantizer::QuantizerType::QT_8bit,
        faiss::MetricType metric = MetricType::METRIC_L2, bool encodeResidual = true,
        AscendIndexIVFSQConfig config = AscendIndexIVFSQConfig());

    virtual ~AscendIndexIVFSQImpl();

    // Initialize ourselves from the given CPU index; will overwrite
    // all data in ourselves
    void copyFrom(const faiss::IndexIVFScalarQuantizer *index);

    // Copy ourselves to the given CPU index; will overwrite all data
    // in the index instance
    void copyTo(faiss::IndexIVFScalarQuantizer *index) const;

    void train(idx_t n, const float *x);

    // AscendIndex object is NON-copyable
    AscendIndexIVFSQImpl(const AscendIndexIVFSQImpl&) = delete;
    AscendIndexIVFSQImpl& operator=(const AscendIndexIVFSQImpl&) = delete;

protected:
    // Construct an empty index when AscendIndexIVFSQ is parent class for custom classes
    AscendIndexIVFSQImpl(AscendIndexIVFSQ *intf, int dims, int nlist, bool dummy,
        faiss::ScalarQuantizer::QuantizerType qtype = ScalarQuantizer::QuantizerType::QT_8bit,
        faiss::MetricType metric = MetricType::METRIC_L2, bool encodeResidual = true,
        AscendIndexIVFSQConfig config = AscendIndexIVFSQConfig());

    std::shared_ptr<::ascend::Index> createIndex(int deviceId) override;

    // Called from AscendIndex for add/add_with_ids
    void addImpl(int n, const float *x, const idx_t *ids) override;

    virtual void checkParams() const;

    void trainResidualQuantizer(idx_t n, const float *x);

    virtual void updateDeviceSQTrainedValue();

    virtual void copyCodes(const faiss::IndexIVFScalarQuantizer *index);

    virtual void calcPrecompute(const uint8_t *codes, float *compute, size_t n, float *xMem = nullptr);

    size_t getAddElementSize() const override;

    void indexIVFSQFastAdd(IndexParam<uint16_t, uint16_t, ascend_idx_t> param,
                           std::vector<std::vector<int>> &offsumMap,
                           std::vector<std::vector<int>> &deviceAddNumMap,
                           std::vector<std::vector<float>> &precomputeVals,
                           const InvertedLists *ivf);

    void indexIVFSQAdd(IndexParam<uint8_t, uint8_t, ascend_idx_t> param, const float *precomputedVal);

    void indexIVFSQUpdateTrainedValue(int deviceId, int dim, uint16_t *vmin, uint16_t *vdiff) const;

    inline ::ascend::IndexIVFSQ<float>* getActualIndex(int deviceId) const
    {
        FAISS_THROW_IF_NOT_FMT(indexes.find(deviceId) != indexes.end(),
                               "deviceId is out of range, deviceId=%d.", deviceId);
        FAISS_THROW_IF_NOT(aclrtSetDevice(deviceId) == ACL_ERROR_NONE);
        std::shared_ptr<::ascend::Index> index = indexes.at(deviceId);
        auto *pIndex = dynamic_cast<::ascend::IndexIVFSQ<float> *>(index.get());
        FAISS_THROW_IF_NOT_FMT(pIndex != nullptr, "Invalid index device id: %d\n", deviceId);
        return pIndex;
    }

    void checkInvertedLists(const faiss::IndexIVFScalarQuantizer *index);

    void copyFromInner(const faiss::IndexIVFScalarQuantizer *index);

    AscendIndexIVFSQ *intf_;
    faiss::ScalarQuantizer sq;

private:
    AscendIndexIVFSQConfig ivfsqConfig;

    // whether to encode code by residual
    bool byResidual;
};
} // ascend
} // faiss
#endif