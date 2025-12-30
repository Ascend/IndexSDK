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


#ifndef ASCEND_INDEX_IVFSQC_IMPL_INCLUDED
#define ASCEND_INDEX_IVFSQC_IMPL_INCLUDED

#include <faiss/IndexScalarQuantizer.h>

#include "ascend/custom/impl/AscendIndexIVFSQFuzzyImpl.h"
#include "index_custom/IndexIVFSQCIPAicpu.h"

struct fp16;

namespace faiss {
namespace ascend {
class AscendIndexIVFSQCImpl : public AscendIndexIVFSQFuzzyImpl {
public:
    virtual ~AscendIndexIVFSQCImpl();

    void train(idx_t n, const float *x);

    void fineTune(size_t n, const float *x);

    void copyFrom(const faiss::IndexIVFScalarQuantizer *index);

    // AscendIndex object is NON-copyable
    AscendIndexIVFSQCImpl(const AscendIndexIVFSQCImpl &) = delete;

    AscendIndexIVFSQCImpl &operator = (const AscendIndexIVFSQCImpl &) = delete;

protected:
    // Construct an empty index when AscendIndexIVFSQC is parent class for other custom classes
    AscendIndexIVFSQCImpl(AscendIndexIVFSQ *intf, int dimIn, int dimOut, int nlist, bool dummy,
        faiss::ScalarQuantizer::QuantizerType qtype = ScalarQuantizer::QuantizerType::QT_8bit,
        faiss::MetricType metric = MetricType::METRIC_INNER_PRODUCT,
        AscendIndexIVFSQConfig config = AscendIndexIVFSQConfig());

    std::shared_ptr<::ascend::Index> createIndex(int deviceId) override;

    void split(int n, const float *x);

    void checkParams() const override;

    void trainScalarQuantizer(idx_t n, const float *x);

    void updateCompressValue();

    void trainCompress(idx_t n, const float *x);

    void computeCompressIndex(idx_t n, const float *x);

    void computeCompressValue(idx_t n, const float *x);

    void compress(idx_t n, const float *x, float *res) const;

    void normalize(idx_t n, float *x) const;

    // append CompressData to sq.trained to clone index
    void appendCompressData();

    // split CompressData from sq.trained to load from a index
    void splitCompressData();

    void appendTrained() override;

    void splitTrained() override;

    void initTrainedValue();

    inline ::ascend::IndexIVFSQCIPAicpu* getActualIndex (int deviceId) const
    {
        FAISS_THROW_IF_NOT_FMT(indexes.find(deviceId) != indexes.end(),
                               "deviceId is out of range, deviceId=%d.", deviceId);
        FAISS_THROW_IF_NOT(aclrtSetDevice(deviceId) == ACL_ERROR_NONE);
        std::shared_ptr<::ascend::Index> index = indexes.at(deviceId);
        auto *pIndex = dynamic_cast<::ascend::IndexIVFSQCIPAicpu *>(index.get());
        FAISS_THROW_IF_NOT_FMT(pIndex != nullptr, "Invalid index device id: %d\n", deviceId);
        return pIndex;
    }

protected:
    // cpu version for training and endcoding
    int dimIn;

    int dimOut;

    bool addImplIp = false;

    AscendIndexIVFSQConfig ivfsqcConfig;

    AscendIndexIVFSQConfig finetuneConfig;

    size_t ratio;

    std::vector<float> correlation;

    std::vector<std::vector<float>> compressValue;

    std::vector<std::vector<size_t>> compressIndex;
};
} // ascend
} // faiss
#endif