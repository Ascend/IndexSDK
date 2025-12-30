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


#ifndef ASCEND_INDEX_IVF_SP_SQ_IMPL_INCLUDED
#define ASCEND_INDEX_IVF_SP_SQ_IMPL_INCLUDED

#include <faiss/IndexScalarQuantizer.h>
#include <faiss/MetaIndexes.h>
#include <ascendsearch/ascend/custom/AscendIndexIVFSPSQ.h>
#include "ascend/impl/AscendIndexImpl.h"


namespace faiss {
namespace ascendSearch {

class AscendIndexIVFSPSQImpl : public AscendIndexImpl {
public:
    AscendIndexIVFSPSQImpl(int dims, int dims2, int k, int nlist, AscendIndex *intf,
                       faiss::ScalarQuantizer::QuantizerType qType = ScalarQuantizer::QuantizerType::QT_8bit,
                       faiss::MetricType metric = MetricType::METRIC_L2, bool encodeResidual = true,
                       AscendIndexIVFSPSQConfig config = AscendIndexIVFSPSQConfig());

    virtual ~AscendIndexIVFSPSQImpl();

    void setNumProbes(int nprobes);

    void add(idx_t n, const float *feature, const idx_t *ids);

    // `x` and `ids` need to be resident on the CPU;
    // Handles paged adds if the add set is too large;
    void add_with_ids(idx_t n, const float *x, const idx_t *ids);

    void loadAllData(const char *dataPath);

    void loadAllData(const char *dataPath, const AscendIndexIVFSPSQImpl &loadedIndex);

    void loadAllData(const uint8_t* data, size_t dataLen);

    void loadAllData(const uint8_t* data, size_t dataLen, const AscendIndexIVFSPSQImpl &loadedIndex);

    void loadCodeBookOnly(const uint8_t* data, size_t dataLen);

    void saveAllData(const char *dataPath);

    void saveAllData(uint8_t* &data, size_t &dataLen);

    void searchPostProcess(size_t devices, std::vector<std::vector<float>> &dist,
        std::vector<std::vector<ascend_idx_t>> &label, int n, int k, float *distances,
        idx_t *labels) const override;

    void reset();

    // AscendIndex object is NON-copyable
    AscendIndexIVFSPSQImpl(const AscendIndexIVFSPSQImpl&) = delete;
    AscendIndexIVFSPSQImpl& operator=(const AscendIndexIVFSPSQImpl&) = delete;

    void trainCodeBook(const AscendIndexCodeBookTrainerConfig &codeBookTrainerConfig);

    void addCodeBook(int n, int dim, const float *x, idx_t *offset);

    void addCodeBook(const AscendIndexIVFSPSQImpl &loadedIndex);

    // Overridden to actually perform the search
    void searchImpl(int n, const float *x, int k, float *distances, idx_t *labels) const override;

    void search_with_masks(idx_t n, const float *x, idx_t k,
                           float *distances, idx_t *labels, const void *mask) const;

    void search_with_filter(idx_t n, const float *x, idx_t k,
                            float *distances, idx_t *labels, const void *filters) const;

    void search_with_filter(idx_t n, const float *x, idx_t k,
                            float *distances, idx_t *labels, float *l1distances, const void *filters) const;

    void train(idx_t n, const float *x);

    size_t remove_ids(const std::vector<idx_t> sel) ;

    size_t remove_ids(const idx_t minRange, const idx_t maxRange) ;

    void addCodeBook(const std::string &path);

    int getDims() const;

    int getDims2() const;

    int getNumList() const;

    bool getFilterable() const;

    void addFinish() const;

    void checkAndSetAddFinish() const;

public:
    // cpu version for training and endcoding
    faiss::ScalarQuantizer spSq;

protected:
    int CreateIndex(rpcContext ctx) override;

    // Compare the config parameters(excecpt deviceList) of index with another index
    void checkParamsSame(AscendIndexImpl &index) override;

    void initDeviceAddNumMap();

    // Called from AscendIndex for add
    void addPaged(int n, const float* x, const idx_t* ids);

    void getCodeword(int n, const float *feature, float *codeWord, idx_t *ids);

    void addImpl(int n, const float *x, const idx_t *ids) override;

    void addCodeImpl(int n, const float *x, const idx_t *ids);

    void getBaseImpl(int deviceId, int offset, int n, char *x) const override;

    size_t getAddElementSize() const override;

    size_t getBaseElementSize() const override;

private:
    void checkParams();

    void updateDeviceSPSQTrainedValue();

    void matMul(float *dst, const float *c, const float *b, size_t n, size_t k, size_t m, bool transpose = false);

    void checkSharedCodebookParams(const AscendIndexIVFSPSQImpl &loadedIndex) const;

private:
    AscendIndexIVFSPSQConfig spSqConfig;
    int dims2;
    std::shared_ptr<std::vector<float>> codeBook;
    std::vector<idx_t> codeOffset;
    bool codebookFinished = false;
    std::vector<std::vector<int>> deviceAddNumMap;

    // whether to encode code by residual
    bool byResidual;
    int nCentroid;
    int k;
    int nlist;

    // top nprobe for quantizer searching
    int nprobe;
    bool oriFeature;
};
} // ascend
} // faiss
#endif // ASCEND_INDEX_SP_INCLUDED