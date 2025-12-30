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


#ifndef ASCEND_INDEX_IVFSP_IMPL_INCLUDED
#define ASCEND_INDEX_IVFSP_IMPL_INCLUDED

#include <faiss/IndexScalarQuantizer.h>
#include "ascend/impl/AscendIndexIVFImpl.h"
#include "ascend/ivfsp/AscendIndexIVFSP.h"
#include "ascendsearch/ascend/custom/AscendIndexIVFSPSQ.h"

namespace faiss {
namespace ascend {

class AscendIndexIVFSPImpl : public AscendIndexImpl {
public:
    AscendIndexIVFSPImpl(std::shared_ptr<AscendIndexIVFSP> intf,
        std::shared_ptr<faiss::ascendSearch::AscendIndexIVFSPSQ> ivfspsq,
        int dims, int nonzeroNum, int nlist, faiss::ScalarQuantizer::QuantizerType qType,
        faiss::MetricType metric, const AscendIndexIVFSPConfig &config);

    AscendIndexIVFSPImpl(AscendIndexIVFSP *intf, int dims, int nonzeroNum, int nlist,
        faiss::ScalarQuantizer::QuantizerType qType, faiss::MetricType metric, const AscendIndexIVFSPConfig &config);

    void addCodeBook(const char *codeBookPath);

    void addCodeBook(const AscendIndexIVFSPImpl &codeBookSharedImpl);

    virtual ~AscendIndexIVFSPImpl();

    // `x` need to be resident on CPU
    // Handles paged adds if the add set is too large;
    void add(idx_t n, const float *x);

    // `x` and `ids` need to be resident on the CPU;
    // Handles paged adds if the add set is too large;
    void add_with_ids(idx_t n, const float *x, const idx_t *ids); // override;

    size_t remove_ids(const faiss::IDSelector &sel);

    void reset();

    void loadAllData(const char *dataPath);

    void saveAllData(const char *dataPath);

    static std::shared_ptr<AscendIndexIVFSPImpl> loadAllData(std::shared_ptr<AscendIndexIVFSP> intf,
        const AscendIndexIVFSPConfig &config, const uint8_t *data, size_t dataLen,
        const std::shared_ptr<AscendIndexIVFSPImpl> codeBookSharedIdx);

    void saveAllData(uint8_t *&data, size_t &dataLen) const;

    void trainCodeBook(const AscendIndexCodeBookInitParams &codeBookInitParams) const;

    void trainCodeBookFromMem(const AscendIndexCodeBookInitFromMemParams &codeBookInitFromMemParams) const;
    // AscendIndex object is NON-copyable
    AscendIndexIVFSPImpl(const AscendIndexIVFSPImpl&) = delete;
    AscendIndexIVFSPImpl& operator=(const AscendIndexIVFSPImpl&) = delete;

    void CheckIndexParams(IndexImplBase &index, bool checkFilterable = false) const override;
    void setNumProbes(int nprobes);

    void search(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels) const;

    void search_with_filter(idx_t n, const float *x, idx_t k,
                            float *distances, idx_t *labels, const void *filters) const;

    static void SearchMultiIndex(int deviceId, std::vector<faiss::ascendSearch::AscendIndex*> indexes,
        idx_t n, const float *x, idx_t k,
        float *distances, idx_t *labels, bool merged);

    static void SearchWithFilterMultiIndex(int deviceId, std::vector<faiss::ascendSearch::AscendIndex*> indexes,
        idx_t n, const float *x, idx_t k,
        float *distances, idx_t *labels, const void *filters, bool merged);

    static void SearchWithFilterMultiIndex(int deviceId, std::vector<faiss::ascendSearch::AscendIndex*> indexes,
        idx_t n, const float *x, idx_t k,
        float *distances, idx_t *labels, void *filters[], bool merged);

    faiss::ascendSearch::AscendIndexIVFSPSQ* GetIVFSPSQPtr() const override;

protected:

    // Called from AscendIndex for add/add_with_ids.
    void addImpl(int n, const float *x, const idx_t *ids) override;

    std::shared_ptr<::ascend::Index> createIndex(int deviceId) override;

    AscendIndexIVFSP *intf_;

private:
    int ivfspNonzeroNum;
    
    int ivfspNList;
    
    AscendIndexIVFSPConfig ivfspConfig;

    std::shared_ptr<faiss::ascendSearch::AscendIndexIVFSPSQ> pIVFSPSQ;

    friend void AscendIndexIVFSP::setVerbose(bool verbose);

    friend void Search(std::vector<AscendIndex *> indexes, idx_t n, const float *x, idx_t k,
        float *distances, idx_t *labels, bool merged);
    
    friend void SearchWithFilter(std::vector<AscendIndex *> indexes, idx_t n, const float *x, idx_t k,
        float *distances, idx_t *labels, const void *filters, bool merged);
    
    friend void SearchWithFilter(std::vector<AscendIndex *> indexes, idx_t n, const float *x, idx_t k,
        float *distances, idx_t *labels, void *filters[], bool merged);

    static std::vector<std::shared_mutex> mtxVec;
};
} // ascend
} // faiss
#endif