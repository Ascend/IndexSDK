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


#ifndef ASCEND_INDEX_IVF_SP_SQ_INCLUDED
#define ASCEND_INDEX_IVF_SP_SQ_INCLUDED

#include <faiss/IndexScalarQuantizer.h>
#include <faiss/MetaIndexes.h>
#include <ascendsearch/ascend/AscendIndex.h>

namespace faiss {
namespace ascendSearch {

const int64_t IVF_SP_SQ_DEFAULT_MEM = 0x8000000; // 0x8000000 mean 128M(resource mem pool's size)

struct AscendIndexIVFSPSQConfig : public AscendIndexConfig {
    inline AscendIndexIVFSPSQConfig() : AscendIndexConfig({ 0 }, IVF_SP_SQ_DEFAULT_MEM)
    {}

    inline AscendIndexIVFSPSQConfig(std::initializer_list<int> devices, int64_t resourceSize = IVF_SP_SQ_DEFAULT_MEM)
        : AscendIndexConfig(devices, resourceSize)
    {}

    inline AscendIndexIVFSPSQConfig(std::vector<int> devices, int64_t resourceSize = IVF_SP_SQ_DEFAULT_MEM)
        : AscendIndexConfig(devices, resourceSize)
    {}

    int handleBatch = 64;
    int nprobe = 64;
    int searchListSize = 32768;
};

struct AscendIndexCodeBookTrainerConfig {
    AscendIndexCodeBookTrainerConfig() {}

    int numIter = 1;
    int device = 0;
    float ratio = 1.0;
    int batchSize = 32768;
    int codeNum = 32768;
    bool verbose = true;
    std::string codeBookOutputDir = "";
    std::string learnDataPath = "";
    const float *memLearnData = nullptr; // 如果useMemLearnData，使用该指针指向的数据
    size_t memLearnDataSize = 0; // memLearnData指向的数据的元素数量 (float类型数据的数量而非总数据长度)
    bool trainAndAdd = false;
};

class AscendIndexIVFSPSQImpl;
class AscendIndexIVFSPSQ : public AscendIndex {
public:
    AscendIndexIVFSPSQ(int dims, int dims2, int k, int nlist,
                       faiss::ScalarQuantizer::QuantizerType qType = ScalarQuantizer::QuantizerType::QT_8bit,
                       faiss::MetricType metric = MetricType::METRIC_L2, bool encodeResidual = true,
                       AscendIndexIVFSPSQConfig config = AscendIndexIVFSPSQConfig());

    virtual ~AscendIndexIVFSPSQ();

    void reset();
    
    void add(idx_t n, const float *feature, const idx_t *ids);

    /* 通过索引路径加载数据 */
    void loadAllData(const char *dataPath);

    /* 通过索引路径加载数据，允许传入加载好的索引实例共享码本 */
    void loadAllData(const char *dataPath, const AscendIndexIVFSPSQ &loadedIndex);

    /* 通过指针加载数据 */
    void loadAllData(const uint8_t* data, size_t dataLen);

    /* 通过指针加载数据，允许传入加载好的索引实例共享码本 */
    void loadAllData(const uint8_t* data, size_t dataLen, const AscendIndexIVFSPSQ &loadedIndex);

    /* 通过指针，加载一个仅有码本没有底库数据的索引 */
    void loadCodeBookOnly(const uint8_t* data, size_t dataLen);

    /* 通过索引路径存储数据 */
    void saveAllData(const char *dataPath);

    /* 通过指针存储数据 */
    void saveAllData(uint8_t* &data, size_t &dataLen) const;

    /* 静态方法创建索引实例并通过指针加载数据 */
    static std::shared_ptr<AscendIndexIVFSPSQ> createAndLoadData(const uint8_t* data, size_t dataLen,
        const std::vector<int> &devices, int64_t resourceSize);

    /* 静态方法创建索引实例并通过指针加载数据，并允许传入加载好的索引实例共享码本 */
    static std::shared_ptr<AscendIndexIVFSPSQ> createAndLoadData(const uint8_t* data, size_t dataLen,
        const std::vector<int> &devices, int64_t resourceSize, const AscendIndexIVFSPSQ &loadedIndex);

    void setNumProbes(int nprobes);

    // Returns the vectors of we contain
    void getBase(int deviceId, char* xb) const;

    // AscendIndex object is NON-copyable
    AscendIndexIVFSPSQ(const AscendIndexIVFSPSQ&) = delete;
    AscendIndexIVFSPSQ& operator=(const AscendIndexIVFSPSQ&) = delete;

    void trainCodeBook(const AscendIndexCodeBookTrainerConfig &codeBookTrainerConfig) const;

    void addCodeBook(int n, int dim, const float *x, idx_t *offset);

    void addCodeBook(const AscendIndexIVFSPSQ &loadedIndex);

    void search_with_masks(idx_t n, const float *x, idx_t k,
                           float *distances, idx_t *labels, const void *mask) const;

    void search_with_filter(idx_t n, const float *x, idx_t k,
                            float *distances, idx_t *labels, const void *filters) const;

    void search_with_filter(idx_t n, const float *x, idx_t k,
                            float *distances, idx_t *labels, float *l1distances, const void *filters) const;

    void train(idx_t n, const float *x);

    static void SearchMultiIndex(std::vector<AscendIndex*> indexes, idx_t n, const float *x, idx_t k,
                       float *distances, idx_t *labels, bool merged);

    static void SearchWithFilterMultiIndex(std::vector<AscendIndex *> indexes, idx_t n,
        const float *x, idx_t k, float *distances, idx_t *labels, const void *filters, bool merged);

    static void SearchWithFilterMultiIndex(std::vector<AscendIndex *> indexes, idx_t n,
        const float *x, idx_t k, float *distances, idx_t *labels, void *filters[], bool merged);

    size_t remove_ids(const std::vector<idx_t> sel) ;

    size_t remove_ids(const idx_t minRange, const idx_t maxRange) ;

    void addCodeBook(const std::string &path);

    int getDims() const;

    int getDims2() const;

    int getNumList() const;

    bool getFilterable() const;

    faiss::MetricType getMetric() const;

    faiss::ScalarQuantizer::QuantizerType getQuantizerType() const;

    void addFinish();

protected:
    std::shared_ptr<AscendIndexIVFSPSQImpl> impl_;

private:
    bool static checkCodeBookOnlyIndex(const uint8_t *data, size_t dataLen); // 检查当前序列化的索引是否是一个仅有码本信息的索引
};
} // ascend
} // faiss
#endif // ASCEND_INDEX_SP_INCLUDED