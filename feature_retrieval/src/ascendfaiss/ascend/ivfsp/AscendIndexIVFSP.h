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


#ifndef ASCEND_INDEX_IVF_SP_INCLUDED
#define ASCEND_INDEX_IVF_SP_INCLUDED

#include <memory>
#include <faiss/IndexScalarQuantizer.h>
#include "ascend/AscendIndex.h"

namespace faiss {
namespace ascend {

const int64_t IVF_SP_DEFAULT_MEM = 0x8000000; // 0x8000000 mean 128M(resource mem pool's size)

struct AscendIndexIVFSPConfig : public AscendIndexConfig {
    inline AscendIndexIVFSPConfig() : AscendIndexConfig({ 0 }, IVF_SP_DEFAULT_MEM, DEFAULT_BLOCK_SIZE) {}

    inline explicit AscendIndexIVFSPConfig(std::initializer_list<int> devices,
        int64_t resources = IVF_SP_DEFAULT_MEM, uint32_t blockSize = DEFAULT_BLOCK_SIZE)
        : AscendIndexConfig(devices, resources, blockSize) {}

    inline explicit AscendIndexIVFSPConfig(std::vector<int> devices,
        int64_t resources = IVF_SP_DEFAULT_MEM, uint32_t blockSize = DEFAULT_BLOCK_SIZE)
        : AscendIndexConfig(devices, resources, blockSize)
    {}

    int handleBatch = 64;
    int nprobe = 64;
    int searchListSize = 32768;
};

struct AscendIndexCodeBookInitParams {
    AscendIndexCodeBookInitParams(int numIter, int device, float ratio, int batchSize, int codeNum,
        std::string codeBookOutputDir, std::string learnDataPath, bool verbose) : numIter(numIter), device(device),
        ratio(ratio), batchSize(batchSize), codeNum(codeNum), codeBookOutputDir(codeBookOutputDir),
        learnDataPath(learnDataPath), verbose(verbose) {}
 
    int numIter = 1;
    int device = 0;
    float ratio = 1.0;
    int batchSize = 32768;
    int codeNum = 32768;
    std::string codeBookOutputDir = "";
    std::string learnDataPath = "";
    bool verbose = true;
};

struct AscendIndexCodeBookInitFromMemParams {
    AscendIndexCodeBookInitFromMemParams(int numIter, int device, float ratio, int batchSize, int codeNum,
        bool verbose, std::string codeBookOutputDir,
        const float *memLearnData, size_t memLearnDataSize, bool isTrainAndAdd) : numIter(numIter), device(device),
        ratio(ratio), batchSize(batchSize), codeNum(codeNum), verbose(verbose), codeBookOutputDir(codeBookOutputDir),
        memLearnData(memLearnData), memLearnDataSize(memLearnDataSize), isTrainAndAdd(isTrainAndAdd) {}
 
    int numIter = 1;
    int device = 0;
    float ratio = 1.0;
    int batchSize = 32768;
    int codeNum = 32768;
    bool verbose = true;
    std::string codeBookOutputDir = "";
    const float *memLearnData = nullptr; // 如果useMemLearnData，使用该指针指向的数据
    size_t memLearnDataSize = 0; // memLearnData指向的数据的元素数量 (float类型数据的数量而非总数据长度)
    bool isTrainAndAdd = false;
};

class AscendIndexIVFSPImpl;
class AscendIndexIVFSP : public AscendIndex {
public:
    AscendIndexIVFSP(int dims, int nonzeroNum, int nlist, const char *codeBookPath,
        faiss::ScalarQuantizer::QuantizerType qType = ScalarQuantizer::QuantizerType::QT_8bit,
        faiss::MetricType metric = MetricType::METRIC_L2,
        AscendIndexIVFSPConfig config = AscendIndexIVFSPConfig());

    AscendIndexIVFSP(int dims, int nonzeroNum, int nlist, const AscendIndexIVFSP &codeBookSharedIdx,
        faiss::ScalarQuantizer::QuantizerType qType = ScalarQuantizer::QuantizerType::QT_8bit,
        faiss::MetricType metric = MetricType::METRIC_L2,
        AscendIndexIVFSPConfig config = AscendIndexIVFSPConfig());
    
    AscendIndexIVFSP(int dims, int nonzeroNum, int nlist,
        faiss::ScalarQuantizer::QuantizerType qType = ScalarQuantizer::QuantizerType::QT_8bit,
        faiss::MetricType metric = MetricType::METRIC_L2,
        AscendIndexIVFSPConfig config = AscendIndexIVFSPConfig());

    virtual ~AscendIndexIVFSP();

    static std::shared_ptr<AscendIndexIVFSP> loadAllData(const AscendIndexIVFSPConfig &config, const uint8_t *data,
        size_t dataLen, const AscendIndexIVFSP *codeBookSharedIdx = nullptr);

    void saveAllData(uint8_t *&data, size_t &dataLen) const;

    void addCodeBook(const char *codeBookPath);

    void trainCodeBook(const AscendIndexCodeBookInitParams &codeBookInitParams) const;

    void trainCodeBookFromMem(const AscendIndexCodeBookInitFromMemParams &codeBookInitFromMemParams) const;

    // `x` need to be resident on CPU
    // Handles paged adds if the add set is too large;
    void add(idx_t n, const float *x) override;

    // `x` and `ids` need to be resident on the CPU;
    // Handles paged adds if the add set is too large;
    void add_with_ids(idx_t n, const float *x, const idx_t *ids) override;

    size_t remove_ids(const faiss::IDSelector &sel) override;

    void reset() override;

    void loadAllData(const char *dataPath);

    void saveAllData(const char *dataPath);

    // AscendIndex object is NON-copyable
    AscendIndexIVFSP(const AscendIndexIVFSP&) = delete;
    AscendIndexIVFSP& operator=(const AscendIndexIVFSP&) = delete;

    void setVerbose(bool verbose);

    void setNumProbes(int nprobes);

    void search(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels,
        const SearchParameters *params = nullptr) const override;

    void search_with_filter(idx_t n, const float *x, idx_t k,
                            float *distances, idx_t *labels, const void *filters) const override;

private:
    AscendIndexIVFSP();

    std::shared_ptr<AscendIndexIVFSPImpl> impl_;
};

}
}
#endif