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


#include <string>
#include <gtest/gtest.h>
#include <faiss/ascend/AscendIndexIVFSP.h>
#include <faiss/ascend/AscendMultiIndexSearch.h>
#include "common.h"

using namespace std;
using namespace faiss::ascend;

namespace {
struct IVFSPParams {
    int dim;
    int nonzeroNum;
    int nlist;
    int nprobe;
    int handleBatch;
    int searchListSize;
    string codeBookPath;
};

shared_ptr<AscendIndexIVFSP> CreateIVFSP(const IVFSPParams &params)
{
    faiss::ascend::AscendIndexIVFSPConfig conf({ 0 });
    conf.nprobe = params.nprobe;
    conf.handleBatch = params.handleBatch;
    conf.searchListSize = params.searchListSize;

    return make_shared<AscendIndexIVFSP>(params.dim, params.nonzeroNum, params.nlist,
        params.codeBookPath.c_str(), faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType::METRIC_L2, conf);
}

shared_ptr<AscendIndexIVFSP> CreateIVFSP(const IVFSPParams &params, AscendIndexIVFSP &sharedIndex)
{
    faiss::ascend::AscendIndexIVFSPConfig conf({ 0 });
    conf.nprobe = params.nprobe;
    conf.handleBatch = params.handleBatch;
    conf.searchListSize = params.searchListSize;

    return make_shared<AscendIndexIVFSP>(params.dim, params.nonzeroNum, params.nlist,
        sharedIndex, faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::MetricType::METRIC_L2, conf);
}

void CheckSearchRecall(shared_ptr<AscendIndexIVFSP> ivfsp, TestData &feature)
{
    int topK = 100;
    shapeMap &dataShape = feature.dataShape;
    std::vector<float> dist(dataShape["query"] * topK, 0);
    std::vector<faiss::idx_t> label(dataShape["query"] * topK, 0);

    std::vector<float> &query = feature.query;
    ivfsp->search(dataShape["query"], query.data(), topK, dist.data(), label.data());

    std::vector<int64_t> &gt = feature.gt;
    auto Recall = calNewRecall(label, gt.data(), dataShape["query"]);
    printf("ivfsp:%p, r1 = %.2f, r10 = %.2f, r100 = %.2f\n", ivfsp.get(),
        Recall[1], Recall[10], Recall[100]);
}

void AddFeatures(shared_ptr<AscendIndexIVFSP> ivfsp, TestData &feature)
{
    std::vector<float> &base = feature.base;
    shapeMap &dataShape = feature.dataShape;
    ivfsp->add(dataShape["base"], base.data());
}

void AddFeatures(shared_ptr<AscendIndexIVFSP> ivfsp, int n, TestData &feature)
{
    std::vector<float> &base = feature.base;
    ivfsp->add(n, base.data());
}

// python3 ivfsp_generate_model.py --cores 8 -d 256 -nonzero_num 64 -nlist 256 -handle_batch 64 -code_num 32768
IVFSPParams param1 {
    .dim = 256,
    .nonzeroNum = 64,
    .nlist = 256,
    .nprobe = 64,
    .handleBatch = 64,
    .searchListSize = 32768,
    .codeBookPath = "/home/ivfsp_codebooks/codebook_256_64_256.bin"
};

// python3 ivfsp_generate_model.py --cores 8 -d 256 -nonzero_num 32 -nlist 512 -handle_batch 64 -code_num 32768
IVFSPParams param2 {
    .dim = 256,
    .nonzeroNum = 32,
    .nlist = 512,
    .nprobe = 64,
    .handleBatch = 64,
    .searchListSize = 32768,
    .codeBookPath = "/home/ivfsp_codebooks/codebook_dir_256_32_512_32768/codebook_256_32_512.bin"
};

// python3 ivfsp_generate_model.py --cores 8 -d 256 -nonzero_num 16 -nlist 256 -handle_batch 64 -code_num 16384
IVFSPParams param3 {
    .dim = 256,
    .nonzeroNum = 16,
    .nlist = 256,
    .nprobe = 64,
    .handleBatch = 64,
    .searchListSize = 16384,
    .codeBookPath = "/home/ivfsp_codebooks/codebook_dir_256_16_256_16384/codebook_256_16_256.bin"
};

TestData &InitFeature() {
    static bool flag = true;
    static TestData feature = readBinFeature("/home/1230_largetest_data/webfaceLPS2m_bin/");
    return feature;
}

// 1. IVFSP1 add
// 2. IVFSP2 add  
// 3. IVFSP1 search
// 4. IVFSP3 add search
// 5. IVFSP2 search
TEST(TestAscendIndexIVFSP, diff_index_add_search_down)
{
    // 测试算子是否齐全，避免后续跑一半跑异常心态爆炸
    {
        auto ivfsp = CreateIVFSP(param1);
        cout << "ivfsp1 op down." << endl;
        ivfsp = CreateIVFSP(param2);
        cout << "ivfsp2 op down." << endl;
        ivfsp = CreateIVFSP(param3);
        cout << "ivfsp3 op down." << endl;
    }

    auto feature = InitFeature();

    // 1. 创建IVFSP1 add
    auto ivfsp1 = CreateIVFSP(param1);
    cout << "CreateIVFSP 1 down." << endl;
    AddFeatures(ivfsp1, feature);
    cout << "AddFeatures down." << endl;

    // 2. 创建IVFSP2 add
    auto ivfsp2 = CreateIVFSP(param2);
    cout << "CreateIVFSP 2 down." << endl;
    AddFeatures(ivfsp2, feature);
    cout << "AddFeatures down." << endl;

    // 3. IVFSP1 search
    CheckSearchRecall(ivfsp1, feature);
    cout << "CheckSearchRecall down." << endl;

    // 4. IVFSP3 add search
    auto ivfsp3 = CreateIVFSP(param3);
    cout << "CreateIVFSP3 down." << endl;
    AddFeatures(ivfsp3, feature);
    cout << "AddFeatures down." << endl;
    CheckSearchRecall(ivfsp3, feature);
    cout << "CheckSearchRecall down." << endl;

    // 5. IVFSP2 search
    CheckSearchRecall(ivfsp2, feature);
    cout << "CheckSearchRecall down." << endl;
}

TEST(TestAscendIndexIVFSP, multi_index_share_codebook)
{
    auto feature = InitFeature();

    auto ivfsp1 = CreateIVFSP(param1);
    vector<shared_ptr<AscendIndexIVFSP>> sharedIndexVec;
    for (int i = 0; i < 10; i++) {
        auto ivfsp = CreateIVFSP(param1, *ivfsp1);
        sharedIndexVec.emplace_back(ivfsp);
        int ntotal = 20000 * (i + 1);
        AddFeatures(ivfsp, ntotal, feature);
        printf("index[%d] AddFeatures down\r\n", i);
    }

    vector<faiss::Index *> indexVec;
    for (auto shared : sharedIndexVec) {
        indexVec.emplace_back(shared.get());
    }

    int batch = 49;
    int topK = 10;
    auto indexNum = indexVec.size();
    std::vector<float> multiDist(batch * topK * indexNum, 0);
    std::vector<faiss::idx_t> multiLabel(batch * topK * indexNum, 0);
    std::vector<float> &query = feature.query;
    Search(indexVec, batch, query.data(), topK, multiDist.data(), multiLabel.data(), false);
    for (size_t i = 0; i < indexNum; i++) {
        std::vector<float> dist(batch * topK, 0);
        std::vector<faiss::idx_t> label(batch * topK, 0);
        sharedIndexVec[i]->search(batch, query.data(), topK, dist.data(), label.data());
        for (size_t j = 0; j < dist.size(); j++) {
            if (label[j] == multiLabel[i * batch * topK + j]) {
                continue;
            }
            if (dist[j] == multiDist[i * batch * topK + j]) {
                continue;
            }
            printf("Error! index[%zu][%zu] dist[%f] multiDist[%f] label[%lld] multiLabel[%lld]\r\n", i, j,
                dist[j], multiDist[i * batch * topK + j], label[j], multiLabel[i * batch * topK + j]);
            EXPECT_TRUE(false);
            break;
        }
    }
}

TEST(TestAscendIndexIVFSP, can_not_shared_not_match_codebook)
{
    auto ivfsp1 = CreateIVFSP(param1);
    string msg;
    try {
        auto ivfsp2 = CreateIVFSP(param2, *ivfsp1);
        cout << "error! can_not_shared_not_match_codebook!" << endl;
    } catch (exception &e) {
        msg = e.what();
    }

    EXPECT_FALSE(msg.empty());
}

size_t physical_memory_used_by_process()
{
    FILE* file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];

    while (fgets(line, 128, file) != nullptr) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            int len = strlen(line);

            const char* p = line;
            for (; isdigit(*p) == false; ++p) {}

            line[len - 3] = 0;
            result = atoi(p);

            break;
        }
    }

    fclose(file);

    return result;
}

TEST(TestAscendIndexIVFSP, shared_codebook_can_not_use_much_mem)
{
    auto ivfsp1 = CreateIVFSP(param1);
    auto mem = physical_memory_used_by_process();
    vector<shared_ptr<AscendIndexIVFSP>> vec;
    printf("start create ivfsp\r\n");
    for (int i = 0; i < 10000; i++) {
        vec.emplace_back(CreateIVFSP(param1, *ivfsp1));
        if (i % 1000 == 0) {
            auto memTemp = physical_memory_used_by_process() - mem;
            printf("index[%d] mem[%zu]M\r\n", i, memTemp / 1024);
        }
    }
    mem = physical_memory_used_by_process() - mem;
    printf("total ivfsp [%zu]M\r\n", mem / 1024);
}

} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
