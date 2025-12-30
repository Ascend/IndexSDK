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

#include <sys/time.h>
#include <securec.h>
#include <fstream>

#include <gtest/gtest.h>
#include <faiss/ascend/AscendNNInference.h>
#include <faiss/ascend/AscendIndexFlat.h>


namespace {
inline double GetMillisecs()
{
    struct timeval tv = { 0, 0 };
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

TEST(TestAscendNNInference, All)
{
    int dimIn = 128;
    int dimOut = 32;
    int batchSize = 32;
    int ntotal = 1000000;
    std::vector<int> deviceList = { 0 };
    std::string modelPath = "./nnDimReduction.om"; // Converted from neural network model
    int maxSize = ntotal * dimIn;

    std::vector<float> data(maxSize);
    std::vector<float> outputData(ntotal * dimOut);
    for (int i = 0; i < maxSize; i++) {
        data[i] = drand48();
    }

    // read model file
    std::ifstream istrm(modelPath.c_str(), std::ios::binary);
    std::stringstream buffer;
    buffer << istrm.rdbuf();
    std::string nnom(buffer.str());
    istrm.close();

    faiss::ascend::AscendNNInference dimInference(deviceList, nnom.data(), nnom.size());
    printf("dimIn:%d, dimOut:%d, batchSize:%d, inputType:%d, outputType:%d\n", dimInference.getDimIn(),
        dimInference.getDimOut(), dimInference.getDimBatch(), dimInference.getInputType(),
        dimInference.getOutputType());

    ASSERT_EQ(dimInference.getDimIn(), dimIn);
    ASSERT_EQ(dimInference.getDimOut(), dimOut);
    ASSERT_EQ(dimInference.getDimBatch(), batchSize);

    double ts = GetMillisecs();
    dimInference.infer(ntotal, (char *)data.data(), (char *)outputData.data());
    double te = GetMillisecs();
    printf("deviceCount:%d, ntotal:%d, dimIn:%d, dimOut:%d, batchSize:%d, QPS:%.4f\n", deviceList.size(), ntotal, dimIn,
        dimOut, batchSize, 1000.0 * ntotal / (te - ts));
}

static std::unique_ptr<float[]> FvecsRead(const std::string &fname, size_t &d_out, size_t &n_out)
{
    FILE *f = fopen(fname.c_str(), "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname.c_str());
        perror("");
        abort();
    }
    int d;
    size_t nr = fread(&d, 1, sizeof(int), f);
    static_cast<void>(nr); // unused variable nr
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    size_t n = sz / ((d + 1) * sizeof(float));

    d_out = d;
    n_out = n;

    std::unique_ptr<float[]> x(new float[n * (d + 1)]);
    float *pX = x.get();
    nr = fread(pX, sizeof(float), n * (d + 1), f);
    static_cast<void>(nr); // unused variable nr

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++) {
        auto err = memmove_s(pX + i * d, (n * (d + 1) - i * d) * sizeof(*pX), pX + 1 + i * (d + 1), d * sizeof(*pX));
    }

    fclose(f);
    return x;
}

TEST(TestAscendNNInference, Sift1M)
{
    std::vector<float> benchmark = { 0.3969, 0.8100, 0.9827 };
    double t0 = GetMillisecs();
    if (access("sift1M", F_OK) == -1) {
        std::cout << "sift1M not exist" << std::endl;
        FAIL();
    }
    printf("[%.3f s] Loading database\n", GetMillisecs() - t0);
    size_t ntotal, dimIn;
    size_t dimOut = 32;

    auto pBase = FvecsRead("sift1M/sift_base.fvecs", dimIn, ntotal);
    const float *xb = pBase.get();
    std::vector<float> baseBeforeReduction;
    std::vector<float> baseAfterReduction(ntotal * dimOut);
    baseBeforeReduction.insert(baseBeforeReduction.end(), xb, xb + ntotal * dimIn);
    for (int i = 0; i < baseBeforeReduction.size(); i++) {
        baseBeforeReduction[i] /= 128.0;
    }
    faiss::ascend::AscendIndexFlatConfig conf({ 0 });
    faiss::ascend::AscendIndexFlat index(dimOut, faiss::METRIC_L2, conf);

    std::string modelPath = "./nnDimReduction.om"; // 128->32 the best

    // read model file
    std::ifstream istrm(modelPath.c_str(), std::ios::binary);
    std::stringstream buffer;
    buffer << istrm.rdbuf();
    std::string nnom(buffer.str());
    istrm.close();

    faiss::ascend::AscendNNInference ReductionDim(conf.deviceList, nnom.data(), nnom.size());
    ReductionDim.infer(ntotal, (char *)baseBeforeReduction.data(), (char *)baseAfterReduction.data());

    size_t nq;
    size_t d2;
    auto pQuery = FvecsRead("sift1M/sift_query.fvecs", d2, nq);
    const float *xq = pQuery.get();
    std::vector<float> queryBeforeReduction;
    std::vector<float> queryAfterReduction(nq * dimOut);
    queryBeforeReduction.insert(queryBeforeReduction.end(), xq, xq + nq * d2);
    for (int i = 0; i < queryBeforeReduction.size(); i++) {
        queryBeforeReduction[i] /= 128.0;
    }
    ReductionDim.infer(nq, (char *)queryBeforeReduction.data(), (char *)queryAfterReduction.data());

    for (auto deviceId : conf.deviceList) {
        int len = index.getBaseSize(deviceId);
        ASSERT_EQ(len, 0);
    }

    index.add(ntotal, baseAfterReduction.data());
    {
        int getTotal = 0;
        for (size_t i = 0; i < conf.deviceList.size(); i++) {
            int tmpTotal = index.getBaseSize(conf.deviceList[i]);
            getTotal += tmpTotal;
        }
        EXPECT_EQ(getTotal, ntotal);
    }
    size_t k; // nb of results per query in the GT
    size_t nq2;
    // not very clean, but works as long as sizeof(int) == sizeof(float)
    auto pGT = FvecsRead("sift1M/sift_groundtruth.ivecs", k, nq2);
    auto gt = reinterpret_cast<const int32_t *>(pGT.get());

    int searchK = 100;
    std::vector<float> dist(nq * searchK, 0);
    std::vector<faiss::idx_t> label(nq * searchK, 0);
    double startTime = GetMillisecs();
    index.search(nq, queryAfterReduction.data(), searchK, dist.data(), label.data());
    double costTime = GetMillisecs() - startTime;
    printf("[%.3f ms] Compute recalls, qps = %.3f\n", GetMillisecs() - t0, (nq * 1000.0 / costTime));
    // evaluate result by hand.
    int n1 = 0, n10 = 0, n100 = 0;
    for (size_t i = 0; i < nq; i++) {
        faiss::idx_t gtNn = gt[i * k];
        for (size_t j = 0; j < k; j++) {
            if (label[i * k + j] != gtNn) {
                continue;
            }

            if (j < 1) {
                n1++;
            }
            if (j < 10) {
                n10++;
            }
            if (j < 100) {
                n100++;
            }
        }
    }
    float r1Recall = n1 / static_cast<float>(nq);
    printf("R@1 = %.4f\n", r1Recall);
    EXPECT_TRUE(r1Recall == benchmark[0]);
    float r10Recall = n10 / static_cast<float>(nq);
    printf("R@10 = %.4f\n", r10Recall);
    EXPECT_TRUE(r10Recall == benchmark[1]);
    float r100Recall = n100 / static_cast<float>(nq);
    printf("R@100 = %.4f\n", r100Recall);
    EXPECT_TRUE(r100Recall == benchmark[2]);
}
} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
