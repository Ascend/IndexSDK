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


#include <gtest/gtest.h>
#include <faiss/ascend/AscendIndexIVFSQ.h>
#include <faiss/ascend/custom/AscendIndexIVFSQFuzzy.h>
#include <faiss/ascend/custom/AscendIndexIVFSQC.h>
#include <faiss/ascend/AscendCloner.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/index_io.h>

namespace {
void Norm(float *data, int n, int dim)
{
#pragma omp parallel for if (n > 1)
    for (int i = 0; i < n; ++i) {
        float l2norm = 0;
        for (int j = 0; j < dim; ++j) {
            l2norm += data[i * dim + j] * data[i * dim + j];
        }
        l2norm = sqrt(l2norm);

        for (int j = 0; j < dim; ++j) {
            data[i * dim + j] = data[i * dim + j] / l2norm;
        }
    }
}

TEST(TestAscendIndexIVFSQC, All)
{
    int dimIn = 256;
    int dimOut = 64;
    int ntotal = 125000;
    int ncentroids = 8192;

    std::vector<float> data(dimIn * ntotal);
    for (int i = 0; i < dimIn * ntotal; i++) {
        data[i] = drand48();
    }
    Norm(data.data(), ntotal, dimIn);
    faiss::ascend::AscendIndexIVFSQCConfig conf({ 0 });
    conf.cp.niter = 16;
    faiss::ascend::AscendIndexIVFSQC index(dimIn, dimOut, ncentroids, faiss::ScalarQuantizer::QuantizerType::QT_8bit,
        faiss::METRIC_INNER_PRODUCT, conf);
    index.verbose = true;
    index.setFuzzyK(2);
    index.setThreshold(1.05);
    index.train(ntotal / 10, data.data());

    for (int i = 0; i < ncentroids; i++) {
        int len = index.getListLength(i);
        ASSERT_EQ(len, 0);
    }

    index.add(ntotal, data.data());
    int realTotal = index.ntotal;
    {
        int tmpTotal = 0;
        for (int i = 0; i < ncentroids; i++) {
            tmpTotal += index.getListLength(i);
        }
        ASSERT_EQ(tmpTotal, realTotal * 2); // 2 is FuzzyK
    }

    std::vector<std::vector<uint8_t>> codes(ncentroids, std::vector<uint8_t>());
    std::vector<std::vector<faiss::ascend::ascend_idx_t>> indices(ncentroids,
        std::vector<faiss::ascend::ascend_idx_t>());
    for (int i = 0; i < ncentroids; i++) {
        index.getListCodesAndIds(i, codes[i], indices[i]);
        ASSERT_EQ(codes[i].size(), indices[i].size() * dimOut);
    }

    index.reset();
    for (int i = 0; i < ncentroids; i++) {
        int len = index.getListLength(i);
        ASSERT_EQ(len, 0);
    }

    index.reserveMemory(ntotal);
    index.add(ntotal, data.data());
    {
        int tmpTotal = 0;
        for (int i = 0; i < ncentroids; i++) {
            tmpTotal += index.getListLength(i);
        }
        ASSERT_EQ(tmpTotal, realTotal * 2); // 2 is FuzzyK
    }

    for (int i = 0; i < ncentroids; i++) {
        std::vector<uint8_t> tmpCode;
        std::vector<faiss::ascend::ascend_idx_t> tmpIndi;
        index.getListCodesAndIds(i, tmpCode, tmpIndi);
        ASSERT_EQ(tmpCode.size(), tmpIndi.size() * dimOut) << "failure index:" << i;
        ASSERT_EQ(tmpCode.size(), codes[i].size()) << "failure index:" << i;
        ASSERT_EQ(tmpIndi.size(), indices[i].size()) << "failure index:" << i;
        ASSERT_EQ(memcmp(tmpCode.data(), codes[i].data(), indices[i].size()), 0) << "failure index:" << i;
        ASSERT_EQ(memcmp(tmpIndi.data(), indices[i].data(), indices[i].size() * sizeof(faiss::ascend::ascend_idx_t)),
            0) << "failure index:" << i;
    }

    {
        int n = 1;
        int k = 1000;
        for (int i = 3; i < 10; i++) {
            std::vector<float> dist(k, 0);
            std::vector<faiss::idx_t> label(k, 0);
            index.search(n, data.data() + i * dimIn, k, dist.data(), label.data());
            ASSERT_EQ(label[0], i);

            faiss::idx_t assign;
            index.assign(1, data.data() + i * dimIn, &assign);
            ASSERT_EQ(assign, i);
        }
    }
}

TEST(TestAscendIndexIVFSQC, CloneAscend2CPU)
{
    int dimIn = 256;
    int dimOut = 64;
    int ntotal = 125000;
    int ncentroids = 8192;
    int nprobe = 64;

    srand48(0);
    std::vector<float> data(dimIn * ntotal);
    for (int i = 0; i < dimIn * ntotal; i++) {
        data[i] = drand48();
    }

    // ascend index
    faiss::ascend::AscendIndexIVFSQCConfig conf({ 0 });
    faiss::ascend::AscendIndexIVFSQC index(dimIn, dimOut, ncentroids, faiss::ScalarQuantizer::QuantizerType::QT_8bit,
        faiss::METRIC_INNER_PRODUCT, conf);
    index.verbose = true;
    index.setFuzzyK(2);
    index.setThreshold(1);
    index.setNumProbes(nprobe);
    EXPECT_EQ(index.getNumProbes(), nprobe);

    // real training data is divided by 10
    index.train(ntotal / 10, data.data());
    // add data
    index.reserveMemory(ntotal);
    index.add(ntotal, data.data());

    // write index with cpu index
    faiss::Index *cpuIndex = faiss::ascend::index_ascend_to_cpu(&index);
    ASSERT_FALSE(cpuIndex == nullptr);
    const char *outfilename = "./ivfsqctest.faiss";
    write_index(cpuIndex, outfilename);

    faiss::IndexIVFScalarQuantizer *cpuIvfSq = dynamic_cast<faiss::IndexIVFScalarQuantizer *>(cpuIndex);
    EXPECT_EQ(index.metric_type, cpuIvfSq->metric_type);
    EXPECT_EQ(index.d, cpuIvfSq->d);
    EXPECT_EQ(index.ntotal, cpuIvfSq->ntotal);
    EXPECT_EQ(index.is_trained, cpuIvfSq->is_trained);
    EXPECT_EQ(index.getNumLists(), cpuIvfSq->nlist);
    EXPECT_EQ(index.getNumProbes(), cpuIvfSq->nprobe);
    delete cpuIndex;
}


TEST(TestAscendIndexIVFSQC, CloneCPU2Ascend)
{
    int dimIn = 256;
    int dimOut = 64;
    int ntotal = 125000;
    int ncentroids = 8192;
    int nprobe = 64;

    srand48(0);
    std::vector<float> data(dimIn * ntotal);
    for (int i = 0; i < dimIn * ntotal; i++) {
        data[i] = drand48();
    }

    const char *indexfilename = "./ivfsqctest.faiss";
    faiss::Index *initIndex = faiss::read_index(indexfilename);
    ASSERT_FALSE(initIndex == nullptr);
    faiss::IndexIVFScalarQuantizer *cpuIndex = dynamic_cast<faiss::IndexIVFScalarQuantizer *>(initIndex);

    // ascend index
    std::vector<int> devices = { 1 };
    faiss::ascend::AscendIndexIVFSQ *ascendIndex =
        dynamic_cast<faiss::ascend::AscendIndexIVFSQ *>(faiss::ascend::index_cpu_to_ascend(devices, initIndex));
    ASSERT_FALSE(ascendIndex == nullptr);

    EXPECT_EQ(ascendIndex->d, dimIn);
    EXPECT_EQ(ascendIndex->getNumLists(), ncentroids);
    EXPECT_EQ(ascendIndex->getNumProbes(), nprobe);
    EXPECT_EQ(ascendIndex->metric_type, cpuIndex->metric_type);
    EXPECT_EQ(ascendIndex->d, cpuIndex->d);
    EXPECT_EQ(ascendIndex->ntotal, cpuIndex->ntotal);
    EXPECT_EQ(ascendIndex->is_trained, cpuIndex->is_trained);
    EXPECT_EQ(ascendIndex->getNumLists(), cpuIndex->nlist);
    EXPECT_EQ(ascendIndex->getNumProbes(), cpuIndex->nprobe);

    delete ascendIndex;
    delete initIndex;
}
} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
