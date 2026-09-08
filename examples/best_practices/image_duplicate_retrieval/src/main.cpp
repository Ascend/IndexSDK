/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

#include <faiss/ascend/AscendIndexFlat.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using Clock = std::chrono::steady_clock;

constexpr int kDevice = 0;
constexpr int kTopK = 100;
constexpr size_t kDimension = 128;
constexpr size_t kBaseRows = 100000000;
constexpr size_t kQueryRows = 1000000;
constexpr size_t kSearchBatchRows = 64;
constexpr size_t kBaseChunkRows = 1000000;
constexpr int64_t kResourceSize = 2LL * 1024 * 1024 * 1024;

std::filesystem::path ParseDataDir(int argc, char **argv)
{
    if (argc == 1)
    {
        return "tmp";
    }
    if (argc == 3 && std::string(argv[1]) == "--data-dir")
    {
        return argv[2];
    }
    throw std::invalid_argument("usage: image_dedup_business [--data-dir PATH]");
}

class FvecsReader
{
   public:
    explicit FvecsReader(const std::string &path) : path_(path), input_(path, std::ios::binary | std::ios::ate)
    {
        if (!input_)
        {
            throw std::runtime_error("failed to open " + path_);
        }
        const std::streamoff fileBytes = input_.tellg();
        input_.seekg(0);
        int32_t dimension = 0;
        input_.read(reinterpret_cast<char *>(&dimension), sizeof(dimension));
        if (!input_ || dimension <= 0)
        {
            throw std::runtime_error("invalid fvecs file " + path_);
        }
        dimension_ = static_cast<size_t>(dimension);
        recordBytes_ = (dimension_ + 1) * sizeof(float);
        if (fileBytes <= 0 || static_cast<size_t>(fileBytes) % recordBytes_ != 0)
        {
            throw std::runtime_error("invalid fvecs file " + path_);
        }
        rows_ = static_cast<size_t>(fileBytes) / recordBytes_;
        input_.seekg(0);
    }

    size_t Rows() const { return rows_; }

    size_t Dimension() const { return dimension_; }

    void Read(size_t rows, std::vector<float> &values)
    {
        values.resize(rows * (dimension_ + 1));
        input_.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(rows * recordBytes_));
        if (!input_)
        {
            throw std::runtime_error("failed to read " + path_);
        }
        for (size_t row = 0; row < rows; ++row)
        {
            std::memmove(values.data() + row * dimension_, values.data() + row * (dimension_ + 1) + 1,
                         dimension_ * sizeof(float));
        }
        values.resize(rows * dimension_);
    }

   private:
    std::string path_;
    std::ifstream input_;
    size_t rows_ = 0;
    size_t dimension_ = 0;
    size_t recordBytes_ = 0;
};

void WriteIvecs(std::ofstream &output, const std::vector<faiss::idx_t> &labels, size_t queryCount)
{
    std::vector<int32_t> record(kTopK + 1);
    record[0] = kTopK;
    for (size_t query = 0; query < queryCount; ++query)
    {
        for (int rank = 0; rank < kTopK; ++rank)
        {
            record[rank + 1] = static_cast<int32_t>(labels[query * kTopK + rank]);
        }
        output.write(reinterpret_cast<const char *>(record.data()),
                     static_cast<std::streamsize>(record.size() * sizeof(int32_t)));
    }
}
}  // namespace

int main(int argc, char **argv)
{
    try
    {
        const std::filesystem::path dataDir = ParseDataDir(argc, argv);
        const std::string basePath = (dataDir / "base.fvecs").string();
        const std::string queryPath = (dataDir / "query.fvecs").string();
        const std::string outputPath = (dataDir / "business-results/output.ivecs").string();
        const std::string performancePath = (dataDir / "business-results/performance.csv").string();
        FvecsReader database(basePath);
        FvecsReader queryInput(queryPath);
        if (database.Rows() != kBaseRows || queryInput.Rows() != kQueryRows || database.Dimension() != kDimension ||
            queryInput.Dimension() != kDimension)
        {
            throw std::runtime_error("unexpected business data shape");
        }

        faiss::ascend::AscendIndexFlatConfig config({kDevice}, kResourceSize);
        faiss::ascend::AscendIndexFlat index(static_cast<int>(database.Dimension()), faiss::METRIC_INNER_PRODUCT,
                                             config);
        {
            std::vector<float> baseChunk;
            size_t added = 0;
            while (added < database.Rows())
            {
                const size_t rows = std::min(kBaseChunkRows, database.Rows() - added);
                database.Read(rows, baseChunk);
                index.add(static_cast<faiss::idx_t>(rows), baseChunk.data());
                added += rows;
            }
        }

        std::ofstream output(outputPath, std::ios::binary);
        if (!output)
        {
            throw std::runtime_error("failed to open " + outputPath);
        }
        std::vector<float> queries;
        std::vector<float> distances(kSearchBatchRows * kTopK);
        std::vector<faiss::idx_t> labels(kSearchBatchRows * kTopK);
        double seconds = 0.0;
        for (size_t startRow = 0; startRow < queryInput.Rows(); startRow += kSearchBatchRows)
        {
            const size_t rows = std::min(kSearchBatchRows, queryInput.Rows() - startRow);
            queryInput.Read(rows, queries);
            const Clock::time_point start = Clock::now();
            index.search(static_cast<faiss::idx_t>(rows), queries.data(), kTopK, distances.data(), labels.data());
            seconds += std::chrono::duration<double>(Clock::now() - start).count();
            WriteIvecs(output, labels, rows);
        }
        output.close();
        if (!output)
        {
            throw std::runtime_error("failed to write " + outputPath);
        }

        std::ofstream performance(performancePath);
        performance << "algorithm,n,device_count,qps\nflat," << database.Rows() << ",1," << std::setprecision(10)
                    << static_cast<double>(queryInput.Rows()) / seconds << '\n';
        performance.close();
        if (!performance)
        {
            throw std::runtime_error("failed to write " + performancePath);
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
